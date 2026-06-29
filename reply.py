"""Reply feature module: cache user messages and handle [reply:ID] tags.

This module provides QQ native reply (quote) functionality by:
1. Caching original user messages with their message_id before debounce merging
2. Building prompt context with quotable message list for LLM
3. Parsing [reply:ID] tags from LLM output and sending with Reply components
"""

import re
import time
from collections import deque
from dataclasses import dataclass, field

from astrbot.api import logger
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


@dataclass
class CachedMessage:
    """A cached user message for reply targeting."""

    message_id: int
    text: str  # Display text for prompt (may include media placeholders like [图片])
    sender: str = ""  # Optional sender label (used in group chats to disambiguate)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReplySegment:
    """A parsed segment of LLM output, ready to send."""

    text: str
    reply_message_id: int | None = None
    should_recall: bool = False
    # When True, a poke action should be fired right after this segment is sent.
    # A segment may carry should_poke with empty text (a [poke] on its own line),
    # in which case only the poke fires and no message is sent for it.
    should_poke: bool = False


class ReplyManager:
    """Manages message caching, prompt building, and reply tag parsing."""

    def __init__(
        self,
        cache_ttl: int = 600,
        cache_max_per_session: int = 50,
        prompt_template: str = "",
        reply_tag: str = "[reply:{id}]",
        recall_tag: str = "[recall]",
        segment_separator: str = "",
        poke_tag: str = "[poke]",
    ):
        self.cache_ttl = max(1, min(cache_ttl, 86400))
        self.cache_max = max(1, min(cache_max_per_session, 500))
        self.prompt_template = prompt_template
        self.reply_tag = reply_tag
        self.recall_tag = recall_tag
        self.segment_separator = segment_separator
        self.poke_tag = poke_tag

        self._cache: dict[str, deque[CachedMessage]] = {}
        # Build dynamic regex pattern from reply_tag config
        self._reply_tag_pattern = self._build_reply_pattern(reply_tag)
        # Build fallback patterns for common malformed variants (case-insensitive, negative IDs)
        self._reply_fallback_patterns = [
            re.compile(r'\[reply:-?\d+\]', re.IGNORECASE),
            re.compile(r'<reply:-?\d+>', re.IGNORECASE),
            re.compile(r'【reply:-?\d+】', re.IGNORECASE),
            re.compile(r'\(reply:-?\d+\)', re.IGNORECASE),
            re.compile(r'\[引用:-?\d+\]', re.IGNORECASE),
        ]

    def _build_reply_pattern(self, tag_template: str) -> re.Pattern:
        """Build regex pattern from reply_tag template.
        
        Examples:
            [reply:{id}]  -> \[reply:(\d+)\]
            <reply:{id}>  -> <reply:(\d+)>
            【reply:{id}】 -> 【reply:(\d+)】
        
        Allows negative IDs and is case-insensitive to catch common LLM mistakes.
        """
        if '{id}' not in tag_template:
            logger.warning(
                f"[ActiveFunction] Invalid reply_tag template '{tag_template}': "
                f"missing {{id}} placeholder, falling back to [reply:(\\d+)]"
            )
            return re.compile(r'\[reply:(\d+)\]', re.IGNORECASE)
        
        try:
            # Escape special regex characters
            escaped = re.escape(tag_template)
            # Replace escaped {id} with capture group for any integer (including negative)
            pattern_str = escaped.replace(r'\{id\}', r'(-?\d+)')
            return re.compile(pattern_str, re.IGNORECASE)
        except re.error as e:
            logger.warning(
                f"[ActiveFunction] Failed to build reply_tag regex from '{tag_template}': {e}, "
                f"falling back to [reply:(\\d+)]"
            )
            return re.compile(r'\[reply:(\d+)\]', re.IGNORECASE)

    # ==================== Message Caching ====================

    def extract_message_id(self, event: AiocqhttpMessageEvent) -> int | None:
        """Extract numeric message_id from the event object."""
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return None

        # Try message_obj.message_id first
        mid = getattr(msg_obj, "message_id", None)
        if mid is not None:
            mid_str = str(mid)
            if mid_str.lstrip("-").isdigit():
                return int(mid_str)

        # Try raw_message dict/object
        raw = getattr(msg_obj, "raw_message", None)
        if raw is not None:
            raw_mid = None
            if isinstance(raw, dict):
                raw_mid = raw.get("message_id")
            else:
                raw_mid = getattr(raw, "message_id", None)
                if raw_mid is None:
                    try:
                        raw_mid = raw["message_id"]
                    except Exception:
                        pass
            if raw_mid is not None:
                raw_mid_str = str(raw_mid)
                if raw_mid_str.lstrip("-").isdigit():
                    return int(raw_mid_str)

        return None

    def cache_message(
        self, session_key: str, message_id: int, text: str, sender: str = ""
    ) -> None:
        """Add a message to the reply cache with TTL eviction.

        ``sender`` is an optional label (e.g. ``昵称(QQ号)``) shown in the
        quotable-message list so the LLM can tell who said what in group chats.
        """
        text = text.strip()
        if not text:
            return

        if session_key not in self._cache:
            self._cache[session_key] = deque(maxlen=self.cache_max)

        queue = self._cache[session_key]

        # Evict expired entries
        self._prune_cache(session_key)

        # Avoid duplicate entries for the same message_id
        for item in queue:
            if item.message_id == message_id:
                return

        queue.append(
            CachedMessage(
                message_id=message_id,
                text=text,
                sender=sender.strip(),
                timestamp=time.time(),
            )
        )
        logger.debug(
            f"[ActiveFunction] Reply cache: stored mid={message_id}, text={text[:40]}"
        )

    def _prune_cache(self, session_key: str) -> None:
        """Remove expired entries from the cache."""
        queue = self._cache.get(session_key)
        if not queue:
            return
        cutoff = time.time() - self.cache_ttl
        while queue and queue[0].timestamp < cutoff:
            queue.popleft()

    # ==================== Prompt Building ====================

    def build_reply_prompt(self, session_key: str) -> str:
        """Build the reply instruction prompt with cached message list."""
        if not self.prompt_template:
            return ""

        message_list = self._build_message_list(session_key)
        # Replace {reply_tag} with example showing actual configured format
        example_tag = self.reply_tag.replace('{id}', 'ID')
        prompt = self.prompt_template.replace("{reply_tag}", example_tag)
        prompt = prompt.replace("{message_list}", message_list)
        return prompt

    def _build_message_list(self, session_key: str) -> str:
        """Format cached messages as a list for prompt injection."""
        self._prune_cache(session_key)
        queue = self._cache.get(session_key)
        if not queue:
            return "(当前没有可引用的消息)"

        lines = []
        for item in queue:
            truncated = item.text[:100] + ("..." if len(item.text) > 100 else "")
            if item.sender:
                lines.append(f"[{item.message_id}] {item.sender}：{truncated}")
            else:
                lines.append(f"[{item.message_id}] {truncated}")
        return "\n".join(lines)

    # ==================== Reply Tag Parsing ====================

    def has_reply_tags(self, text: str) -> bool:
        """Check if text contains any [reply:ID] tags."""
        return bool(self._reply_tag_pattern.search(text))

    def strip_all_reply_variants(self, text: str) -> str:
        """Strip all known reply tag variants from text (for fallback cleanup).
        
        This catches malformed tags that LLM might produce:
        - Case variations: [Reply:123], [REPLY:123]
        - Negative IDs: [reply:-123]
        - Different brackets: <reply:123>, 【reply:123】, (reply:123)
        - Chinese variants: [引用:123]
        """
        cleaned = text
        # Strip main configured pattern
        cleaned = self._reply_tag_pattern.sub('', cleaned)
        # Strip common fallback variants
        for pattern in self._reply_fallback_patterns:
            cleaned = pattern.sub('', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        return cleaned

    def parse_segments(self, text: str, session_key: str) -> list[ReplySegment]:
        """Parse text into ReplySegments, splitting on [reply:ID] tags.

        Each reply-tagged block is further split by the segment_separator regex,
        so that long replies are sent as multiple messages. The first sub-segment
        carries the Reply component; subsequent sub-segments are plain text.

        Handles both [reply:ID] and recall tags independently per sub-segment.
        """
        # Split by [reply:ID] pattern
        # re.split with capturing group: [leading, id1, text1, id2, text2, ...]
        parts = self._reply_tag_pattern.split(text)

        segments: list[ReplySegment] = []

        i = 0
        while i < len(parts):
            if i == 0:
                # Leading text before any reply tag
                leading = parts[0].strip()
                if leading:
                    # Split first, then check recall per sub-segment
                    sub_segments = self._split_by_separator(leading)
                    for sub in sub_segments:
                        seg_text, should_recall = self._strip_recall_tag(sub)
                        seg_text, should_poke = self._strip_poke_tag(seg_text)
                        if seg_text or should_poke:
                            segments.append(
                                ReplySegment(
                                    text=seg_text,
                                    reply_message_id=None,
                                    should_recall=should_recall and bool(seg_text),
                                    should_poke=should_poke,
                                )
                            )
                i += 1
            else:
                # parts[i] is a captured message_id, parts[i+1] is the text after it
                target_id = int(parts[i])
                reply_text = parts[i + 1].strip() if i + 1 < len(parts) else ""

                # Validate message_id exists in cache
                valid_id = self._validate_reply_id(session_key, target_id)
                if not valid_id:
                    logger.warning(
                        f"[ActiveFunction] Reply ID {target_id} not found in cache, "
                        f"sending as plain text"
                    )

                if reply_text:
                    # Split first, then check recall per sub-segment
                    sub_segments = self._split_by_separator(reply_text)
                    for idx, sub in enumerate(sub_segments):
                        seg_text, should_recall = self._strip_recall_tag(sub)
                        seg_text, should_poke = self._strip_poke_tag(seg_text)
                        if seg_text or should_poke:
                            segments.append(
                                ReplySegment(
                                    text=seg_text,
                                    # Only the first sub-segment gets the Reply component
                                    reply_message_id=(target_id if valid_id else None)
                                    if idx == 0 and seg_text
                                    else None,
                                    should_recall=should_recall and bool(seg_text),
                                    should_poke=should_poke,
                                )
                            )
                i += 2

        return segments

    def _split_by_separator(self, text: str) -> list[str]:
        """Split text using the configured segment_separator regex.

        Uses re.findall (same as AstrBot's segmented_reply behavior).
        Returns the original text as a single-item list if no separator is configured
        or if the regex produces no results.
        """
        if not self.segment_separator:
            return [text]

        try:
            parts = re.findall(self.segment_separator, text, re.DOTALL | re.MULTILINE)
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                return parts
        except re.error as e:
            logger.error(
                f"[ActiveFunction] Reply segment regex error: {e}, falling back to no split"
            )

        return [text]

    def _validate_reply_id(self, session_key: str, message_id: int) -> bool:
        """Check if a message_id exists in the cache."""
        self._prune_cache(session_key)
        queue = self._cache.get(session_key)
        if not queue:
            return False
        return any(item.message_id == message_id for item in queue)

    def get_cached_text(self, session_key: str, message_id: int) -> str | None:
        """Return the cached display text for a message_id, or None if absent/expired.

        Used to render the {quote} placeholder when annotating [reply:ID] tags in
        conversation history.
        """
        self._prune_cache(session_key)
        queue = self._cache.get(session_key)
        if not queue:
            return None
        for item in queue:
            if item.message_id == message_id:
                return item.text
        return None

    def _strip_poke_tag(self, text: str) -> tuple[str, bool]:
        """Strip the poke tag from text and return (cleaned_text, had_poke).

        Used so a [poke] tag is honoured at the position it appears in the LLM
        output: the segment it sits in is sent first, then the poke fires.
        """
        if not self.poke_tag or self.poke_tag not in text:
            return text, False
        return text.replace(self.poke_tag, "").strip(), True

    def _strip_recall_tag(self, text: str) -> tuple[str, bool]:
        """Strip recall tag and [NEXT] wakeup tag from text and return (cleaned_text, should_recall)."""
        if not self.recall_tag:
            should_recall = False
            cleaned = text
        else:
            should_recall = self.recall_tag in text
            cleaned = text.replace(self.recall_tag, "").strip()

        # Strip [NEXT: Xm] wakeup tags to prevent leaking to user
        cleaned = re.sub(r"\[?\s*(?:next|Next|NEXT)\s*(?:[:\uff1a]\s*[^\]]*\s*)?\]?", "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned, should_recall
