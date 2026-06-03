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
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReplySegment:
    """A parsed segment of LLM output, ready to send."""

    text: str
    reply_message_id: int | None = None
    should_recall: bool = False


class ReplyManager:
    """Manages message caching, prompt building, and reply tag parsing."""

    def __init__(
        self,
        cache_ttl: int = 600,
        cache_max_per_session: int = 50,
        prompt_template: str = "",
        recall_tag: str = "[recall]",
        segment_separator: str = "",
    ):
        self.cache_ttl = max(1, min(cache_ttl, 86400))
        self.cache_max = max(1, min(cache_max_per_session, 500))
        self.prompt_template = prompt_template
        self.recall_tag = recall_tag
        self.segment_separator = segment_separator

        self._cache: dict[str, deque[CachedMessage]] = {}
        self._reply_tag_pattern = re.compile(r"\[reply:(\d+)\]")

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

    def cache_message(self, session_key: str, message_id: int, text: str) -> None:
        """Add a message to the reply cache with TTL eviction."""
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
            CachedMessage(message_id=message_id, text=text, timestamp=time.time())
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
        prompt = self.prompt_template.replace("{reply_tag}", "[reply:ID]")
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
            lines.append(f"[{item.message_id}] {truncated}")
        return "\n".join(lines)

    # ==================== Reply Tag Parsing ====================

    def has_reply_tags(self, text: str) -> bool:
        """Check if text contains any [reply:ID] tags."""
        return bool(self._reply_tag_pattern.search(text))

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
                        if seg_text:
                            segments.append(
                                ReplySegment(
                                    text=seg_text,
                                    reply_message_id=None,
                                    should_recall=should_recall,
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
                        if seg_text:
                            segments.append(
                                ReplySegment(
                                    text=seg_text,
                                    # Only the first sub-segment gets the Reply component
                                    reply_message_id=(target_id if valid_id else None)
                                    if idx == 0
                                    else None,
                                    should_recall=should_recall,
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
