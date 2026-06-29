import asyncio
import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain, Record
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from .poke import PokeManager
from .reply import ReplyManager

# Regex to strip [NEXT: Xm] wakeup tags from visible output
_NEXT_TAG_PATTERN = re.compile(r"\[?\s*(?:next|Next|NEXT)\s*(?:[:\uff1a]\s*[^\]]*\s*)?\]?", re.IGNORECASE)


@register(
    "astrbot_plugin_active_function",
    "L1ke40oz",
    "为 Bot 提供主动能力：撤回消息、引用回复、戳一戳等",
    "0.1.0",
)
class ActiveFunctionPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # group support: master toggle for using these features in group chats.
        # When off (default), the plugin only acts in private chats, preserving
        # the original behavior. When on, recall/reply/poke also work in groups
        # (each still gated by its own enable flag).
        group_cfg = config.get("group", {})
        self.group_enable: bool = group_cfg.get("enable", False)

        # recall config
        recall_cfg = config.get("recall", {})
        self.recall_enable: bool = recall_cfg.get("enable", True)
        self.recall_delay: float = recall_cfg.get("recall_delay", 5.0)
        self.recall_tag: str = recall_cfg.get("recall_tag", "[recall]")
        self.segment_separator: str = config.get("segment_separator", "")
        self.segment_interval: float = config.get("segment_interval", 1.5)

        # prompt injection config
        prompt_cfg = config.get("prompt", {})
        self.prompt_enable: bool = prompt_cfg.get("enable", True)
        self.recall_prompt_template: str = prompt_cfg.get(
            "recall_prompt",
            "你可以在回复中使用 {recall_tag} 标签来标记需要在发送后短暂展示然后自动撤回的内容。"
            '当你想表达一些俏皮话、吐槽、或者"说完就后悔"的内容时可以使用它。'
            "标签放在句首，表示这一整句（到下一个分隔符或句末为止）需要撤回。"
            '示例："{recall_tag}其实我觉得你有点笨"。'
            "带有该标签的消息会在用户看到几秒后自动撤回。"
            "注意：只有紧跟在 {recall_tag} 后面的那一段会被撤回，后续内容正常发送。",
        )

        # track async recall tasks for cleanup
        self._recall_tasks: list[asyncio.Task] = []

        # reply config
        reply_cfg = config.get("reply", {})
        self.reply_enable: bool = reply_cfg.get("enable", True)
        self.reply_tag: str = reply_cfg.get("reply_tag", "[reply:{id}]")
        reply_cache_ttl: int = reply_cfg.get("cache_ttl", 600)
        reply_cache_max: int = reply_cfg.get("cache_max_per_session", 50)
        reply_prompt_template: str = prompt_cfg.get(
            "reply_prompt",
            "你可以引用用户之前发送的消息进行回复。使用 {reply_tag} 标签（将 ID 替换为下方消息列表中的数字 ID）"
            "放在回复内容开头即可。用户不会看到该标签，系统会将其转换为 QQ 原生引用回复。"
            "以下是最近可引用的消息列表：\n{message_list}\n"
            "重要规则：\n"
            "1. 只在你确实想针对某条特定消息回应时才使用引用，不要每句话都引用。大多数情况下不需要引用。\n"
            "2. ID 只能从上方消息列表中选取，禁止使用列表中不存在的数字。\n"
            "3. 不要复述被引用的原文。",
        )

        # Initialize reply manager
        self._reply_mgr = ReplyManager(
            cache_ttl=reply_cache_ttl,
            cache_max_per_session=reply_cache_max,
            prompt_template=reply_prompt_template,
            reply_tag=self.reply_tag,
            recall_tag=self.recall_tag,
            segment_separator=self.segment_separator,
        )

        # poke config
        poke_cfg = config.get("poke", {})
        self.poke_enable: bool = poke_cfg.get("enable", True)
        self.poke_tag: str = poke_cfg.get("poke_tag", "[poke]")
        # Keep the reply manager's poke tag in sync so per-segment parsing can
        # detect and honour [poke] at the position it appears.
        self._reply_mgr.poke_tag = self.poke_tag
        self.poke_prompt_template: str = prompt_cfg.get(
            "poke_prompt",
            "你可以在回复中使用 {poke_tag} 标签来戳对方。"
            "当你想表达亲昵、调皮、或者引起对方注意时可以使用它。"
            "标签可以放在回复文本的任意位置：系统会在发送到该标签所在的那一段消息之后，"
            "立即执行戳一戳动作，因此你可以把它放在某句话之后，在对话进行到一半时戳对方。"
            "若开启了分段发送，戳一戳会出现在它所在分段的位置；一条回复里可以使用多次。"
            '示例："陪我玩一会{poke_tag}可以吗"。',
        )

        # Initialize poke manager
        self._poke_mgr = PokeManager(
            enable=self.poke_enable,
            cooldown=poke_cfg.get("cooldown", 5.0),
            poke_prompt=poke_cfg.get("poke_prompt", ""),
            poke_tag=self.poke_tag,
        )

        # history tag handling config
        # How control tags ([recall]/[reply:ID]/[poke]) are represented in the
        # SAVED conversation history (this never affects what is sent to the user;
        # the message sent to users is always filtered in on_decorating_result).
        #   strip    -> remove tags entirely (clean history)
        #   keep     -> leave tags as-is (raw tags stay in history)
        #   annotate -> replace each tag with a customizable summary string
        history_cfg = config.get("history", {})
        self.history_tag_mode: str = history_cfg.get("tag_mode", "annotate")
        if self.history_tag_mode not in ("strip", "keep", "annotate"):
            self.history_tag_mode = "annotate"
        self.annotate_reply_template: str = history_cfg.get(
            "annotate_reply_template", "（引用了「{quote}」）"
        )
        self.annotate_recall_template: str = history_cfg.get(
            "annotate_recall_template", "（接下来这句已被撤回）"
        )
        self.annotate_poke_template: str = history_cfg.get(
            "annotate_poke_template", "（戳了戳对方）"
        )
        # Precompiled patterns that match the (customizable) annotate summary
        # templates. In annotate mode the summaries are injected into the LLM
        # context as history; the model sometimes mimics them and emits the
        # summary text verbatim in a new reply. Those leaked summaries carry no
        # real control tag, so the action handlers ignore them and they reach
        # the user. We strip them from the outgoing message (silent failure).
        self._summary_res = [
            self._compile_summary_pattern(t)
            for t in (
                self.annotate_reply_template,
                self.annotate_recall_template,
                self.annotate_poke_template,
            )
        ]
        self._summary_res = [r for r in self._summary_res if r is not None]

    @staticmethod
    def _compile_summary_pattern(template: str):
        """Build a regex that matches an annotate summary template.

        ``{quote}`` / ``{id}`` placeholders become non-greedy wildcards so the
        rendered summary (with arbitrary quoted content) is matched. Returns
        None for empty templates.
        """
        if not template or not template.strip():
            return None
        sentinel = "\x00"
        tmp = template.replace("{quote}", sentinel).replace("{id}", sentinel)
        escaped = re.escape(tmp).replace(re.escape(sentinel), ".*?")
        try:
            return re.compile(escaped, re.DOTALL)
        except re.error:
            return None

    def _strip_leaked_summaries(self, text: str) -> str:
        """Remove any annotate summary templates the model echoed into output."""
        if not text or not self._summary_res:
            return text
        cleaned = text
        for pat in self._summary_res:
            cleaned = pat.sub("", cleaned)
        # Collapse whitespace left behind by removed summaries.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    async def initialize(self):
        """Plugin initialization."""
        logger.info(
            f"[ActiveFunction] Initialized. Recall enabled={self.recall_enable}, "
            f"delay={self.recall_delay}s, tag='{self.recall_tag}'. "
            f"Reply enabled={self.reply_enable}, tag='{self.reply_tag}'. "
            f"Poke enabled={self.poke_enable}, tag='{self.poke_tag}'. "
            f"History tag mode='{self.history_tag_mode}'."
        )

    async def terminate(self):
        """Cancel all pending recall tasks on plugin unload."""
        for task in self._recall_tasks:
            task.cancel()
        await asyncio.gather(*self._recall_tasks, return_exceptions=True)
        self._recall_tasks.clear()

    def _remove_task(self, task: asyncio.Task):
        """Callback to remove completed tasks from tracking list."""
        try:
            self._recall_tasks.remove(task)
        except ValueError:
            pass

    def _scene_disabled(self, event: AstrMessageEvent) -> bool:
        """Whether this event's chat scene is disabled for active-function features.

        Private chat is always enabled. Group chat is only enabled when the
        master group toggle (``group.enable``) is on. Used as the single guard
        in every hook so private behavior is unchanged when groups are off.
        """
        if event.get_group_id():
            return not self.group_enable
        return False

    def _own_plugin_name(self) -> str:
        """Best-effort resolve this plugin's registered name for session checks."""
        try:
            from astrbot.core.star.star import star_map

            meta = star_map.get(self.__class__.__module__)
            if meta and meta.name:
                return meta.name
        except Exception:
            pass
        return "astrbot_plugin_active_function"

    async def _session_inactive(self, event: AstrMessageEvent) -> bool:
        """Whether this plugin is disabled for the current session.

        AstrBot's per-session plugin management (会话自定义规则) only filters
        command / message handlers; lifecycle hooks such as on_llm_request and
        on_decorating_result are NOT filtered by it. Without this guard our
        prompt would still be injected (and tags processed) in a session where
        the user disabled the plugin via custom rules. We honour that session
        config ourselves here. Fails open (returns False) when the API is
        unavailable, preserving behaviour on frameworks without this feature.
        """
        try:
            from astrbot.core.star.session_plugin_manager import (
                SessionPluginManager,
            )
        except Exception:
            return False
        try:
            enabled = await SessionPluginManager.is_plugin_enabled_for_session(
                event.unified_msg_origin, self._own_plugin_name()
            )
            return not enabled
        except Exception:
            return False

    async def _disabled(self, event: AstrMessageEvent) -> bool:
        """Combined hook guard: scene disabled (group toggle) OR session-disabled.

        Lets every lifecycle hook bail out both when the chat scene is off
        (private/group toggle) and when the plugin has been disabled for this
        session via AstrBot custom rules.
        """
        return self._scene_disabled(event) or await self._session_inactive(event)

    async def _send_msg(self, event: AiocqhttpMessageEvent, bot, message: list):
        """Send a OneBot message payload to the correct target.

        Routes to the group endpoint (``send_group_msg``) in group chats and to
        the private endpoint (``send_private_msg``) otherwise, so the reply /
        recall / poke flows work in both scenes. Returns the OneBot send result
        dict (which contains ``message_id``); callers wrap this in try/except.
        """
        gid = event.get_group_id()
        if gid:
            return await bot.send_group_msg(group_id=int(gid), message=message)
        return await bot.send_private_msg(
            user_id=int(event.get_sender_id()), message=message
        )

    def _build_system_prompt_suffix(self, event: AstrMessageEvent | None = None) -> str:
        """Build the system prompt suffix from configured templates.

        When the event's scene is disabled (e.g. a group chat while group
        support is off), we advertise none of the tags. Otherwise the LLM might
        emit [recall]/[poke] in a scene where the handlers skip processing,
        leaking the raw tags to users.
        """
        if event is not None and self._scene_disabled(event):
            return ""

        parts = []
        if self.recall_enable and self.recall_prompt_template:
            recall_prompt = self.recall_prompt_template.replace(
                "{recall_tag}", self.recall_tag
            )
            parts.append(recall_prompt)

        # Reply prompt injection (only for aiocqhttp private chat)
        if (
            self.reply_enable
            and event is not None
            and isinstance(event, AiocqhttpMessageEvent)
            and not self._scene_disabled(event)
        ):
            reply_prompt = self._reply_mgr.build_reply_prompt(event.unified_msg_origin)
            if reply_prompt:
                parts.append(reply_prompt)

        # Poke prompt injection (for aiocqhttp platform)
        if (
            self.poke_enable
            and self.poke_prompt_template
            and event is not None
            and isinstance(event, AiocqhttpMessageEvent)
        ):
            poke_prompt = self.poke_prompt_template.replace("{poke_tag}", self.poke_tag)
            parts.append(poke_prompt)

        return "\n".join(parts)

    # ==================== Reply: Message Caching ====================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def cache_user_message(self, event: AstrMessageEvent):
        """Cache original user message before debounce plugin merges it.

        Priority 100 ensures this runs BEFORE the debounce plugin (priority 50).
        We do NOT stop the event, so debounce can still merge messages normally.

        Supports caching messages that contain images or voice (Record) by
        generating descriptive placeholders like [图片] or [语音], so the LLM
        can reference them via [reply:ID] even if there's no text content.
        """
        if not self.reply_enable:
            return
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if self._scene_disabled(event):
            return

        message_id = self._reply_mgr.extract_message_id(event)
        if message_id is None:
            logger.debug(
                "[ActiveFunction] Reply cache: cannot extract message_id, skipping"
            )
            return

        # Build display text from message chain, including media placeholders
        text = self._build_cache_text(event)
        if not text:
            return

        # In group chats, attach a sender label so the quotable-message list can
        # show who said what (private chats only have one speaker, so skip it).
        sender = ""
        if event.get_group_id():
            name = (event.get_sender_name() or "").strip()
            uid = str(event.get_sender_id() or "").strip()
            if name and uid:
                sender = f"{name}({uid})"
            else:
                sender = name or uid

        self._reply_mgr.cache_message(
            event.unified_msg_origin, message_id, text, sender=sender
        )

    def _build_cache_text(self, event: AiocqhttpMessageEvent) -> str:
        """Build display text for cache from the message chain.

        For text-only messages, returns the plain text.
        For messages with images/voice, appends descriptive placeholders
        so the LLM knows the message contained media and can reference it.
        """
        msg_obj = getattr(event, "message_obj", None)
        message_chain = getattr(msg_obj, "message", None) if msg_obj else None

        if not message_chain:
            # Fallback to message_str
            return (event.message_str or "").strip()

        parts = []
        for comp in message_chain:
            if isinstance(comp, Plain):
                t = comp.text.strip()
                if t:
                    parts.append(t)
            elif isinstance(comp, Image):
                parts.append("[图片]")
            elif isinstance(comp, Record):
                parts.append("[语音]")

        return " ".join(parts).strip()

    # ==================== Poke: Event Interception ====================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=90)
    async def handle_poke_event(self, event: AstrMessageEvent):
        """Intercept poke notice events and inject readable text.

        Priority 90 ensures this runs early. When a user pokes the bot,
        we inject readable text into the event and set is_at_or_wake_command
        so the framework processes it through the normal LLM pipeline
        (including debounce merging with other messages).
        """
        if not self.poke_enable:
            return
        if not isinstance(event, AiocqhttpMessageEvent):
            return

        if self._scene_disabled(event):
            return

        # Get raw message to check for poke event
        raw_message = getattr(event.message_obj, "raw_message", None)
        if not raw_message:
            return

        # Convert to dict if needed
        if not isinstance(raw_message, dict):
            try:
                raw_message = (
                    dict(raw_message) if hasattr(raw_message, "__getitem__") else None
                )
            except Exception:
                raw_message = None
        if not raw_message:
            return

        # Check if this is a poke event
        if not self._poke_mgr.is_poke_event(raw_message):
            return

        # Get bot ID and check if poke targets the bot
        bot_id = event.get_self_id() or str(raw_message.get("self_id", ""))
        if not self._poke_mgr.is_poke_targeting_bot(raw_message, bot_id):
            event.stop_event()
            return

        # Get sender info
        sender_id = self._poke_mgr.get_poke_sender_id(raw_message)
        if not sender_id:
            return

        # Access control and cooldown check
        allowed, reason = self._poke_mgr.should_respond(sender_id)
        if not allowed:
            logger.debug(f"[ActiveFunction] Poke ignored: {reason} (user={sender_id})")
            event.stop_event()
            return

        # Inject readable text into the event so it merges with debounce
        username = event.get_sender_name() or sender_id
        poke_text = self._poke_mgr.format_poke_injection(username)
        event.message_str = poke_text
        if hasattr(event.message_obj, "message_str"):
            event.message_obj.message_str = poke_text
        if hasattr(event.message_obj, "message"):
            event.message_obj.message = [Plain(poke_text)]

        # Mark as wake command so the framework processes it through LLM pipeline
        event.is_at_or_wake_command = True
        event.set_extra("_poke_trigger", True)

        logger.info(
            f"[ActiveFunction] Poke #{self._poke_mgr.poke_count} | "
            f"user: {username}({sender_id}) | injected into pipeline"
        )

    # ==================== Poke: on_decorating_result ====================

    @filter.on_decorating_result(priority=15)
    async def strip_leaked_summaries_decorate(self, event: AstrMessageEvent):
        """Strip annotate summary text the model echoed into its reply.

        Runs before the poke/reply/recall handlers (priority 12/11/10) so the
        cleaned text flows into them. Only active in ``annotate`` mode, since
        that is the only mode that feeds summaries into the LLM context (and
        thus the only one that can induce the model to mimic them). Real
        ``[reply:ID]`` / ``[recall]`` / ``[poke]`` tags are untouched — the
        summary patterns never match them.
        """
        if self.history_tag_mode != "annotate" or not self._summary_res:
            return
        if await self._disabled(event):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # Clean the stashed original text used by the reply/poke takeover paths.
        original_text = event.get_extra("_active_func_original_text")
        if original_text:
            event.set_extra(
                "_active_func_original_text",
                self._strip_leaked_summaries(original_text),
            )
        recall_full = event.get_extra("_recall_full_text")
        if recall_full:
            event.set_extra(
                "_recall_full_text", self._strip_leaked_summaries(recall_full)
            )

        # Clean the outgoing chain (covers the normal, non-takeover send path).
        for i, comp in enumerate(result.chain):
            if isinstance(comp, Plain):
                cleaned = self._strip_leaked_summaries(comp.text)
                if cleaned != comp.text:
                    result.chain[i] = Plain(cleaned)

    @filter.on_decorating_result(priority=12)
    async def handle_poke_decorate(self, event: AstrMessageEvent):
        """Parse [poke] tags from LLM output, strip them, and execute poke action.

        Priority 12 ensures this runs before reply (11) and recall (10) handlers.
        This handler strips [poke] tags from the text and immediately executes
        the poke action. The cleaned text continues to flow through reply/recall
        handlers and normal segment splitting.

        When poke is disabled, tags are still stripped to prevent leaking to users.

        Compatible with both normal messages (AiocqhttpMessageEvent with event.bot)
        and proactive messages (generic AstrMessageEvent, bot fetched from platform).
        """
        # Check platform is aiocqhttp (works for both normal and proactive events)
        platform_name = getattr(event.platform_meta, "name", "")
        if platform_name != "aiocqhttp":
            return

        if await self._disabled(event):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # Resolve a text source that still contains the tags: prefer the original
        # (pre-strip) LLM text so [poke] positions are preserved, else the chain.
        original_text = event.get_extra("_active_func_original_text")
        if original_text:
            text_src = original_text
        else:
            text_src = "".join(
                comp.text for comp in result.chain if isinstance(comp, Plain)
            )

        if not self._poke_mgr.has_poke_tag(text_src):
            return

        def _strip_poke_from_chain() -> None:
            for i, comp in enumerate(result.chain):
                if isinstance(comp, Plain) and self.poke_tag in comp.text:
                    result.chain[i] = Plain(comp.text.replace(self.poke_tag, ""))

        is_streaming = (
            result.result_content_type == ResultContentType.STREAMING_FINISH
            if hasattr(result, "result_content_type")
            else False
        )

        # Streaming mode: content was already streamed out with [poke] inside, so
        # we cannot intersperse the poke. Recall + re-send the cleaned text, then
        # fire a single poke at the end (legacy behaviour).
        if is_streaming and isinstance(event, AiocqhttpMessageEvent):
            _strip_poke_from_chain()
            await self._fix_streaming_poke(event, original_text or "")
            event.set_extra("_recall_handled", True)
            if self.poke_enable:
                bot = getattr(event, "bot", None) or self._get_bot_from_platform(event)
                if bot is not None:
                    await self._fire_poke(event, bot)
            return

        # Proactive / non-aiocqhttp events: we cannot take over segmented sending,
        # so strip the tag and fire a single poke (legacy behaviour).
        if not isinstance(event, AiocqhttpMessageEvent):
            _strip_poke_from_chain()
            if self.poke_enable:
                bot = getattr(event, "bot", None) or self._get_bot_from_platform(event)
                if bot is not None:
                    await self._fire_poke(event, bot)
            return

        # Non-streaming aiocqhttp. The poke must fire at its [poke] position, which
        # means whoever sends the segments fires it:
        #   * reply or recall active -> that handler takes over and fires per-segment
        #     (leave [poke] in the text for it; do NOT strip the chain).
        #   * poke active but reply/recall not -> WE take over and send segment by
        #     segment so the poke lands in position.
        #   * nothing active -> strip [poke] from the chain so it does not leak.
        has_reply = bool(re.search(r"\[reply:\d+\]", text_src))
        has_recall = bool(self.recall_tag) and self.recall_tag in text_src
        if (has_reply and self.reply_enable) or (has_recall and self.recall_enable):
            return

        if not self.poke_enable:
            _strip_poke_from_chain()
            logger.debug("[ActiveFunction] Poke disabled, stripped tag from output")
            return

        logger.info(
            "[ActiveFunction] on_decorating_result: handling poke (positional)"
        )

        # Build ordered chunks and take over sending so the poke lands in position.
        ordered_chunks = self._build_takeover_chunks(result, original_text)
        # reply/recall are inactive here; if their (disabled) tags are present,
        # strip them so the takeover sender does not honour a disabled feature.
        cleaned_chunks: list = []
        for chunk in ordered_chunks:
            if isinstance(chunk, str):
                t = chunk
                if not self.reply_enable:
                    t = re.sub(r"\[reply:\d+\]", "", t)
                if not self.recall_enable and self.recall_tag:
                    t = t.replace(self.recall_tag, "")
                cleaned_chunks.append(t)
            else:
                cleaned_chunks.append(chunk)

        await self._send_reply_segments_ordered(
            event, cleaned_chunks, event.unified_msg_origin
        )
        event.clear_result()
        event._has_send_oper = True
        event.set_extra("_recall_handled", True)

    async def _fire_poke(self, event: AstrMessageEvent, bot) -> None:
        """Execute a poke at the current point in sending.

        Uses ``group_poke`` in group chats and ``friend_poke`` in private. This
        is best-effort: unsupported OneBot implementations / failures are
        swallowed by the poke manager and simply never surface to the user.
        """
        try:
            target_user_id = int(event.get_sender_id())
        except (TypeError, ValueError):
            return
        gid = event.get_group_id()
        if gid:
            success = await self._poke_mgr.send_group_poke(
                bot, int(gid), target_user_id
            )
        else:
            success = await self._poke_mgr.send_poke(bot, target_user_id)
        if success:
            logger.info(f"[ActiveFunction] Poke sent to user {target_user_id}")
        else:
            logger.debug(
                f"[ActiveFunction] Poke action failed/skipped for {target_user_id}"
            )

    def _build_takeover_chunks(self, result, original_text: str | None) -> list:
        """Build an ordered list of chunks for a segmented takeover send.

        Each element is either a text ``str`` (which may still contain
        [reply:ID]/[recall]/[poke] tags, acted on per-segment downstream) or a
        non-Plain component (e.g. a Record from the TTS plugin) kept in position.
        ``<tts>...</tts>`` 和 ``<tts:emotion>...</tts>`` 标签会从文本中剥离，因为 TTS 插件
        already converted them into Record components in the chain.
        """
        ordered_chunks: list = []
        if original_text:
            text = re.sub(r"<tts(?::\w+)?>.*?</tts(?::\w+)?>", "", original_text, flags=re.DOTALL)
            text = re.sub(r"  +", " ", text).strip()
            if text:
                ordered_chunks.append(text)
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    ordered_chunks.append(comp)
        else:
            current_text = ""
            for comp in result.chain:
                if isinstance(comp, Plain):
                    current_text += comp.text
                else:
                    if current_text:
                        ordered_chunks.append(current_text)
                        current_text = ""
                    ordered_chunks.append(comp)
            if current_text:
                ordered_chunks.append(current_text)
        return ordered_chunks

    def _get_bot_from_platform(self, event: AstrMessageEvent):
        """Get bot instance from platform adapter (for proactive messages)."""
        try:
            platform_id = event.get_platform_id()
            platform_inst = self.context.get_platform_inst(platform_id)
            if platform_inst is None:
                return None
            # aiocqhttp adapter exposes bot via get_client() or .bot
            if hasattr(platform_inst, "get_client"):
                return platform_inst.get_client()
            return getattr(platform_inst, "bot", None)
        except Exception:
            return None

    async def _fix_streaming_poke(
        self, event: AiocqhttpMessageEvent, original_text: str
    ):
        """Fix [poke] tag leaking in streaming mode.

        In streaming mode, the message was already sent with [poke] in it
        (because streaming chunks are buffered and sent before on_decorating_result fires).
        We recall the sent message and re-send it without the [poke] tag.
        """
        try:
            bot = getattr(event, "bot", None)
            if bot is None:
                bot = self._get_bot_from_platform(event)
            if bot is None:
                return

            uid = int(event.get_sender_id())
            bot_id = event.get_self_id() or ""
            gid = event.get_group_id()

            # Get recent messages to find the one we just sent (group vs private).
            try:
                if gid:
                    result = await bot.call_action(
                        "get_group_msg_history",
                        group_id=int(gid),
                        count=10,
                    )
                else:
                    result = await bot.call_action(
                        "get_friend_msg_history",
                        user_id=uid,
                        count=5,
                    )
                messages = result.get("messages", []) if result else []
            except Exception:
                # API not available on this OneBot implementation
                logger.debug(
                    "[ActiveFunction] msg history API not available, "
                    "cannot fix streaming poke tag"
                )
                return

            if not messages:
                return

            # Find the last message from the bot that contains [poke]
            target_msg_id = None
            for msg in reversed(messages):
                sender_id = str(msg.get("sender", {}).get("user_id", ""))
                if sender_id == bot_id:
                    # Check if this message contains [poke]
                    msg_content = ""
                    raw_msg = msg.get("message", [])
                    if isinstance(raw_msg, list):
                        for seg in raw_msg:
                            if isinstance(seg, dict) and seg.get("type") == "text":
                                msg_content += seg.get("data", {}).get("text", "")
                    elif isinstance(raw_msg, str):
                        msg_content = raw_msg

                    if self.poke_tag in msg_content:
                        target_msg_id = msg.get("message_id")
                        break

            if not target_msg_id:
                return

            # Recall the message with [poke] tag
            await bot.delete_msg(message_id=int(target_msg_id))

            # Re-send without [poke] (also strip [recall], [reply:ID], and <tts> tags)
            clean_text = original_text.replace(self.poke_tag, "")
            clean_text = clean_text.replace(self.recall_tag, "")
            clean_text = re.sub(r"\[reply:\d+\]", "", clean_text)
            clean_text = re.sub(r"<tts(?::\w+)?>.*?</tts(?::\w+)?>", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"  +", " ", clean_text).strip()

            if clean_text:
                await self._send_msg(
                    event,
                    bot,
                    [{"type": "text", "data": {"text": clean_text}}],
                )

            logger.info(
                "[ActiveFunction] Fixed streaming poke: recalled and re-sent without tag"
            )
        except Exception as e:
            logger.debug(f"[ActiveFunction] Failed to fix streaming poke: {e}")

    # ==================== Reply: on_decorating_result ====================

    @filter.on_decorating_result(priority=11)
    async def handle_reply_decorate(self, event: AstrMessageEvent):
        """Parse [reply:ID] tags from LLM output and send as QQ native replies.

        Priority 11 ensures this runs before the recall handler (priority 10).
        If reply tags are found, this handler takes over sending entirely.
        If no reply tags but recall tags exist, this handler does nothing and
        lets the recall handler process it.

        When reply is disabled, tags are still stripped to prevent leaking to users.

        This handler preserves the ordering of non-Plain components (e.g. Record
        from TTS plugin) within the message chain, so that voice messages are sent
        in their correct position relative to text segments rather than all at the end.
        """
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if await self._disabled(event):
            return
        # Skip if already handled by streaming poke fix
        if event.get_extra("_recall_handled"):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # Use original text from on_llm_response if available (tags were stripped for history)
        original_text = event.get_extra("_active_func_original_text")
        if original_text and re.search(r"\[reply:\d+\]", original_text):
            full_text = original_text
        else:
            # Fallback: get plain text from chain (for tag detection only)
            full_text = ""
            for comp in result.chain:
                if isinstance(comp, Plain):
                    full_text += comp.text

        if not self._reply_mgr.has_reply_tags(full_text):
            return

        # If reply is disabled, just strip the tags and let normal sending proceed
        if not self.reply_enable:
            logger.debug(
                "[ActiveFunction] Reply disabled, stripping reply tags from output"
            )
            for i, comp in enumerate(result.chain):
                if isinstance(comp, Plain):
                    result.chain[i] = Plain(re.sub(r"\[reply:\d+\]", "", comp.text))
            return

        logger.info("[ActiveFunction] on_decorating_result: handling reply tags")

        # Build an ordered list of "chunks" preserving chain order.
        # Each chunk is either a text string (from consecutive Plains) or a
        # non-Plain component (e.g. Record from TTS plugin).
        # This preserves the interleaving order so voice is sent in position.
        #
        # When original_text is available (tags were stripped for history saving),
        # we use it as the text source instead of the chain's cleaned text.
        # [poke] is kept in the text so it can be fired per-segment below; we only
        # strip <tts>...</tts> 和 <tts:emotion>...</tts> 标签，因为 TTS 插件 (priority 13) 已经
        # processed them into Record components in the chain.
        ordered_chunks: list[str | BaseMessageComponent] = []
        if original_text and re.search(r"\[reply:\d+\]", original_text):
            # Use original text with reply tags; keep [poke] so the segmented
            # sender below can fire it at the position it appears.
            text_for_reply = original_text
            # Strip <tts>...</tts> 和 <tts:emotion>...</tts> 或 <tts:emotion>...</tts:emotion> 标签 — TTS 插件已转换为 Record
            # components which are preserved as non-Plain in the chain below.
            # Keep only the text outside the tags; the TTS audio is sent via Record.
            text_for_reply = re.sub(
                r"<tts(?::\w+)?>.*?</tts(?::\w+)?>", "", text_for_reply, flags=re.DOTALL
            )
            # Clean up extra whitespace left by tag removal
            text_for_reply = re.sub(r"  +", " ", text_for_reply).strip()
            # Use original text with tags; preserve non-Plain components from chain
            non_plain_components = [
                comp for comp in result.chain if not isinstance(comp, Plain)
            ]
            # Insert text as the first chunk, then non-Plain components
            ordered_chunks.append(text_for_reply)
            for comp in non_plain_components:
                ordered_chunks.append(comp)
        else:
            current_text = ""
            for comp in result.chain:
                if isinstance(comp, Plain):
                    current_text += comp.text
                else:
                    if current_text:
                        ordered_chunks.append(current_text)
                        current_text = ""
                    ordered_chunks.append(comp)
            if current_text:
                ordered_chunks.append(current_text)

        # Check if all text content is empty (only non-Plain components remain)
        all_text = "".join(c for c in ordered_chunks if isinstance(c, str))
        stripped_all = re.sub(r"\[reply:\d+\]", "", all_text).strip()
        if not stripped_all:
            # No text content after stripping reply tags — let non-Plain send normally
            logger.info(
                "[ActiveFunction] Reply segments have no text content, "
                "stripping reply tags to avoid QQ incompatibility"
            )
            new_chain = []
            for chunk in ordered_chunks:
                if isinstance(chunk, str):
                    cleaned = re.sub(r"\[reply:\d+\]", "", chunk)
                    cleaned = cleaned.replace(self.poke_tag, "").strip()
                    if cleaned:
                        new_chain.append(Plain(cleaned))
                else:
                    new_chain.append(chunk)
            result.chain = new_chain
            return

        # Send all chunks in order, preserving interleaving of text and non-Plain
        session_key = event.unified_msg_origin
        await self._send_reply_segments_ordered(event, ordered_chunks, session_key)

        # Prevent the framework from sending anything
        event.clear_result()
        event._has_send_oper = True
        # Mark as handled so recall handler and after_message_sent don't double-process
        event.set_extra("_recall_handled", True)

    async def _send_reply_segments_ordered(
        self,
        event: AiocqhttpMessageEvent,
        ordered_chunks: list,
        session_key: str,
    ) -> None:
        """Send chunks in order, preserving interleaving of text and non-Plain components.

        ordered_chunks is a list where each element is either:
        - str: text that may contain [reply:ID] and [recall] tags
        - BaseMessageComponent: a non-Plain component (e.g. Record from TTS)

        Text chunks are parsed into ReplySegments (handling reply/recall tags).
        Non-Plain components are sent in their original position in the chain.
        """
        bot = event.bot
        sent_count = 0

        for chunk in ordered_chunks:
            if isinstance(chunk, str):
                # Parse text chunk into reply segments
                segments = self._reply_mgr.parse_segments(chunk, session_key)
                for segment in segments:
                    if not segment.text:
                        # A bare [poke] segment: fire the poke, send nothing.
                        if segment.should_poke and self.poke_enable:
                            await self._fire_poke(event, bot)
                        continue

                    # Interval between messages
                    if sent_count > 0:
                        await asyncio.sleep(self.segment_interval)

                    # Build message payload
                    message = []
                    if segment.reply_message_id is not None:
                        message.append(
                            {
                                "type": "reply",
                                "data": {"id": str(segment.reply_message_id)},
                            }
                        )
                    message.append({"type": "text", "data": {"text": segment.text}})

                    try:
                        send_result = await self._send_msg(event, bot, message)
                    except Exception as e:
                        logger.error(
                            f"[ActiveFunction] Failed to send reply segment: {e}"
                        )
                        continue

                    sent_count += 1

                    # Schedule recall if needed
                    if segment.should_recall and send_result:
                        message_id = send_result.get("message_id")
                        if message_id:
                            task = asyncio.create_task(
                                self._delayed_recall(bot, int(message_id))
                            )
                            task.add_done_callback(self._remove_task)
                            self._recall_tasks.append(task)
                            logger.info(
                                f"[ActiveFunction] Reply+Recall scheduled: msg_id={message_id}"
                            )

                    # Fire poke right after this segment, if tagged.
                    if segment.should_poke and self.poke_enable:
                        await self._fire_poke(event, bot)
            else:
                # Non-Plain component (e.g. Record from TTS plugin)
                if sent_count > 0:
                    await asyncio.sleep(self.segment_interval)

                if isinstance(chunk, Record):
                    await self._send_record_component(event, bot, chunk)
                else:
                    try:
                        msg_dict = chunk.toDict()
                        await self._send_msg(event, bot, [msg_dict])
                    except Exception as e:
                        logger.debug(
                            f"[ActiveFunction] Skipped non-Plain component: {e}"
                        )

                sent_count += 1

    async def _send_record_component(
        self, event: AiocqhttpMessageEvent, bot, record: Record
    ) -> None:
        """Send a Record (voice) component via OneBot API (group or private)."""
        try:
            # Convert Record to base64 format for sending (same as aiocqhttp platform)
            bs64 = await record.convert_to_base64()
            message = [{"type": "record", "data": {"file": f"base64://{bs64}"}}]
            await self._send_msg(event, bot, message)
            logger.info("[ActiveFunction] Sent voice message")
        except Exception as e:
            logger.error(f"[ActiveFunction] Failed to send voice: {e}")
            # Fallback: send the text content if available
            if record.text:
                try:
                    await self._send_msg(
                        event,
                        bot,
                        [{"type": "text", "data": {"text": record.text}}],
                    )
                except Exception:
                    pass

    # ==================== Prompt Injection ====================

    @filter.on_llm_request()
    async def inject_prompt(self, event: AstrMessageEvent, request: ProviderRequest):
        """Inject function usage instructions into the LLM system prompt.

        Also applies the configured history tag mode to the conversation
        context that is about to be sent to the LLM, so that pre-existing raw
        tags in older history are handled consistently (strip removes them,
        annotate rewrites them into summaries, keep leaves them untouched).
        This only mutates the in-flight request context, not the stored history.
        """
        if await self._session_inactive(event):
            return
        if not self.prompt_enable:
            return
        suffix = self._build_system_prompt_suffix(event)
        if suffix:
            request.system_prompt += "\n\n" + suffix

        if self.history_tag_mode != "keep":
            self._apply_history_mode_to_contexts(request, event.unified_msg_origin)

    def _apply_history_mode_to_contexts(self, request: ProviderRequest, session_key: str):
        """Rewrite control tags in assistant history messages per history_tag_mode.

        Only relevant for ``strip`` and ``annotate``; ``keep`` is a no-op and
        never calls this. Mutates the request contexts in place (in-flight only).
        """
        if not request.contexts:
            return

        for ctx in request.contexts:
            if not isinstance(ctx, dict):
                continue
            if ctx.get("role") != "assistant":
                continue

            content = ctx.get("content")
            if isinstance(content, str):
                if self._has_control_tag(content):
                    ctx["content"] = self._transform_history_text(content, session_key)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if self._has_control_tag(text):
                            part["text"] = self._transform_history_text(
                                text, session_key
                            )

    def _has_control_tag(self, text: str) -> bool:
        """Whether the text contains any [recall]/[reply:ID]/[poke] control tag."""
        return (
            (bool(self.recall_tag) and self.recall_tag in text)
            or (bool(self.poke_tag) and self.poke_tag in text)
            or self._reply_mgr.has_reply_tags(text)
        )

    def _transform_history_text(self, text: str, session_key: str) -> str:
        """Transform control tags in a text for conversation-history representation.

        Honors ``history_tag_mode``:
          - ``keep``     -> returned unchanged
          - ``strip``    -> all control tags removed
          - ``annotate`` -> each tag replaced by its (customizable) summary string

        Never used for the message that is actually sent to the user.
        """
        mode = self.history_tag_mode
        if mode == "keep":
            return text

        if mode == "strip":
            cleaned = text
            if self.recall_tag:
                cleaned = cleaned.replace(self.recall_tag, "")
            if self.poke_tag:
                cleaned = cleaned.replace(self.poke_tag, "")
            cleaned = self._reply_mgr.strip_all_reply_variants(cleaned)
            return re.sub(r"  +", " ", cleaned).strip()

        # annotate
        def _reply_repl(match: re.Match) -> str:
            rid = int(match.group(1))
            cached = self._reply_mgr.get_cached_text(session_key, rid)
            if not cached:
                # The referenced message is not (or no longer) in the cache, so
                # the quote never resolved — i.e. the reply effectively failed.
                # Do NOT emit a hollow "（引用了「」）" summary: it is useless to
                # the model and, once in history, the model tends to mimic it
                # and leak the literal summary text to users. Strip it instead.
                return ""
            quote = cached.replace("\n", " ").strip()
            if len(quote) > 30:
                quote = quote[:30] + "…"
            try:
                return self.annotate_reply_template.format(quote=quote, id=rid)
            except (KeyError, IndexError, ValueError):
                # User template has unexpected placeholders/braces; use as-is.
                return self.annotate_reply_template

        annotated = self._reply_mgr._reply_tag_pattern.sub(_reply_repl, text)
        if self.recall_tag:
            annotated = annotated.replace(self.recall_tag, self.annotate_recall_template)
        if self.poke_tag:
            annotated = annotated.replace(self.poke_tag, self.annotate_poke_template)
        return re.sub(r"  +", " ", annotated).strip()

    # ==================== LLM Response Hook ====================

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """When LLM responds, prepare both the sending stage and saved history.

        We always stash the original text (with raw tags) in event extras so the
        on_decorating_result handlers can execute the corresponding actions
        (poke / reply / recall) and filter the tags out of what is actually sent
        to the user. The message sent to users is therefore unaffected by the
        history tag mode.

        For the SAVED conversation history we apply ``history_tag_mode``:
          - ``keep``     -> leave completion_text / run_context untouched (raw tags)
          - ``strip``    -> remove tags from the persisted assistant message
          - ``annotate`` -> rewrite tags into summary strings in the persisted message

        We deliberately do NOT touch response.result_chain here: that chain feeds
        the outgoing message, and the on_decorating_result handlers rely on the
        raw tags still being present to clean the output correctly.
        """
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if await self._disabled(event):
            return

        text = response.completion_text if response else ""
        if not text:
            return

        # Check if any of our control tags are present
        has_recall = self.recall_tag in text
        has_poke = self.poke_tag in text
        has_reply = bool(re.search(r"\[reply:\d+\]", text))

        if not has_recall and not has_poke and not has_reply:
            return

        # Stash the raw text (with tags) for the on_decorating_result handlers.
        event.set_extra("_active_func_original_text", text)

        if has_recall:
            event.set_extra("_recall_pending", True)
            event.set_extra("_recall_full_text", text)

        # Apply the history tag mode to what gets persisted. "keep" leaves the
        # raw tags in place (no rewrite needed).
        if self.history_tag_mode != "keep":
            history_text = self._transform_history_text(
                text, event.unified_msg_origin
            )
            response.completion_text = history_text
            self._patch_last_assistant_message(event, history_text)

        logger.info(
            "[ActiveFunction] Detected control tags in LLM response "
            f"(history_tag_mode={self.history_tag_mode}). "
            f"recall={has_recall}, poke={has_poke}, reply={has_reply}"
        )

    def _patch_last_assistant_message(self, event: AstrMessageEvent, new_text: str):
        """Patch the last assistant message in run_context.messages with new_text.

        The agent runner appends the assistant message to run_context.messages
        BEFORE on_llm_response fires, so we rewrite its TextPart to control what
        gets persisted to conversation history (used by strip/annotate modes).
        """
        try:
            from astrbot.core.pipeline.process_stage.follow_up import (
                _ACTIVE_AGENT_RUNNERS,
            )

            runner = _ACTIVE_AGENT_RUNNERS.get(event.unified_msg_origin)
            if not runner or not hasattr(runner, "run_context"):
                return

            messages = runner.run_context.messages
            if not messages:
                return

            last_msg = messages[-1]
            if getattr(last_msg, "role", None) != "assistant":
                return

            content = getattr(last_msg, "content", None)
            if not content:
                return

            if isinstance(content, list):
                for part in content:
                    if (
                        hasattr(part, "type")
                        and part.type == "text"
                        and hasattr(part, "text")
                    ):
                        part.text = new_text
                        break
            elif isinstance(content, str):
                last_msg.content = new_text
        except Exception as e:
            logger.debug(f"[ActiveFunction] Failed to patch assistant message: {e}")

    # ==================== on_decorating_result (non-streaming) ====================

    @filter.on_decorating_result(priority=10)
    async def handle_recall_decorate(self, event: AstrMessageEvent):
        """For non-streaming mode: intercept before send, handle recall logic.

        When recall is disabled, tags are still stripped to prevent leaking to users.

        This hook may NOT fire in streaming mode.
        """
        result = event.get_result()
        if not result or not result.chain:
            return

        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if await self._disabled(event):
            return
        # Skip if already handled by streaming poke fix
        if event.get_extra("_recall_handled"):
            return

        # Use original text from on_llm_response if available (tags were stripped for history)
        original_text = event.get_extra("_active_func_original_text")
        if original_text and self.recall_tag in original_text:
            # Keep [poke] so it can be fired in position by _send_with_recall.
            full_text = original_text
            # Strip <tts>...</tts> 和 <tts:emotion>...</tts> 或 <tts:emotion>...</tts:emotion> 标签 — TTS 插件已转换为 Record
            # components; sending raw tags as text would leak them to the user.
            full_text = re.sub(r"<tts(?::\w+)?>.*?</tts(?::\w+)?>", "", full_text, flags=re.DOTALL)
            full_text = re.sub(r"  +", " ", full_text).strip()
        else:
            # Fallback: get plain text from chain
            full_text = ""
            for comp in result.chain:
                if isinstance(comp, Plain):
                    full_text += comp.text

        if self.recall_tag not in full_text:
            return

        # If recall is disabled, just strip the tags and let normal sending proceed
        if not self.recall_enable:
            logger.debug(
                "[ActiveFunction] Recall disabled, stripping recall tags from output"
            )
            stripped_text = full_text.replace(self.recall_tag, "")
            # Rebuild chain with stripped text
            new_chain = []
            text_replaced = False
            for comp in result.chain:
                if isinstance(comp, Plain) and not text_replaced:
                    new_chain.append(Plain(stripped_text))
                    text_replaced = True
                elif not isinstance(comp, Plain):
                    new_chain.append(comp)
            result.chain = new_chain
            return

        logger.info(
            f"[ActiveFunction] on_decorating_result: handling recall. "
            f"separator='{self.segment_separator}' (len={len(self.segment_separator)})"
        )

        # Take over sending
        await self._send_with_recall(event, full_text)

        # Prevent the framework from sending anything.
        # Setting result to None or clearing it entirely so RespondStage skips.
        event.clear_result()

        # Mark that we did send something (for framework bookkeeping)
        event._has_send_oper = True
        # Mark as handled so after_message_sent doesn't double-process
        event.set_extra("_recall_handled", True)

    # ==================== after_message_sent (streaming fallback) ====================

    @filter.after_message_sent()
    async def handle_recall_after_sent(self, event: AstrMessageEvent):
        """Fallback for streaming mode: after message is sent, recall if needed.

        In streaming mode, on_decorating_result doesn't fire, so the message
        with [recall] tags gets sent as-is. We then recall it post-hoc.
        """
        if not self.recall_enable:
            return
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if await self._disabled(event):
            return

        # Skip if already handled by on_decorating_result
        if event.get_extra("_recall_handled"):
            return

        # Check if this event was marked by on_llm_response
        if not event.get_extra("_recall_pending"):
            return

        logger.info("[ActiveFunction] after_message_sent: streaming fallback recall")

        # In streaming mode, the message was already sent with the tag text.
        # We need to find the sent message and recall it.
        # Use get_msg or recent message history from bot API.

        # Strategy: send a corrected version (without tag) and recall the original.
        # But we can't easily get the message_id of what was just sent by the framework.
        # Best effort: use the OneBot "get_msg" or check recent messages.
        # For now, we'll just log a warning - full streaming support requires
        # patching the send method or using a different approach.
        full_text = event.get_extra("_recall_full_text", "")
        if not full_text:
            return

        # For private chat, we can try to get recent messages
        try:
            # Get recent message history to find our sent message
            # This is OneBot v11 API: get_friend_msg_history (not always available)
            # Fallback: just send the clean version and note the limitation
            logger.warning(
                "[ActiveFunction] Streaming mode recall: limited support. "
                "Consider disabling streaming for full recall functionality."
            )
        except Exception as e:
            logger.error(f"[ActiveFunction] Streaming fallback error: {e}")

    # ==================== Core Logic ====================

    async def _send_with_recall(self, event: AiocqhttpMessageEvent, full_text: str):
        """Send message segments, scheduling recall for tagged ones."""
        segments = self._split_into_segments(full_text)
        logger.info(
            f"[ActiveFunction] Split into {len(segments)} segment(s): "
            f"{[s[:30] for s in segments]}"
        )
        bot = event.bot

        for i, segment in enumerate(segments):
            # Check if this segment starts with the recall tag (after stripping)
            needs_recall = segment.startswith(self.recall_tag)
            # Remove the recall tag from displayed text
            display_text = segment.removeprefix(self.recall_tag).strip()
            # Detect and strip the poke tag (fired right after this segment sends)
            seg_has_poke = bool(self.poke_tag) and self.poke_tag in display_text
            if seg_has_poke:
                display_text = display_text.replace(self.poke_tag, "").strip()
            # Remove [NEXT] tag from displayed text
            display_text = _NEXT_TAG_PATTERN.sub("", display_text).strip()

            if not display_text:
                # A bare [poke] segment: fire the poke, send nothing.
                if seg_has_poke and self.poke_enable:
                    await self._fire_poke(event, bot)
                continue

            # Send via bot API to get message_id (group or private)
            try:
                send_result = await self._send_msg(
                    event,
                    bot,
                    [{"type": "text", "data": {"text": display_text}}],
                )
            except Exception as e:
                logger.error(f"[ActiveFunction] Failed to send segment: {e}")
                continue

            # Schedule recall if needed
            if needs_recall and send_result:
                message_id = send_result.get("message_id")
                if message_id:
                    task = asyncio.create_task(
                        self._delayed_recall(bot, int(message_id))
                    )
                    task.add_done_callback(self._remove_task)
                    self._recall_tasks.append(task)
                    logger.info(
                        f"[ActiveFunction] Recall scheduled: msg_id={message_id}, "
                        f"delay={self.recall_delay}s"
                    )

            # Fire poke right after this segment, if tagged.
            if seg_has_poke and self.poke_enable:
                await self._fire_poke(event, bot)

            # Interval between segments
            if i < len(segments) - 1:
                await asyncio.sleep(self.segment_interval)

    def _split_into_segments(self, text: str) -> list[str]:
        """Split text into segments using the configured regex (same as AstrBot's segmented_reply).

        The separator config is a regex pattern used with re.findall() to extract segments,
        NOT a literal delimiter for str.split(). This matches AstrBot's built-in behavior.
        Example: regex '[^\\n$\\\\]+' extracts all runs of chars that aren't newline, $, or \\.
        """
        if self.segment_separator:
            try:
                segments = re.findall(
                    self.segment_separator, text, re.DOTALL | re.MULTILINE
                )
                # Filter out empty/whitespace-only segments
                segments = [seg.strip() for seg in segments if seg.strip()]
                if segments:
                    return segments
            except re.error as e:
                logger.error(
                    f"[ActiveFunction] Segment regex error: {e}, falling back to no split"
                )
        return [text]

    async def _delayed_recall(self, bot, message_id: int):
        """Wait for the configured delay, then recall the message."""
        await asyncio.sleep(self.recall_delay)
        try:
            await bot.delete_msg(message_id=message_id)
            logger.info(f"[ActiveFunction] Recalled message: {message_id}")
        except Exception as e:
            logger.error(f"[ActiveFunction] Failed to recall message {message_id}: {e}")

    # ==================== NEXT tag cleanup (fallback) ====================

    @filter.on_decorating_result(priority=1)
    @filter.on_decorating_result(priority=5)
    async def fallback_strip_control_tags(self, event: AstrMessageEvent):
        """Fallback cleanup: strip all control tags that weren't handled by higher-priority hooks.

        This catches edge cases where tags leak to users:
        - Plugin disabled but LLM still outputs tags
        - Malformed tags (uppercase Reply, negative IDs, wrong brackets)
        - Non-supported platforms
        - Session-level plugin disabled

        Runs at low priority (5) after all functional handlers have finished.
        """
        if await self._session_inactive(event):
            return
        
        # Skip if already handled by functional handlers
        if event.get_extra("_recall_handled"):
            return
        
        result = event.get_result()
        if not result or not result.chain:
            return

        # Check if any control tags are present
        full_text = "".join(comp.text for comp in result.chain if isinstance(comp, Plain))
        if not full_text:
            return
        
        has_tags = (
            (self.recall_tag and self.recall_tag in full_text)
            or (self.poke_tag and self.poke_tag in full_text)
            or self._reply_mgr.has_reply_tags(full_text)
        )
        
        if not has_tags:
            return
        
        # Strip all control tag variants
        logger.warning(
            f"[ActiveFunction] Fallback cleanup: detected unhandled control tags, "
            f"stripping from output. Session: {event.unified_msg_origin}"
        )
        
        for i, comp in enumerate(result.chain):
            if isinstance(comp, Plain):
                cleaned = comp.text
                # Strip recall tag
                if self.recall_tag:
                    cleaned = cleaned.replace(self.recall_tag, "")
                # Strip poke tag
                if self.poke_tag:
                    cleaned = cleaned.replace(self.poke_tag, "")
                # Strip all reply tag variants (including malformed ones)
                cleaned = self._reply_mgr.strip_all_reply_variants(cleaned)
                # Clean up whitespace
                cleaned = re.sub(r"  +", " ", cleaned).strip()
                if cleaned != comp.text:
                    result.chain[i] = Plain(cleaned)

    # ==================== NEXT tag cleanup (fallback) ====================

    async def strip_next_tags_fallback(self, event: AstrMessageEvent):
        """Low-priority fallback: strip [NEXT: Xm] tags from any remaining Plain text in result chain.

        This ensures the wakeup plugin's scheduling tags never leak to the user,
        even when the message doesn't go through reply/recall handling paths.
        """
        if await self._session_inactive(event):
            return
        result = event.get_result()
        if not result or not result.chain:
            return

        for i, comp in enumerate(result.chain):
            if isinstance(comp, Plain) and _NEXT_TAG_PATTERN.search(comp.text):
                cleaned = _NEXT_TAG_PATTERN.sub("", comp.text).strip()
                result.chain[i] = Plain(cleaned)
