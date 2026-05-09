import asyncio
import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Record
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from .poke import PokeManager
from .reply import ReplyManager


@register(
    "astrbot_plugin_active_function",
    "YourName",
    "为 Bot 提供主动能力：撤回消息、引用回复、戳一戳等",
    "0.1.0",
)
class ActiveFunctionPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

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
            recall_tag=self.recall_tag,
            segment_separator=self.segment_separator,
        )

        # poke config
        poke_cfg = config.get("poke", {})
        self.poke_enable: bool = poke_cfg.get("enable", True)
        self.poke_tag: str = poke_cfg.get("poke_tag", "[poke]")
        self.poke_prompt_template: str = prompt_cfg.get(
            "poke_prompt",
            "你可以在回复中使用 {poke_tag} 标签来戳对方。"
            "当你想表达亲昵、调皮、或者引起对方注意时可以使用它。"
            "标签可以放在回复文本的任意位置，系统会在发送该段消息后执行戳一戳动作。"
            '示例："哼，不理你了 {poke_tag}"。'
            "注意：每条回复最多使用一次 {poke_tag}。",
        )

        # Initialize poke manager
        self._poke_mgr = PokeManager(
            enable=self.poke_enable,
            cooldown=poke_cfg.get("cooldown", 5.0),
            poke_prompt=poke_cfg.get("poke_prompt", ""),
            poke_tag=self.poke_tag,
        )

    async def initialize(self):
        """Plugin initialization."""
        logger.info(
            f"[ActiveFunction] Initialized. Recall enabled={self.recall_enable}, "
            f"delay={self.recall_delay}s, tag='{self.recall_tag}'. "
            f"Reply enabled={self.reply_enable}. "
            f"Poke enabled={self.poke_enable}, tag='{self.poke_tag}'."
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

    def _build_system_prompt_suffix(self, event: AstrMessageEvent | None = None) -> str:
        """Build the system prompt suffix from configured templates."""
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
            and not event.get_group_id()
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

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def cache_user_message(self, event: AstrMessageEvent):
        """Cache original user message before debounce plugin merges it.

        Priority 100 ensures this runs BEFORE the debounce plugin (priority 50).
        We do NOT stop the event, so debounce can still merge messages normally.
        """
        if not self.reply_enable:
            return
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if event.get_group_id():
            return

        message_id = self._reply_mgr.extract_message_id(event)
        if message_id is None:
            logger.debug(
                "[ActiveFunction] Reply cache: cannot extract message_id, skipping"
            )
            return

        text = (event.message_str or "").strip()
        if not text:
            return

        self._reply_mgr.cache_message(event.unified_msg_origin, message_id, text)

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

        # Private chat only
        if event.get_group_id():
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

        # Private chat only
        if event.get_group_id():
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # Get plain text from chain
        full_text = ""
        for comp in result.chain:
            if isinstance(comp, Plain):
                full_text += comp.text

        if not self._poke_mgr.has_poke_tag(full_text):
            return

        # Always strip [poke] tags from text to prevent leaking to users
        for i, comp in enumerate(result.chain):
            if isinstance(comp, Plain) and self.poke_tag in comp.text:
                result.chain[i] = Plain(comp.text.replace(self.poke_tag, ""))

        # If poke is disabled, just strip the tag and return
        if not self.poke_enable:
            logger.debug("[ActiveFunction] Poke disabled, stripped tag from output")
            return

        logger.info("[ActiveFunction] on_decorating_result: handling poke tag")

        # Get bot instance: try event.bot first, then fall back to platform instance
        bot = getattr(event, "bot", None)
        if bot is None:
            bot = self._get_bot_from_platform(event)

        if bot is None:
            logger.debug("[ActiveFunction] Cannot get bot instance for poke action")
            return

        # Execute poke action
        target_user_id = int(event.get_sender_id())
        success = await self._poke_mgr.send_poke(bot, target_user_id)
        if success:
            logger.info(f"[ActiveFunction] Poke sent to user {target_user_id}")
        else:
            logger.debug(
                f"[ActiveFunction] Poke action failed for user {target_user_id}"
            )

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

    # ==================== Reply: on_decorating_result ====================

    @filter.on_decorating_result(priority=11)
    async def handle_reply_decorate(self, event: AstrMessageEvent):
        """Parse [reply:ID] tags from LLM output and send as QQ native replies.

        Priority 11 ensures this runs before the recall handler (priority 10).
        If reply tags are found, this handler takes over sending entirely.
        If no reply tags but recall tags exist, this handler does nothing and
        lets the recall handler process it.

        When reply is disabled, tags are still stripped to prevent leaking to users.

        This handler also preserves non-Plain components (e.g. Record from TTS plugin)
        and sends them alongside the text segments.
        """
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if event.get_group_id():
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # Get plain text from chain
        full_text = ""
        for comp in result.chain:
            if isinstance(comp, Plain):
                full_text += comp.text

        if not self._reply_mgr.has_reply_tags(full_text):
            return

        # If reply is disabled, just strip the tags and let normal sending proceed
        if not self.reply_enable:
            logger.debug("[ActiveFunction] Reply disabled, stripping reply tags from output")
            stripped_text = re.sub(r"\[reply:\d+\]", "", full_text)
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

        logger.info("[ActiveFunction] on_decorating_result: handling reply tags")

        # Collect non-Plain components (e.g. Record from TTS) in order
        non_plain_components: list[BaseMessageComponent] = [
            comp for comp in result.chain if not isinstance(comp, Plain)
        ]

        # Parse into segments (handles both reply and recall tags)
        session_key = event.unified_msg_origin
        segments = self._reply_mgr.parse_segments(full_text, session_key)

        if not segments:
            return

        # Send all segments, passing non-plain components for proper handling
        await self._send_reply_segments(event, segments, non_plain_components)

        # Prevent the framework from sending anything
        event.clear_result()
        event._has_send_oper = True
        # Mark as handled so recall handler and after_message_sent don't double-process
        event.set_extra("_recall_handled", True)

    async def _send_reply_segments(
        self,
        event: AiocqhttpMessageEvent,
        segments: list,
        non_plain_components: list[BaseMessageComponent] | None = None,
    ) -> None:
        """Send parsed reply segments with Reply components and recall scheduling.

        Also handles non-Plain components (e.g. Record from TTS plugin) by sending
        them after all text segments have been sent.
        """
        bot = event.bot
        uid = event.get_sender_id()

        for i, segment in enumerate(segments):
            if not segment.text:
                continue

            # Build message payload
            message = []
            if segment.reply_message_id is not None:
                message.append(
                    {"type": "reply", "data": {"id": str(segment.reply_message_id)}}
                )
            message.append({"type": "text", "data": {"text": segment.text}})

            try:
                send_result = await bot.send_private_msg(
                    user_id=int(uid),
                    message=message,
                )
            except Exception as e:
                logger.error(f"[ActiveFunction] Failed to send reply segment: {e}")
                continue

            # Schedule recall if this segment also has a recall tag
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

            # Interval between segments
            if i < len(segments) - 1:
                await asyncio.sleep(self.segment_interval)

        # Send non-Plain components (e.g. Record/voice from TTS plugin)
        if non_plain_components:
            for comp in non_plain_components:
                if isinstance(comp, Record):
                    await self._send_record_component(bot, int(uid), comp)
                    await asyncio.sleep(self.segment_interval)
                else:
                    # For other component types, try generic dict conversion
                    try:
                        msg_dict = comp.toDict()
                        await bot.send_private_msg(
                            user_id=int(uid), message=[msg_dict]
                        )
                        await asyncio.sleep(self.segment_interval)
                    except Exception as e:
                        logger.debug(
                            f"[ActiveFunction] Skipped non-Plain component: {e}"
                        )

    async def _send_record_component(
        self, bot, user_id: int, record: Record
    ) -> None:
        """Send a Record (voice) component via OneBot API."""
        try:
            # Convert Record to base64 format for sending (same as aiocqhttp platform)
            bs64 = await record.convert_to_base64()
            message = [{"type": "record", "data": {"file": f"base64://{bs64}"}}]
            await bot.send_private_msg(user_id=user_id, message=message)
            logger.info(f"[ActiveFunction] Sent voice message to user {user_id}")
        except Exception as e:
            logger.error(f"[ActiveFunction] Failed to send voice: {e}")
            # Fallback: send the text content if available
            if record.text:
                try:
                    await bot.send_private_msg(
                        user_id=user_id,
                        message=[{"type": "text", "data": {"text": record.text}}],
                    )
                except Exception:
                    pass

    # ==================== Prompt Injection ====================

    @filter.on_llm_request()
    async def inject_prompt(self, event: AstrMessageEvent, request: ProviderRequest):
        """Inject function usage instructions into the LLM system prompt."""
        if not self.prompt_enable:
            return
        suffix = self._build_system_prompt_suffix(event)
        if suffix:
            request.system_prompt += "\n\n" + suffix

    # ==================== LLM Response Hook ====================

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """When LLM responds, check if recall tag is present and mark the event.

        This fires BEFORE the result is set and BEFORE history is saved.
        We mark the event so that after_message_sent can handle recall.
        We do NOT modify the response here - history should contain the full text.
        """
        if not self.recall_enable:
            return
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        if event.get_group_id():
            return

        text = response.completion_text if response else ""
        if not text or self.recall_tag not in text:
            return

        # Mark this event as needing recall processing
        event.set_extra("_recall_pending", True)
        event.set_extra("_recall_full_text", text)
        logger.info("[ActiveFunction] LLM response contains recall tag, marked event.")

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
        if event.get_group_id():
            return

        # Get plain text from chain
        full_text = ""
        for comp in result.chain:
            if isinstance(comp, Plain):
                full_text += comp.text

        if self.recall_tag not in full_text:
            return

        # If recall is disabled, just strip the tags and let normal sending proceed
        if not self.recall_enable:
            logger.debug("[ActiveFunction] Recall disabled, stripping recall tags from output")
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
        if event.get_group_id():
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
        uid = event.get_sender_id()

        for i, segment in enumerate(segments):
            # Check if this segment starts with the recall tag (after stripping)
            needs_recall = segment.startswith(self.recall_tag)
            # Remove the recall tag from displayed text
            display_text = segment.removeprefix(self.recall_tag).strip()

            if not display_text:
                continue

            # Send via bot API to get message_id
            try:
                send_result = await bot.send_private_msg(
                    user_id=int(uid),
                    message=[{"type": "text", "data": {"text": display_text}}],
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
