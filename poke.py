"""Poke feature module: handle poke events and [poke] tag in LLM output.

This module provides poke (戳一戳) functionality by:
1. Listening for poke notice events (user pokes bot) and injecting readable text
   into the message so it merges with the normal debounce/LLM pipeline
2. Parsing [poke] tags from LLM output and executing poke actions via OneBot API
3. Currently supports aiocqhttp private chat only
"""

from __future__ import annotations

import time
from string import Template
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    pass

# Default poke prompt template ($username will be substituted)
DEFAULT_POKE_PROMPT = (
    "$username 戳了戳你。\n\n"
    "【重要】请按优先级判断并回应：\n\n"
    "1. 首先检查上下文中用户最近的消息：\n"
    "   - 如果用户刚发了消息但没@你 → 直接回应那条消息\n"
    "   - 如果有未回答的问题 → 回答它\n"
    "   - 如果有正在讨论的话题 → 继续该话题\n\n"
    "2. 如果你之前说了什么，用户可能在回应 → 顺着对话继续\n\n"
    "3. 只有当上下文完全为空时 → 才可以俏皮回应戳一戳本身\n\n"
    "不要主动开新话题。优先延续现有对话。"
)

# Cooldown cleanup constants
COOLDOWN_EXPIRE_SECONDS = 600
CLEANUP_INTERVAL = 50


class PokeManager:
    """Manages poke event detection, cooldown, prompt formatting, and poke action execution."""

    def __init__(
        self,
        enable: bool = True,
        cooldown: float = 5.0,
        poke_prompt_private: str = "",
        poke_prompt_group: str = "",
        poke_tag: str = "[poke]",
    ):
        self.enable = enable
        self.poke_tag = poke_tag

        # Safely parse cooldown
        try:
            self.cooldown = float(cooldown) if cooldown is not None else 5.0
        except (ValueError, TypeError):
            self.cooldown = 5.0

        self.poke_prompt_private = poke_prompt_private or DEFAULT_POKE_PROMPT
        self.poke_prompt_group = poke_prompt_group or "[这是一个真实的戳一戳动作] 用户 $userid 戳了戳你。"

        # Cooldown tracking: user_id -> last_poke_time
        self._cooldown_map: dict[str, float] = {}
        self._poke_count = 0

    # ==================== Poke Event Detection ====================

    def is_poke_event(self, raw_message: Any) -> bool:
        """Check if a raw message is a poke notice event."""
        if not raw_message:
            return False

        # Handle both dict and object-like access
        if isinstance(raw_message, dict):
            return (
                raw_message.get("post_type") == "notice"
                and raw_message.get("notice_type") == "notify"
                and raw_message.get("sub_type") == "poke"
            )

        # Object-like access (aiocqhttp Event)
        return (
            getattr(raw_message, "post_type", None) == "notice"
            and getattr(raw_message, "notice_type", None) == "notify"
            and getattr(raw_message, "sub_type", None) == "poke"
        )

    def is_poke_targeting_bot(self, raw_message: Any, bot_id: str) -> bool:
        """Check if the poke event targets the bot."""
        if isinstance(raw_message, dict):
            target_id = raw_message.get("target_id")
        else:
            target_id = getattr(raw_message, "target_id", None)

        if target_id is None or not bot_id:
            return False
        return str(target_id) == str(bot_id)

    def get_poke_sender_id(self, raw_message: Any) -> str | None:
        """Extract the sender (poker) user_id from a poke event."""
        if isinstance(raw_message, dict):
            uid = raw_message.get("user_id")
        else:
            uid = getattr(raw_message, "user_id", None)
        return str(uid) if uid is not None else None

    # ==================== Cooldown ====================

    def check_cooldown(self, user_id: str) -> bool:
        """Check if user can trigger poke. Returns True if allowed."""
        now = time.time()
        last_time = self._cooldown_map.get(user_id, 0.0)
        if now - last_time < self.cooldown:
            return False
        self._cooldown_map[user_id] = now
        return True

    def cleanup_cooldown(self) -> None:
        """Remove expired cooldown entries."""
        now = time.time()
        expired = [
            uid
            for uid, ts in self._cooldown_map.items()
            if now - ts > COOLDOWN_EXPIRE_SECONDS
        ]
        for uid in expired:
            del self._cooldown_map[uid]

    # ==================== Access Control ====================

    def should_respond(self, sender_id: str) -> tuple[bool, str]:
        """Check if the poke should be responded to.

        Returns (allowed, reason) tuple.
        """
        if not self.enable:
            return False, "poke disabled"

        if not self.check_cooldown(sender_id):
            return False, "cooldown"

        self._poke_count += 1
        if self._poke_count % CLEANUP_INTERVAL == 0:
            self.cleanup_cooldown()

        return True, "ok"

    # ==================== Prompt Formatting ====================

    def format_poke_injection(self, username: str, userid: str = "", is_group: bool = False) -> str:
        """Format the poke prompt to inject into user message text.
        
        Args:
            username: User nickname
            userid: User QQ ID
            is_group: True if in group chat, False if private chat
        """
        template_str = self.poke_prompt_group if is_group else self.poke_prompt_private
        template = Template(template_str)
        return template.safe_substitute(username=username, userid=userid)

    # ==================== Poke Tag Parsing ====================

    def has_poke_tag(self, text: str) -> bool:
        """Check if text contains the [poke] tag."""
        return self.poke_tag in text

    def strip_poke_tag(self, text: str) -> str:
        """Remove [poke] tags from text."""
        return text.replace(self.poke_tag, "").strip()

    # ==================== Poke Action Execution ====================

    async def send_poke(self, bot: Any, target_user_id: int) -> bool:
        """Send a poke action to the target user via OneBot API (private chat).

        Tries friend_poke via call_action for compatibility with NapCat/LLOneBot.
        Returns True if poke was sent successfully.
        """
        try:
            if hasattr(bot, "call_action"):
                await bot.call_action(
                    "friend_poke",
                    user_id=target_user_id,
                )
                return True
        except Exception as e:
            logger.debug(f"[ActiveFunction] Poke action failed: {e}")

        return False

    async def send_group_poke(
        self, bot: Any, group_id: int, target_user_id: int
    ) -> bool:
        """Send a poke action to a user inside a group via OneBot API.

        Uses the ``group_poke`` action (supported by NapCat / LLOneBot /
        Lagrange). The original go-cqhttp does not implement it, so failures
        are swallowed and we simply return False — the caller treats a poke as
        best-effort and never surfaces the error to the user.
        """
        try:
            if hasattr(bot, "call_action"):
                await bot.call_action(
                    "group_poke",
                    group_id=group_id,
                    user_id=target_user_id,
                )
                return True
        except Exception as e:
            logger.debug(
                f"[ActiveFunction] Group poke action not supported / failed: {e}"
            )

        return False

    @property
    def poke_count(self) -> int:
        """Total number of poke events responded to."""
        return self._poke_count
