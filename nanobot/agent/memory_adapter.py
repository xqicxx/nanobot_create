"""MemU-backed memory adapter for nanobot (MemoryAgent API)."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

if TYPE_CHECKING:
    from nanobot.config.schema import MemuConfig, MemuLLMConfig


_GREETING_TOKENS = {
    "好",
    "嗯",
    "嗯嗯",
    "哈",
    "哈哈",
    "ok",
    "okay",
    "okey",
    "收到",
    "谢谢",
    "谢谢你",
    "谢了",
    "行",
    "可以",
    "好的",
    "了解",
    "了解了",
    "明白",
    "明白了",
    "知道了",
    "yes",
    "yep",
    "yeah",
    "thx",
    "thanks",
    "thankyou",
}

_DEEP_RECALL_PATTERNS = (
    r"再想想",
    r"再想一下",
    r"再仔细想",
    r"再回忆",
    r"回忆一下",
    r"再查一下记忆",
    r"重新检索",
    r"重新回忆",
    r"想起来了吗",
    r"think again",
    r"recall again",
    r"check memory again",
    r"search memory again",
    r"(还|能|有没有).{0,6}(记得|回忆|想起)",
    r"(上次|之前|前面|\bearlier\b|\bpreviously\b).{0,10}(说|聊|提|讲|tell|mention|discuss)",
    r"(忘了|不记得|想不起来).{0,6}(上次|之前|那个|那件事)",
)

_DEEP_RECALL_MEMORY_TERMS = {
    "记忆",
    "记得",
    "回忆",
    "想起",
    "想起来",
    "memory",
    "remember",
    "recall",
}

_DEEP_RECALL_REFERENCE_TERMS = {
    "上次",
    "之前",
    "前面",
    "刚才",
    " earlier ",
    " previous ",
    " before ",
    "last time",
    "we said",
    "we talked",
    "you said",
}

_DEEP_RECALL_ACTION_TERMS = {
    "再",
    "重新",
    "再查",
    "再看",
    "检索",
    "搜索",
    "回顾",
    "再想",
    "again",
    "re-",
    "search",
    "check",
    "look up",
}

_DEEP_RECALL_UNCERTAIN_TERMS = {
    "忘了",
    "不记得",
    "想不起来",
    "没印象",
    "记不清",
    "forgot",
    "can't remember",
    "cannot remember",
    "not sure",
}


def _contains_term(text_lower: str, text_flat: str, term: str) -> bool:
    t = (term or "").lower().strip()
    if not t:
        return False
    if " " in t:
        return t in text_lower
    return t in text_flat


def _count_terms(text_lower: str, text_flat: str, terms: set[str]) -> int:
    return sum(1 for term in terms if _contains_term(text_lower, text_flat, term))


def _normalize_greeting(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _is_pure_emoji_or_punct(text: str) -> bool:
    has_non_ws = False
    for ch in text:
        if ch.isspace():
            continue
        has_non_ws = True
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        if cat == "So":
            continue
        return False
    return has_non_ws


def _only_greeting_or_ack(text: str) -> bool:
    normalized = _normalize_greeting(text)
    return normalized in _GREETING_TOKENS


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


@dataclass
class MemoryContext:
    text: str


class MemoryAdapter:
    """MemU-backed memory adapter using MemoryAgent API."""

    def __init__(
        self,
        *,
        workspace: Path,
        memu_service: Any | None = None,
        resources_dir: Path | None = None,
        retrieve_top_k: int = 5,
        enable_memory: bool = True,
        memu_config: "MemuConfig | None" = None,
    ):
        self.workspace = workspace
        self.retrieve_top_k = retrieve_top_k
        self.retrieve_history_window = max(1, int(os.getenv("NANOBOT_MEMU_HISTORY_WINDOW", "4")))
        self.retrieve_top_k_full = max(self.retrieve_top_k, int(os.getenv("NANOBOT_MEMU_RETRIEVE_TOP_K_FULL", "12")))
        self.retrieve_history_window_full = max(
            self.retrieve_history_window,
            int(os.getenv("NANOBOT_MEMU_HISTORY_WINDOW_FULL", "12")),
        )
        if memu_config is not None and not memu_config.enabled:
            enable_memory = False
        self.enable_memory = enable_memory
        self.resources_dir = ensure_dir(resources_dir or (workspace / ".memu" / "resources"))
        self.memu_config = memu_config
        self._memory_agent = None
        self._recall_agent = None
        self._init_agents() if self.enable_memory else None

    def _init_agents(self) -> None:
        """Initialize MemoryAgent and RecallAgent with LLM client."""
        try:
            from memu.memory import MemoryAgent, RecallAgent
            from memu.llm import OpenAIClient, DeepSeekClient
        except Exception as exc:
            logger.error(f"Failed to import MemU modules: {exc}")
            self.enable_memory = False
            return

        # Build LLM client from config
        llm_client = self._create_llm_client()
        if llm_client is None:
            logger.error("Failed to create LLM client for MemU")
            self.enable_memory = False
            return

        memory_dir = str(self.workspace / ".memu" / "memory")

        try:
            self._memory_agent = MemoryAgent(
                llm_client=llm_client,
                agent_id="nanobot",
                user_id="default",
                memory_dir=memory_dir,
                enable_embeddings=True,
            )
            self._recall_agent = RecallAgent(
                memory_dir=memory_dir,
                agent_id="nanobot",
                user_id="default",
            )
            logger.info("MemU agents initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to initialize MemU agents: {exc}")
            self.enable_memory = False

    def _create_llm_client(self) -> Any | None:
        """Create LLM client from configuration."""
        try:
            from memu.llm import OpenAIClient, DeepSeekClient
        except Exception:
            return None

        memu_cfg = self.memu_config
        default_cfg = getattr(memu_cfg, "default", None) if memu_cfg else None

        if default_cfg is None:
            return None

        provider = getattr(default_cfg, "provider", "openai")
        api_key = getattr(default_cfg, "api_key", "") or os.getenv("DEEPSEEK_API_KEY", "")
        base_url = getattr(default_cfg, "base_url", "") or "https://api.deepseek.com/v1"
        chat_model = getattr(default_cfg, "chat_model", "deepseek-chat")
        client_backend = getattr(default_cfg, "client_backend", "sdk")

        try:
            if provider == "deepseek" or "deepseek" in base_url.lower():
                return DeepSeekClient(
                    api_key=api_key,
                    base_url=base_url,
                    model=chat_model,
                )
            else:
                return OpenAIClient(
                    api_key=api_key,
                    base_url=base_url,
                    model=chat_model,
                )
        except Exception as exc:
            logger.error(f"Failed to create LLM client: {exc}")
            return None

    def get_retrieve_tuning(self) -> dict[str, int]:
        return {
            "top_k": int(self.retrieve_top_k),
            "top_k_full": int(self.retrieve_top_k_full),
            "history_window": int(self.retrieve_history_window),
            "history_window_full": int(self.retrieve_history_window_full),
        }

    def update_retrieve_tuning(
        self,
        *,
        top_k: int | None = None,
        top_k_full: int | None = None,
        history_window: int | None = None,
        history_window_full: int | None = None,
    ) -> dict[str, Any]:
        def _require_positive(name: str, value: int | None) -> tuple[int | None, str | None]:
            if value is None:
                return None, None
            if not isinstance(value, int) or value <= 0:
                return None, f"{name} must be a positive integer"
            return value, None

        top_k_v, err = _require_positive("top_k", top_k)
        if err:
            return {"ok": False, "error": err}
        top_k_full_v, err = _require_positive("top_k_full", top_k_full)
        if err:
            return {"ok": False, "error": err}
        history_v, err = _require_positive("history_window", history_window)
        if err:
            return {"ok": False, "error": err}
        history_full_v, err = _require_positive("history_window_full", history_window_full)
        if err:
            return {"ok": False, "error": err}

        if top_k_v is not None:
            self.retrieve_top_k = top_k_v
        if history_v is not None:
            self.retrieve_history_window = history_v

        if top_k_full_v is not None:
            self.retrieve_top_k_full = top_k_full_v
        if history_full_v is not None:
            self.retrieve_history_window_full = history_full_v

        self.retrieve_top_k_full = max(self.retrieve_top_k, self.retrieve_top_k_full)
        self.retrieve_history_window_full = max(self.retrieve_history_window, self.retrieve_history_window_full)

        return {"ok": True, "tuning": self.get_retrieve_tuning()}

    @staticmethod
    def build_user_id(channel: str | None, chat_id: str | None, sender_id: str | None) -> str:
        return f"{channel or 'default'}:{chat_id or 'unknown'}:{sender_id or 'system'}"

    def should_skip_write(self, message: str, previous_user_message: str | None = None) -> bool:
        text = message.strip()
        if not text:
            return True
        if _is_pure_emoji_or_punct(text):
            return True
        if _only_greeting_or_ack(text):
            return True
        if previous_user_message is not None and text == previous_user_message.strip():
            return True
        return False

    def should_skip_retrieve(self, message: str) -> bool:
        text = (message or "").strip()
        if not text:
            return True
        if self.should_force_full_retrieve(text):
            return False
        if _is_pure_emoji_or_punct(text):
            return True
        if _only_greeting_or_ack(text):
            return True
        if len(text) <= 4 and not re.search(r"[?？]|怎么|为何|为什么|谁|啥|吗|what|how|why|who|where|when", text, re.IGNORECASE):
            return True
        return False

    def should_force_full_retrieve(self, message: str) -> bool:
        text_raw = (message or "").strip()
        if not text_raw:
            return False
        text_lower = text_raw.lower()
        text_flat = re.sub(r"\s+", "", text_lower)

        if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in _DEEP_RECALL_PATTERNS):
            return True

        mem_hits = _count_terms(text_lower, text_flat, _DEEP_RECALL_MEMORY_TERMS)
        ref_hits = _count_terms(text_lower, text_flat, _DEEP_RECALL_REFERENCE_TERMS)
        action_hits = _count_terms(text_lower, text_flat, _DEEP_RECALL_ACTION_TERMS)
        uncertain_hits = _count_terms(text_lower, text_flat, _DEEP_RECALL_UNCERTAIN_TERMS)

        if mem_hits >= 1 and (ref_hits + action_hits + uncertain_hits) >= 1:
            return True
        if ref_hits >= 2 and (action_hits >= 1 or uncertain_hits >= 1):
            return True
        if ("?" in text_raw or "？" in text_raw) and (mem_hits + uncertain_hits) >= 1 and ref_hits >= 1:
            return True
        return False

    async def retrieve_context(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        history: list[dict[str, Any]],
        current_message: str,
        full_retrieve: bool = False,
    ) -> MemoryContext:
        if not self.enable_memory or self._recall_agent is None:
            return MemoryContext(text="")
        if self.should_skip_retrieve(current_message):
            return MemoryContext(text="")

        user_id = self.build_user_id(channel, chat_id, sender_id)
        top_k = self.retrieve_top_k_full if full_retrieve else self.retrieve_top_k

        try:
            # Use RecallAgent for retrieval
            result = self._recall_agent.retrieve_relevant_memories(
                agent_id="nanobot",
                user_id=user_id,
                query=current_message,
                top_k=top_k,
            )
            return MemoryContext(text=self._format_recall_result(result))
        except Exception as exc:
            logger.warning(f"MemU retrieve failed: {exc}")
            return MemoryContext(text="")

    def _format_recall_result(self, result: dict[str, Any]) -> str:
        """Format RecallAgent result into context text."""
        memories = result.get("memories", []) if isinstance(result, dict) else []
        if not memories:
            return ""

        lines: list[str] = ["# Memory (MemU)"]
        for mem in memories[:self.retrieve_top_k]:
            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            category = mem.get("category", "unknown") if isinstance(mem, dict) else "memory"
            if content:
                lines.append(f"- [{category}] {content}")

        return "\n".join(lines)

    async def memorize_turn(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enable_memory or self._memory_agent is None:
            return

        user_id = self.build_user_id(channel, chat_id, sender_id)

        # Format conversation as content
        content = f"User: {user_message}\nAssistant: {assistant_message}"

        try:
            start = time.perf_counter()
            # Use MemoryAgent.call_function to add activity memory
            result = self._memory_agent.call_function(
                "add_activity_memory",
                {
                    "character_name": user_id,
                    "content": content,
                    "session_date": datetime.now().isoformat(),
                    "generate_embeddings": True,
                }
            )
            elapsed_ms = int(round((time.perf_counter() - start) * 1000))
            logger.info(
                "MemU memorize in {}ms (channel={}, sender={}, msg_len={}, resp_len={})",
                elapsed_ms,
                channel,
                sender_id,
                len(user_message or ""),
                len(assistant_message or ""),
            )
        except Exception as exc:
            logger.warning(f"MemU memorize failed: {exc}")

    def list_category_config(self) -> list[dict[str, Any]]:
        """List available memory categories."""
        if not self.enable_memory or self._memory_agent is None:
            return []

        try:
            result = self._memory_agent.call_function("get_available_categories", {})
            if isinstance(result, dict) and result.get("success"):
                categories = result.get("categories", {})
                return [
                    {"name": name, "description": info.get("description", "")}
                    for name, info in categories.items()
                    if isinstance(info, dict)
                ]
        except Exception as exc:
            logger.warning(f"Failed to list categories: {exc}")
        return []

    def add_category_config(self, name: str, description: str = "") -> dict[str, Any]:
        """Add a new memory category."""
        # MemoryAgent doesn't support dynamic category management
        return {"ok": False, "error": "Dynamic category management not supported in MemoryAgent"}

    def update_category_config(self, name: str, description: str) -> dict[str, Any]:
        """Update a memory category."""
        return {"ok": False, "error": "Dynamic category management not supported in MemoryAgent"}

    def delete_category_config(self, name: str) -> dict[str, Any]:
        """Delete a memory category."""
        return {"ok": False, "error": "Dynamic category management not supported in MemoryAgent"}

    async def memu_status(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        run_checks: bool = True,
        full_checks: bool = False,
    ) -> dict[str, Any]:
        """Get MemU status."""
        status: dict[str, Any] = {
            "enabled": bool(self.enable_memory and self._memory_agent is not None),
            "llm": {},
            "embedding": {},
            "memory": {},
            "health": None,
            "checks": {},
        }

        if not status["enabled"]:
            return status

        # Get status from MemoryAgent
        try:
            agent_status = self._memory_agent.get_status() if self._memory_agent else {}
            status["memory"] = {
                "agent_name": agent_status.get("agent_name"),
                "architecture": agent_status.get("architecture"),
                "memory_types": agent_status.get("memory_types", []),
                "total_actions": agent_status.get("total_actions", 0),
                "embeddings_enabled": agent_status.get("embedding_capabilities", {}).get("embeddings_enabled", False),
            }
        except Exception as exc:
            status["health"] = {"ok": False, "error": str(exc)}

        # Get status from RecallAgent
        try:
            if self._recall_agent:
                recall_status = self._recall_agent.get_status()
                status["memory"]["semantic_search"] = recall_status.get("semantic_search_enabled", False)
        except Exception:
            pass

        return status

    async def query_items(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        limit: int = 20,
        offset: int = 0,
        order_by: str | None = "-created_at",
        time_range: dict[str, str] | None = None,
        memory_type: str | None = None,
        category_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query memorized items (simplified)."""
        if not self.enable_memory or self._recall_agent is None:
            return []

        user_id = self.build_user_id(channel, chat_id, sender_id)

        try:
            result = self._recall_agent.retrieve_relevant_memories(
                agent_id="nanobot",
                user_id=user_id,
                query="*",  # Get all
                top_k=limit,
            )
            memories = result.get("memories", []) if isinstance(result, dict) else []
            return [
                {
                    "content": mem.get("content", ""),
                    "memory_type": mem.get("category", "unknown"),
                    "created_at": mem.get("timestamp", ""),
                }
                for mem in memories if isinstance(mem, dict)
            ]
        except Exception as exc:
            logger.warning(f"Failed to query items: {exc}")
            return []

    async def query_categories(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        limit: int = 50,
        offset: int = 0,
        order_by: str | None = "-created_at",
        time_range: dict[str, str] | None = None,
        category_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query memory categories."""
        categories = self.list_category_config()
        return categories[offset:offset + limit]

    async def clear(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
    ) -> dict[str, Any]:
        """Clear memories (not supported in MemoryAgent)."""
        return {"ok": False, "error": "Clear operation not supported in MemoryAgent"}

    async def health(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        include_counts: bool = False,
    ) -> dict[str, Any]:
        """Health check."""
        status = await self.memu_status(
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            run_checks=False,
        )
        return status.get("health") or {"ok": status.get("enabled", False)}

    def mark_restart_required(self, *, reason: str | None = None) -> dict[str, Any]:
        """Mark restart required (not applicable)."""
        return {"ok": False, "error": "Not supported in MemoryAgent"}

    def clear_restart_required(self) -> bool:
        """Clear restart required flag."""
        return False

    def is_restart_required(self) -> bool:
        """Check if restart is required."""
        return False
