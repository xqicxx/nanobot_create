"""MemU-backed memory adapter for nanobot (memu-py 0.2.x API)."""

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
    "好", "嗯", "嗯嗯", "哈", "哈哈", "ok", "okay", "okey",
    "收到", "谢谢", "谢谢你", "谢了", "行", "可以", "好的",
    "了解", "了解了", "明白", "明白了", "知道了",
    "yes", "yep", "yeah", "thx", "thanks", "thankyou",
}

_DEEP_RECALL_PATTERNS = (
    r"再想想", r"再想一下", r"再仔细想", r"再回忆", r"回忆一下",
    r"再查一下记忆", r"重新检索", r"重新回忆", r"想起来了吗",
    r"think again", r"recall again", r"check memory again",
    r"search memory again",
)


@dataclass
class MemoryContext:
    text: str


class MemoryAdapter:
    """MemU-backed memory adapter for memu-py 0.2.x."""

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
        self._memory_agent = memu_service
        if self._memory_agent is None and self.enable_memory:
            self._init_agents()

    def _init_agents(self) -> None:
        """Initialize MemoryAgent with LLM client."""
        try:
            from memu.memory import MemoryAgent
            from memu.llm import OpenAIClient, DeepSeekClient
        except Exception as exc:
            logger.error(f"Failed to import MemU modules: {exc}")
            self.enable_memory = False
            return

        # Setup embedding configuration
        self._setup_embedding_env()

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
                memory_dir=memory_dir,
                enable_embeddings=True,
                agent_id="nanobot",
                user_id="default",
            )
            logger.info("MemU MemoryAgent initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to initialize MemoryAgent: {exc}")
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

        try:
            if provider == "deepseek" or "deepseek" in base_url.lower():
                return DeepSeekClient(
                    api_key=api_key,
                    endpoint=base_url,
                    model_name=chat_model,
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

    def _setup_embedding_env(self) -> None:
        """Setup embedding environment variables from config."""
        memu_cfg = self.memu_config
        embedding_cfg = getattr(memu_cfg, "embedding", None) if memu_cfg else None
        
        if embedding_cfg:
            api_key = getattr(embedding_cfg, "api_key", "")
            base_url = getattr(embedding_cfg, "base_url", "")
            embed_model = getattr(embedding_cfg, "embed_model", "BAAI/bge-m3")
            
            # Set environment variables for memu-py to read
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
            if embed_model:
                os.environ["OPENAI_EMBED_MODEL"] = embed_model

    @staticmethod
    def build_user_id(channel: str | None, chat_id: str | None, sender_id: str | None) -> str:
        return f"{channel or 'default'}:{chat_id or 'unknown'}:{sender_id or 'system'}"

    def should_skip_retrieve(self, message: str) -> bool:
        text = (message or "").strip()
        if not text:
            return True
        if len(text) <= 2:
            return True
        normalized = re.sub(r"\s+", "", text.lower())
        if normalized in _GREETING_TOKENS:
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
        """Retrieve memory context from files.
        
        In memu-py 0.2.x, memories are stored as markdown files.
        This method reads the memory files directly.
        """
        if not self.enable_memory or self._memory_agent is None:
            return MemoryContext(text="")
        
        if self.should_skip_retrieve(current_message):
            return MemoryContext(text="")
        
        try:
            # Build user path (memu-py 0.2.x uses agent_id/user_id structure)
            user_id = self.build_user_id(channel, chat_id, sender_id)
            memory_dir = self.workspace / ".memu" / "memory"
            user_memory_dir = memory_dir / "nanobot" / user_id
            
            logger.debug(f"Looking for memories in: {user_memory_dir}")
            
            if not user_memory_dir.exists():
                logger.debug(f"Memory directory not found: {user_memory_dir}")
                # Try to find any memory directories
                if memory_dir.exists():
                    import os
                    for root, dirs, files in os.walk(memory_dir):
                        if files:
                            logger.debug(f"Found files in: {root} - {files}")
                return MemoryContext(text="")
            
            # Read all memory files
            memories = []
            memory_files = {
                "profile": "个人档案",
                "event": "重要事件", 
                "reminder": "提醒事项",
                "interest": "兴趣爱好",
                "study": "学习记录",
                "activity": "活动记录"
            }
            
            for category, label in memory_files.items():
                file_path = user_memory_dir / f"{category}.md"
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8").strip()
                    if content and len(content) > 10:  # Filter out empty/short content
                        # Extract first few lines as summary
                        lines = [line.strip() for line in content.split("\n") if line.strip()]
                        if lines:
                            summary = "; ".join(lines[:3])  # First 3 lines
                            memories.append(f"[{label}] {summary}")
            
            if not memories:
                return MemoryContext(text="")
            
            # Format as context
            context_text = "# 历史记忆\n" + "\n".join(f"- {m}" for m in memories[:self.retrieve_top_k])
            return MemoryContext(text=context_text)
            
        except Exception as exc:
            logger.warning(f"Failed to retrieve memory context: {exc}")
            return MemoryContext(text="")

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
        """Memorize a conversation turn.
        
        Note: In memu-py 0.2.x, memorization is done via run() method.
        This method uses run() to process the conversation.
        """
        if not self.enable_memory or self._memory_agent is None:
            return

        user_id = self.build_user_id(channel, chat_id, sender_id)

        # Format conversation for memu-py 0.2.x
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        try:
            start = time.perf_counter()
            result = self._memory_agent.run(
                conversation=conversation,
                character_name=user_id,
                max_iterations=5,  # Limit iterations for performance
            )
            elapsed_ms = int(round((time.perf_counter() - start) * 1000))
            
            if result.get("success"):
                logger.info(
                    "MemU memorize in {}ms (channel={}, sender={})",
                    elapsed_ms,
                    channel,
                    sender_id,
                )
            else:
                logger.warning(f"MemU memorize failed: {result.get('error')}")
        except Exception as exc:
            logger.warning(f"MemU memorize failed: {exc}")

    def list_category_config(self) -> list[dict[str, Any]]:
        """List available memory categories."""
        return []

    def add_category_config(self, name: str, description: str = "") -> dict[str, Any]:
        return {"ok": False, "error": "Category management not supported in memu-py 0.2.x"}

    def update_category_config(self, name: str, description: str) -> dict[str, Any]:
        return {"ok": False, "error": "Category management not supported in memu-py 0.2.x"}

    def delete_category_config(self, name: str) -> dict[str, Any]:
        return {"ok": False, "error": "Category management not supported in memu-py 0.2.x"}

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
            "version": "0.2.x",
        }

        if not status["enabled"]:
            return status

        try:
            agent_status = self._memory_agent.get_status() if self._memory_agent else {}
            status["memory"] = {
                "agent_name": agent_status.get("agent_name"),
                "architecture": agent_status.get("architecture"),
                "memory_types": agent_status.get("memory_types", []),
                "embeddings_enabled": agent_status.get("embedding_capabilities", {}).get("embeddings_enabled", False),
            }
        except Exception as exc:
            status["health"] = {"ok": False, "error": str(exc)}

        return status

    async def query_items(self, **kwargs) -> list[dict[str, Any]]:
        return []

    async def query_categories(self, **kwargs) -> list[dict[str, Any]]:
        return []

    async def clear(self, **kwargs) -> dict[str, Any]:
        return {"ok": False, "error": "Clear not supported in memu-py 0.2.x"}

    async def health(self, **kwargs) -> dict[str, Any]:
        return {"ok": self.enable_memory}

    def mark_restart_required(self, *, reason: str | None = None) -> dict[str, Any]:
        return {"ok": False, "error": "Not supported"}

    def clear_restart_required(self) -> bool:
        return False

    def is_restart_required(self) -> bool:
        return False

    def get_retrieve_tuning(self) -> dict[str, int]:
        return {
            "top_k": int(self.retrieve_top_k),
            "top_k_full": int(self.retrieve_top_k_full),
            "history_window": int(self.retrieve_history_window),
            "history_window_full": int(self.retrieve_history_window_full),
        }

    def update_retrieve_tuning(self, **kwargs) -> dict[str, Any]:
        return {"ok": False, "error": "Tuning update not implemented"}

    def should_skip_write(self, message: str, previous_user_message: str | None = None) -> bool:
        text = message.strip()
        if not text:
            return True
        normalized = re.sub(r"\s+", "", text.lower())
        if normalized in _GREETING_TOKENS:
            return True
        if previous_user_message is not None and text == previous_user_message.strip():
            return True
        return False

    def should_force_full_retrieve(self, message: str) -> bool:
        text_raw = (message or "").strip()
        if not text_raw:
            return False
        text_lower = text_raw.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in _DEEP_RECALL_PATTERNS)
