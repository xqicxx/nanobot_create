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
        # 强制启用 MemU，不受配置影响
        self.enable_memory = True
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
            logger.warning(f"MemU modules not available: {exc}")
            logger.info("Using file-based memory storage instead")
            # 不设置 enable_memory = False，继续使用文件模式
            self._memory_agent = None
            return

        # Setup embedding configuration
        self._setup_embedding_env()

        # Build LLM client from config
        llm_client = self._create_llm_client()
        if llm_client is None:
            logger.warning("Failed to create LLM client for MemU, using file-based storage")
            self._memory_agent = None
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
            logger.warning(f"Failed to initialize MemoryAgent: {exc}")
            logger.info("Using file-based memory storage instead")
            self._memory_agent = None

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

        # Validate API key before creating client
        if not api_key or api_key == "your-deepseek-api-key" or len(api_key) < 20:
            logger.error(
                "❌ INVALID DeepSeek API KEY! "
                "Please check your config at ~/.nanobot/config.json\n"
                "   Current key: {}...\n"
                "   Get a valid key from: https://platform.deepseek.com/",
                api_key[:10] if api_key else "EMPTY"
            )
            return None
        
        try:
            if provider == "deepseek" or "deepseek" in base_url.lower():
                client = DeepSeekClient(
                    api_key=api_key,
                    base_url=base_url,
                    model_name=chat_model,
                )
                logger.info(f"✅ DeepSeek LLM client created (model: {chat_model})")
                return client
            else:
                client = OpenAIClient(
                    api_key=api_key,
                    base_url=base_url,
                    model=chat_model,
                )
                logger.info(f"✅ OpenAI LLM client created (model: {chat_model})")
                return client
        except Exception as exc:
            logger.error(f"❌ Failed to create LLM client: {exc}")
            return None

    def _setup_embedding_env(self) -> None:
        """Setup embedding environment variables from config for memu-py 0.2.x."""
        memu_cfg = self.memu_config
        embedding_cfg = getattr(memu_cfg, "embedding", None) if memu_cfg else None
        
        if embedding_cfg:
            api_key = getattr(embedding_cfg, "api_key", "")
            base_url = getattr(embedding_cfg, "base_url", "")
            embed_model = getattr(embedding_cfg, "embed_model", "BAAI/bge-m3")
            
            # Set environment variables for memu-py 0.2.x
            # Note: memu-py 0.2.x uses different env vars than 0.1.x
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                # Also set for memu-py compatibility
                os.environ["MEMU_LLM_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
                os.environ["MEMU_LLM_BASE_URL"] = base_url
            if embed_model:
                # memu-py 0.2.x uses MEMU_EMBEDDING_MODEL
                os.environ["MEMU_EMBEDDING_MODEL"] = embed_model
                # For backward compatibility
                os.environ["OPENAI_EMBED_MODEL"] = embed_model
            
            # Set provider to openai for SiliconFlow compatibility
            os.environ["MEMU_EMBEDDING_PROVIDER"] = "openai"

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
        # Only check if memory is enabled, not if MemoryAgent is initialized
        # We can read files even if MemoryAgent failed to initialize
        if not self.enable_memory:
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
            
            # Try to use MemoryAgent's storage_manager if available
            if self._memory_agent and hasattr(self._memory_agent, 'storage_manager'):
                try:
                    for category, label in memory_files.items():
                        content = self._memory_agent.storage_manager.read_memory_file(category)
                        if content and len(content) > 10:
                            lines = [line.strip() for line in content.split("\n") if line.strip()]
                            if lines:
                                summary = "; ".join(lines[:3])
                                memories.append(f"[{label}] {summary}")
                except Exception as e:
                    logger.debug(f"Failed to read via storage_manager: {e}, falling back to direct file read")
            
            # Fallback: direct file read
            if not memories:
                for category, label in memory_files.items():
                    file_path = user_memory_dir / f"{category}.md"
                    if file_path.exists():
                        try:
                            content = file_path.read_text(encoding="utf-8").strip()
                            if content and len(content) > 10:
                                lines = [line.strip() for line in content.split("\n") if line.strip()]
                                if lines:
                                    summary = "; ".join(lines[:3])
                                    memories.append(f"[{label}] {summary}")
                        except Exception as e:
                            logger.debug(f"Failed to read {file_path}: {e}")
            
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

        # Clean text to avoid encoding issues with DeepSeek API
        # Replace full-width punctuation with half-width equivalents
        def clean_text(text: str) -> str:
            replacements = {
                '\uff1a': ':',  # Full-width colon
                '\uff0c': ',',  # Full-width comma
                '\u3002': '.',  # Full-width period
                '\uff01': '!',  # Full-width exclamation
                '\uff1f': '?',  # Full-width question mark
                '\uff08': '(',  # Full-width left parenthesis
                '\uff09': ')',  # Full-width right parenthesis
                '\u201c': '"',  # Left double quote
                '\u201d': '"',  # Right double quote
                '\u2018': "'",  # Left single quote
                '\u2019': "'",  # Right single quote
            }
            for full, half in replacements.items():
                text = text.replace(full, half)
            return text

        # Format conversation for memu-py 0.2.x
        conversation = [
            {"role": "user", "content": clean_text(user_message)},
            {"role": "assistant", "content": clean_text(assistant_message)},
        ]

        # 如果没有 MemoryAgent，直接写入文件
        if self._memory_agent is None:
            try:
                self._save_to_file(
                    channel=channel,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                )
                logger.info(
                    "MemU saved to file (channel={}, sender={})",
                    channel,
                    sender_id,
                )
            except Exception as exc:
                logger.warning(f"MemU file save failed: {exc}")
            return

        try:
            start = time.perf_counter()
            result = self._memory_agent.run(
                conversation=conversation,
                character_name=user_id,
                max_iterations=2,  # Reduced from 5 to 2 for faster performance
            )
            elapsed_ms = int(round((time.perf_counter() - start) * 1000))
            
            # Check if any function calls were made (indicates successful processing)
            function_calls = result.get("function_calls", [])
            
            if result.get("success") and function_calls:
                logger.info(
                    "MemU memorize in {}ms (channel={}, sender={}) - {} actions",
                    elapsed_ms,
                    channel,
                    sender_id,
                    len(function_calls),
                )
            elif result.get("success") and not function_calls:
                # LLM call succeeded but no actions taken - likely API/auth issue
                error_msg = result.get("error", "Unknown error")
                logger.error(
                    "MemU memorize API ERROR (channel={}, sender={}): {}. "
                    "Check your DeepSeek API key configuration!",
                    channel,
                    sender_id,
                    error_msg,
                )
            else:
                logger.warning(f"MemU memorize failed: {result.get('error')}")
        except Exception as exc:
            logger.warning(f"MemU memorize failed: {exc}")

    def _save_to_file(self, *, channel, chat_id, sender_id, user_message, assistant_message):
        """Save conversation to file directly (fallback when MemoryAgent not available)."""
        from datetime import datetime
        
        user_id = self.build_user_id(channel, chat_id, sender_id)
        memory_dir = self.workspace / ".memu" / "memory" / "nanobot" / user_id
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = memory_dir / "activity.md"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## {timestamp}\n**User:** {user_message[:200]}\n**Assistant:** {assistant_message[:200]}\n"
        
        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            file_path.write_text(existing + entry, encoding="utf-8")
        else:
            file_path.write_text(f"# Activity Memory\n{entry}", encoding="utf-8")

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
        # 强制启用状态，不依赖 _memory_agent
        status: dict[str, Any] = {
            "enabled": True,
            "version": "0.2.x",
            "memory_agent_initialized": self._memory_agent is not None,
        }

        if self._memory_agent:
            try:
                agent_status = self._memory_agent.get_status()
                status["memory"] = {
                    "agent_name": agent_status.get("agent_name"),
                    "architecture": agent_status.get("architecture"),
                    "memory_types": agent_status.get("memory_types", []),
                    "embeddings_enabled": agent_status.get("embedding_capabilities", {}).get("embeddings_enabled", False),
                }
            except Exception as exc:
                status["health"] = {"ok": False, "error": str(exc)}
        else:
            status["health"] = {"ok": True, "note": "MemoryAgent not initialized, using file-based storage"}

        return status

    async def query_items(self, **kwargs) -> list[dict[str, Any]]:
        """Query memory items from files."""
        if not self.enable_memory:
            return []
        
        channel = kwargs.get("channel")
        chat_id = kwargs.get("chat_id")
        sender_id = kwargs.get("sender_id")
        limit = kwargs.get("limit", 20)
        
        user_id = self.build_user_id(channel, chat_id, sender_id)
        memory_dir = self.workspace / ".memu" / "memory"
        user_memory_dir = memory_dir / "nanobot" / user_id
        
        if not user_memory_dir.exists():
            # Try default user
            user_memory_dir = memory_dir / "nanobot" / "default:unknown:system"
        
        if not user_memory_dir.exists():
            return []
        
        items = []
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
                try:
                    content = file_path.read_text(encoding="utf-8").strip()
                    if content:
                        items.append({
                            "content": content[:500],
                            "memory_type": label,
                            "category": category,
                        })
                except Exception:
                    pass
        
        return items[:limit]

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
