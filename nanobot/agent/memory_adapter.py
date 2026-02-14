"""MemU-backed memory adapter for nanobot."""

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
    """MemU-backed memory adapter."""

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
        self._memu = memu_service or (self._init_memu_service() if self.enable_memory else None)

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

        # Full mode should never be narrower than base mode.
        self.retrieve_top_k_full = max(self.retrieve_top_k, self.retrieve_top_k_full)
        self.retrieve_history_window_full = max(self.retrieve_history_window, self.retrieve_history_window_full)

        retrieve_cfg = getattr(self._memu, "retrieve_config", None) if self._memu is not None else None
        item_cfg = getattr(retrieve_cfg, "item", None) if retrieve_cfg is not None else None
        if item_cfg is not None and hasattr(item_cfg, "top_k"):
            try:
                item_cfg.top_k = int(self.retrieve_top_k)
            except Exception:
                pass

        return {"ok": True, "tuning": self.get_retrieve_tuning()}

    def _init_memu_service(self) -> Any:
        try:
            from memu.app import MemoryService
        except Exception as exc:  # pragma: no cover - import path depends on env
            msg = (
                "MemU is not available. Install memu-py or add it to PYTHONPATH. "
                "Expected import path: memu.app.MemoryService"
            )
            raise RuntimeError(msg) from exc

        retrieve_config = {
            "route_intention": False,
            "sufficiency_check": True,
            "item": {"top_k": self.retrieve_top_k, "ranking": "salience"},
        }
        memorize_config = {
            "enable_item_reinforcement": True,
        }
        blob_config = {
            "resources_dir": str(self.resources_dir),
        }
        def _cfg_value(
            cfg: "MemuLLMConfig | None",
            attr: str,
            env_key: str,
            default: str,
        ) -> str:
            if cfg is not None:
                val = getattr(cfg, attr, "") or ""
                if val:
                    return val
            return os.getenv(env_key, default)

        memu_cfg = self.memu_config
        default_cfg = getattr(memu_cfg, "default", None) if memu_cfg else None
        embedding_cfg = getattr(memu_cfg, "embedding", None) if memu_cfg else None

        llm_profiles = {
            "default": {
                "provider": _cfg_value(default_cfg, "provider", "DEEPSEEK_PROVIDER", "openai"),
                "base_url": _cfg_value(default_cfg, "base_url", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                "api_key": _cfg_value(default_cfg, "api_key", "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
                "chat_model": _cfg_value(default_cfg, "chat_model", "DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
                "client_backend": _cfg_value(default_cfg, "client_backend", "DEEPSEEK_CLIENT_BACKEND", "sdk"),
            },
            "embedding": {
                "provider": _cfg_value(embedding_cfg, "provider", "SILICONFLOW_PROVIDER", "openai"),
                "base_url": _cfg_value(
                    embedding_cfg,
                    "base_url",
                    "SILICONFLOW_BASE_URL",
                    "https://api.siliconflow.cn/v1",
                ),
                "api_key": _cfg_value(embedding_cfg, "api_key", "SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY"),
                "chat_model": _cfg_value(
                    embedding_cfg,
                    "chat_model",
                    "SILICONFLOW_CHAT_MODEL",
                    _cfg_value(default_cfg, "chat_model", "DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
                ),
                "embed_model": _cfg_value(embedding_cfg, "embed_model", "SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3"),
                "client_backend": _cfg_value(embedding_cfg, "client_backend", "SILICONFLOW_CLIENT_BACKEND", "sdk"),
            },
        }

        memu_dir = ensure_dir(self.workspace / ".memu")
        db_dsn = None
        if memu_cfg and getattr(memu_cfg, "db_dsn", None):
            db_dsn = memu_cfg.db_dsn
        if not db_dsn:
            db_dsn = os.getenv("MEMU_DB_DSN")
        if not db_dsn:
            db_path = (memu_dir / "memu.db").resolve()
            db_dsn = f"sqlite:///{db_path}"
        database_config = {
            "metadata_store": {"provider": "sqlite", "dsn": db_dsn},
        }

        return MemoryService(
            llm_profiles=llm_profiles,
            retrieve_config=retrieve_config,
            memorize_config=memorize_config,
            blob_config=blob_config,
            database_config=database_config,
        )

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
        # Short non-question chit-chat is usually low-value for retrieval and adds latency.
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

        # Semantic intent heuristic:
        # 1) Memory concept + (reference/action/uncertainty)
        # 2) Strong historical reference + action/uncertainty
        # 3) Question-form memory probe with historical reference
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
        if not self.enable_memory:
            return MemoryContext(text="")
        if self.should_skip_retrieve(current_message):
            return MemoryContext(text="")

        user_id = self.build_user_id(channel, chat_id, sender_id)

        history_window = self.retrieve_history_window_full if full_retrieve else self.retrieve_history_window
        queries: list[dict[str, Any]] = []
        for msg in history[-history_window:]:
            role = msg.get("role", "user")
            content = _extract_text(msg.get("content"))
            if not content:
                continue
            queries.append({"role": role, "content": {"text": content}})
        queries.append({"role": "user", "content": {"text": current_message}})

        if not queries:
            return MemoryContext(text="")

        retrieve_cfg = getattr(self._memu, "retrieve_config", None)
        item_cfg = getattr(retrieve_cfg, "item", None) if retrieve_cfg is not None else None
        original_item_top_k = getattr(item_cfg, "top_k", None)
        if full_retrieve and item_cfg is not None:
            try:
                item_cfg.top_k = self.retrieve_top_k_full
            except Exception:
                pass

        try:
            result = await self._memu.retrieve(queries=queries, where={"user_id": user_id})
        except Exception as exc:
            logger.warning(f"MemU retrieve failed: {exc}")
            return MemoryContext(text="")
        finally:
            if full_retrieve and item_cfg is not None and isinstance(original_item_top_k, int):
                try:
                    item_cfg.top_k = original_item_top_k
                except Exception:
                    pass

        return MemoryContext(text=self._format_context(result))

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
        if not self.enable_memory:
            return

        user_id = self.build_user_id(channel, chat_id, sender_id)
        payload = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            "metadata": metadata or {},
            "user_id": user_id,
        }
        timestamp = datetime.now().isoformat().replace(":", "-")
        filename = safe_filename(f"turn_{timestamp}.json")
        path = self.resources_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            start = time.perf_counter()
            await self._memu.memorize(
                resource_url=str(path),
                modality="conversation",
                user={"user_id": user_id},
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
        path, raw = self._read_category_config_raw()
        if raw is None:
            return self._fallback_category_config()
        if isinstance(raw, dict):
            raw = raw.get("memory_categories")
        categories = raw if isinstance(raw, list) else []
        return self._normalize_category_config(categories)

    def add_category_config(self, name: str, description: str = "") -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "Category name is required"}
        path, raw = self._read_category_config_raw()
        if raw is None:
            categories = self._fallback_category_config()
            container: Any = {"memory_categories": categories}
        elif isinstance(raw, dict):
            container = raw
            categories = raw.get("memory_categories") if isinstance(raw.get("memory_categories"), list) else []
        else:
            container = raw
            categories = raw if isinstance(raw, list) else []

        normalized = self._normalize_category_config(categories)
        exists = {c.get("name", "").lower() for c in normalized}
        if name.lower() in exists:
            return {"ok": False, "error": f"Category already exists: {name}"}

        normalized.append({"name": name, "description": (description or "").strip()})
        self._write_category_config(path, container, normalized)
        self._mark_memu_restart_required("memory category added")
        return {"ok": True, "path": str(path)}

    def update_category_config(self, name: str, description: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "Category name is required"}
        if description is None:
            return {"ok": False, "error": "Category description is required"}
        path, raw = self._read_category_config_raw()
        if raw is None:
            categories = self._fallback_category_config()
            container: Any = {"memory_categories": categories}
        elif isinstance(raw, dict):
            container = raw
            categories = raw.get("memory_categories") if isinstance(raw.get("memory_categories"), list) else []
        else:
            container = raw
            categories = raw if isinstance(raw, list) else []

        normalized = self._normalize_category_config(categories)
        updated = False
        for cat in normalized:
            if cat.get("name", "").lower() == name.lower():
                cat["description"] = (description or "").strip()
                updated = True
                break
        if not updated:
            return {"ok": False, "error": f"Category not found: {name}"}

        self._write_category_config(path, container, normalized)
        self._mark_memu_restart_required("memory category updated")
        return {"ok": True, "path": str(path)}

    def delete_category_config(self, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "Category name is required"}
        path, raw = self._read_category_config_raw()
        if raw is None:
            categories = self._fallback_category_config()
            container: Any = {"memory_categories": categories}
        elif isinstance(raw, dict):
            container = raw
            categories = raw.get("memory_categories") if isinstance(raw.get("memory_categories"), list) else []
        else:
            container = raw
            categories = raw if isinstance(raw, list) else []

        normalized = self._normalize_category_config(categories)
        kept = [c for c in normalized if c.get("name", "").lower() != name.lower()]
        if len(kept) == len(normalized):
            return {"ok": False, "error": f"Category not found: {name}"}

        self._write_category_config(path, container, kept)
        self._mark_memu_restart_required("memory category deleted")
        return {"ok": True, "path": str(path)}

    async def memu_status(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        run_checks: bool = True,
        full_checks: bool = False,
    ) -> dict[str, Any]:
        status: dict[str, Any] = {
            "enabled": bool(self.enable_memory and self._memu is not None),
            "llm": {},
            "embedding": {},
            "db": {},
            "health": None,
            "checks": {},
        }
        if not status["enabled"]:
            return status

        memu = self._memu

        # Pull resolved profile config from MemU (env+config already merged there).
        profiles = getattr(memu, "llm_profiles", None)
        profile_map = profiles.profiles if profiles is not None else {}

        def _profile_info(name: str) -> dict[str, Any]:
            cfg = profile_map.get(name)
            if cfg is None:
                return {"configured": False}
            api_missing = False
            if hasattr(memu, "_is_missing_api_key"):
                api_missing = memu._is_missing_api_key(cfg.api_key)
            return {
                "configured": True,
                "provider": cfg.provider,
                "base_url": cfg.base_url,
                "chat_model": cfg.chat_model,
                "embed_model": cfg.embed_model,
                "api_key_set": not api_missing,
            }

        status["llm"] = _profile_info("default")
        status["embedding"] = _profile_info("embedding")

        db_cfg = getattr(memu, "database_config", None)
        if db_cfg is not None:
            status["db"] = {
                "provider": db_cfg.metadata_store.provider,
                "dsn": db_cfg.metadata_store.dsn,
            }

        try:
            status["health"] = await memu.health(
                user={"user_id": self.build_user_id(channel, chat_id, sender_id)},
                include_counts=True,
            )
        except Exception as exc:
            status["health"] = {"ok": False, "error": str(exc)}

        if not run_checks:
            return status

        if full_checks:
            test_user_id = self.build_user_id("memu-status", "fullcheck", "system")
            test_path = None
            try:
                if status["embedding"].get("api_key_set"):
                    try:
                        embed_client = memu._get_llm_client("embedding")
                        vecs = await embed_client.embed(["memu health check"])
                        status["checks"]["embedding"] = {"ok": bool(vecs)}
                    except Exception as exc:
                        status["checks"]["embedding"] = {"ok": False, "error": str(exc)}
                else:
                    status["checks"]["embedding"] = {"ok": False, "skipped": "missing_api_key"}

                if status["llm"].get("api_key_set") and status["embedding"].get("api_key_set"):
                    payload = {
                        "messages": [
                            {"role": "user", "content": "memu full health check"},
                            {"role": "assistant", "content": "ok"},
                        ],
                        "metadata": {"healthcheck": True, "mode": "full"},
                        "user_id": test_user_id,
                    }
                    timestamp = datetime.now().isoformat().replace(":", "-")
                    filename = safe_filename(f"healthcheck_full_{timestamp}.json")
                    test_path = self.resources_dir / filename
                    test_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                    try:
                        await memu.memorize(
                            resource_url=str(test_path),
                            modality="conversation",
                            user={"user_id": test_user_id},
                        )
                        status["checks"]["write"] = {"ok": True, "mode": "full"}
                    except Exception as exc:
                        status["checks"]["write"] = {"ok": False, "error": str(exc), "mode": "full"}

                    try:
                        result = await memu.retrieve(
                            queries=[{"role": "user", "content": {"text": "memu full health check"}}],
                            where={"user_id": test_user_id},
                        )
                        status["checks"]["retrieve"] = {
                            "ok": True,
                            "items": len(result.get("items", [])),
                            "mode": "full",
                        }
                    except Exception as exc:
                        status["checks"]["retrieve"] = {"ok": False, "error": str(exc), "mode": "full"}
                else:
                    status["checks"]["write"] = {"ok": False, "skipped": "missing_api_key", "mode": "full"}
                    status["checks"]["retrieve"] = {"ok": False, "skipped": "missing_api_key", "mode": "full"}
            finally:
                try:
                    await memu.clear_memory(where={"user_id": test_user_id})
                except Exception:
                    pass
                if test_path and test_path.exists():
                    try:
                        test_path.unlink()
                    except Exception:
                        pass
            return status

        test_user_id = self.build_user_id("memu-status", "healthcheck", "system")
        vec: list[float] | None = None
        try:
            # Embedding check
            if status["embedding"].get("api_key_set"):
                try:
                    embed_client = memu._get_llm_client("embedding")
                    vecs = await embed_client.embed(["memu health check"])
                    vec = vecs[0] if vecs else None
                    status["checks"]["embedding"] = {"ok": bool(vec)}
                except Exception as exc:
                    status["checks"]["embedding"] = {"ok": False, "error": str(exc)}
            else:
                status["checks"]["embedding"] = {"ok": False, "skipped": "missing_api_key"}

            # Fast write/read/delete check (direct DB ops)
            store = None
            try:
                store = memu._get_database()
            except Exception:
                store = None

            if vec and store:
                try:
                    resource = store.resource_repo.create_resource(
                        url=f"memu://healthcheck/{test_user_id}",
                        modality="text",
                        local_path="",
                        caption="memu health check",
                        embedding=vec,
                        user_data={"user_id": test_user_id},
                    )
                    store.memory_item_repo.create_item(
                        resource_id=resource.id,
                        memory_type="knowledge",
                        summary="memu health check",
                        embedding=vec,
                        user_data={"user_id": test_user_id},
                    )
                    status["checks"]["write"] = {"ok": True}
                except Exception as exc:
                    status["checks"]["write"] = {"ok": False, "error": str(exc)}

                try:
                    hits = store.memory_item_repo.vector_search_items(
                        query_vec=vec,
                        top_k=1,
                        where={"user_id": test_user_id},
                    )
                    status["checks"]["retrieve"] = {"ok": True, "items": len(hits)}
                except Exception as exc:
                    status["checks"]["retrieve"] = {"ok": False, "error": str(exc)}

                try:
                    store.memory_item_repo.clear_items(where={"user_id": test_user_id})
                    store.resource_repo.clear_resources(where={"user_id": test_user_id})
                    status["checks"]["delete"] = {"ok": True}
                except Exception as exc:
                    status["checks"]["delete"] = {"ok": False, "error": str(exc)}
            else:
                reason = "missing_embedding_or_db"
                status["checks"]["write"] = {"ok": False, "skipped": reason}
                status["checks"]["retrieve"] = {"ok": False, "skipped": reason}
                status["checks"]["delete"] = {"ok": False, "skipped": reason}
        finally:
            try:
                await memu.clear_memory(where={"user_id": test_user_id})
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
        """Query memorized items for the current user scope (debug/admin use)."""
        user_id = self.build_user_id(channel, chat_id, sender_id)
        where: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            where["memory_type"] = memory_type
        if category_id:
            where["category_id"] = category_id
        return await self._memu.query_memory_items(
            where=where,
            limit=limit,
            offset=offset,
            order_by=order_by,
            time_range=time_range,
        )

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
        """Query memory categories for the current user scope (debug/admin use)."""
        user_id = self.build_user_id(channel, chat_id, sender_id)
        where: dict[str, Any] = {"user_id": user_id}
        if category_id:
            where["category_id"] = category_id
        return await self._memu.query_memory_categories(
            where=where,
            limit=limit,
            offset=offset,
            order_by=order_by,
            time_range=time_range,
        )

    async def clear(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
    ) -> dict[str, Any]:
        """Dangerous: clear all MemU memories for this user scope."""
        user_id = self.build_user_id(channel, chat_id, sender_id)
        return await self._memu.clear_memory(where={"user_id": user_id})

    async def health(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        include_counts: bool = False,
    ) -> dict[str, Any]:
        """MemU health check. If channel/chat/sender provided, scopes counts to that user."""
        user: dict[str, Any] | None = None
        if channel or chat_id or sender_id:
            user_id = self.build_user_id(channel, chat_id, sender_id)
            user = {"user_id": user_id}
        return await self._memu.health(user=user, include_counts=include_counts)

    def mark_restart_required(self, *, reason: str | None = None) -> dict[str, Any]:
        return self._memu.mark_restart_required(reason=reason)

    def clear_restart_required(self) -> bool:
        return self._memu.clear_restart_required()

    def is_restart_required(self) -> bool:
        return self._memu.is_restart_required()

    def _read_category_config_raw(self) -> tuple[Path, Any | None]:
        try:
            from memu.app.settings import resolve_memory_category_config_path
        except Exception:
            path = (self.workspace / "config" / "memory_categories.json").resolve()
            if not path.exists():
                return path, None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return path, None
            return path, raw

        path = resolve_memory_category_config_path()
        if not path.exists():
            return path, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return path, None
        return path, raw

    def _fallback_category_config(self) -> list[dict[str, Any]]:
        if self._memu is not None and hasattr(self._memu, "category_configs"):
            try:
                return [cfg.model_dump() for cfg in self._memu.category_configs]
            except Exception:
                pass
        try:
            from memu.app.settings import _default_memory_categories  # type: ignore
        except Exception:
            return []
        try:
            return [cfg.model_dump() for cfg in _default_memory_categories()]
        except Exception:
            return []

    def _normalize_category_config(self, categories: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in categories:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            cat: dict[str, Any] = {"name": name}
            if "description" in item:
                cat["description"] = str(item.get("description", "")).strip()
            if "target_length" in item:
                cat["target_length"] = item.get("target_length")
            if "summary_prompt" in item:
                cat["summary_prompt"] = item.get("summary_prompt")
            normalized.append(cat)
        return normalized

    def _write_category_config(self, path: Path, container: Any, categories: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(container, dict):
            data = dict(container)
            data["memory_categories"] = categories
        else:
            data = categories
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mark_memu_restart_required(self, reason: str) -> None:
        if self._memu is None:
            return
        try:
            self._memu.mark_restart_required(reason=reason)
        except Exception:
            return

    def _format_context(self, result: dict[str, Any]) -> str:
        items = result.get("items") or []
        categories = result.get("categories") or []
        resources = result.get("resources") or []

        if not items and not categories and not resources:
            return ""

        lines: list[str] = ["# Memory (MemU)"]

        if categories:
            lines.append("## Categories")
            for cat in categories[: self.retrieve_top_k]:
                name = cat.get("name", "unknown")
                summary = cat.get("summary") or cat.get("description") or ""
                if summary:
                    lines.append(f"- {name}: {summary}")
                else:
                    lines.append(f"- {name}")

        if items:
            lines.append("## Items")
            for item in items[: self.retrieve_top_k]:
                summary = item.get("summary") or ""
                memory_type = item.get("memory_type") or "unknown"
                if summary:
                    lines.append(f"- [{memory_type}] {summary}")

        if resources:
            lines.append("## Resources")
            for res in resources[: self.retrieve_top_k]:
                url = res.get("url") or res.get("resource_url") or "unknown"
                caption = res.get("caption") or ""
                if caption:
                    lines.append(f"- {url} ({caption})")
                else:
                    lines.append(f"- {url}")

        return "\n".join(lines)
