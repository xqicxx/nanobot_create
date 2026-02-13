"""MemU-backed memory adapter for nanobot."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


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
    ):
        self.workspace = workspace
        self.retrieve_top_k = retrieve_top_k
        self.enable_memory = enable_memory
        self.resources_dir = ensure_dir(resources_dir or (workspace / ".memu" / "resources"))
        self._memu = memu_service or self._init_memu_service()

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

        memu_dir = ensure_dir(self.workspace / ".memu")
        db_dsn = os.getenv("MEMU_DB_DSN")
        if not db_dsn:
            db_path = (memu_dir / "memu.db").resolve()
            db_dsn = f"sqlite:///{db_path}"
        database_config = {
            "metadata_store": {"provider": "sqlite", "dsn": db_dsn},
        }

        return MemoryService(
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

    async def retrieve_context(
        self,
        *,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> MemoryContext:
        if not self.enable_memory:
            return MemoryContext(text="")

        user_id = self.build_user_id(channel, chat_id, sender_id)

        queries: list[dict[str, Any]] = []
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = _extract_text(msg.get("content"))
            if not content:
                continue
            queries.append({"role": role, "content": {"text": content}})
        queries.append({"role": "user", "content": {"text": current_message}})

        if not queries:
            return MemoryContext(text="")

        try:
            result = await self._memu.retrieve(queries=queries, where={"user_id": user_id})
        except Exception as exc:
            logger.warning(f"MemU retrieve failed: {exc}")
            return MemoryContext(text="")

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
            await self._memu.memorize(
                resource_url=str(path),
                modality="conversation",
                user={"user_id": user_id},
            )
        except Exception as exc:
            logger.warning(f"MemU memorize failed: {exc}")

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
