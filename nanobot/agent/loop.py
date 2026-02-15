"""Agent loop: the core processing engine."""

import asyncio
import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_adapter import MemoryAdapter
from nanobot.agent.confirmations import ConfirmationStore
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, ListDirTool, WriteFileTool, EditFileTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.readonly_exec import ReadOnlyExecTool
from nanobot.agent.tools.web import (
    DEFAULT_MINIMAX_MCP_HOST,
    WebSearchTool,
    WebFetchTool,
    UnderstandImageTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.session.manager import SessionManager


class _ThinkTagFilter:
    """Incrementally remove <think>...</think> sections from model output."""

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _OPEN_KEEP = len(_OPEN) - 1
    _CLOSE_KEEP = len(_CLOSE) - 1

    def __init__(self, emit_callback: Callable[[str], None] | None = None) -> None:
        self._emit_callback = emit_callback
        self._buffer = ""
        self._in_think = False
        self._visible_parts: list[str] = []
        self._hidden_parts: list[str] = []

    @property
    def visible_text(self) -> str:
        return "".join(self._visible_parts)

    @property
    def hidden_text(self) -> str:
        return "".join(self._hidden_parts)

    def _emit_visible(self, text: str) -> None:
        if not text:
            return
        self._visible_parts.append(text)
        if self._emit_callback:
            self._emit_callback(text)

    def feed(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        self._drain(force=False)

    def finish(self) -> None:
        self._drain(force=True)

    def _drain(self, *, force: bool) -> None:
        while True:
            lower_buf = self._buffer.lower()
            if not self._in_think:
                open_idx = lower_buf.find(self._OPEN)
                if open_idx < 0:
                    if force:
                        self._emit_visible(self._buffer)
                        self._buffer = ""
                    else:
                        flush_upto = max(0, len(self._buffer) - self._OPEN_KEEP)
                        if flush_upto > 0:
                            self._emit_visible(self._buffer[:flush_upto])
                            self._buffer = self._buffer[flush_upto:]
                    return
                self._emit_visible(self._buffer[:open_idx])
                self._buffer = self._buffer[open_idx + len(self._OPEN):]
                self._in_think = True
                continue

            close_idx = lower_buf.find(self._CLOSE)
            if close_idx < 0:
                if force:
                    if self._buffer:
                        self._hidden_parts.append(self._buffer)
                    self._buffer = ""
                else:
                    flush_upto = max(0, len(self._buffer) - self._CLOSE_KEEP)
                    if flush_upto > 0:
                        self._hidden_parts.append(self._buffer[:flush_upto])
                        self._buffer = self._buffer[flush_upto:]
                return

            if close_idx > 0:
                self._hidden_parts.append(self._buffer[:close_idx])
            self._buffer = self._buffer[close_idx + len(self._CLOSE):]
            self._in_think = False

    @classmethod
    def sanitize_text(cls, text: str) -> tuple[str, str]:
        flt = cls()
        flt.feed(text)
        flt.finish()
        return flt.visible_text, flt.hidden_text


class _StreamBuffer:
    def __init__(
        self,
        *,
        bus: MessageBus,
        channel: str,
        chat_id: str,
        min_chunk: int = 80,
        max_chunk: int = 1200,
        min_interval: float = 0.8,
    ) -> None:
        self._bus = bus
        self._channel = channel
        self._chat_id = chat_id
        self._min_chunk = max(1, min_chunk)
        self._max_chunk = max(self._min_chunk, max_chunk)
        self._min_interval = max(0.0, min_interval)
        self._first_chunk_min = max(12, min(32, self._min_chunk))
        self._first_chunk_max = max(self._first_chunk_min, min(280, self._max_chunk))
        self._buffer: str = ""
        self._sent_any = False
        self._last_flush = 0.0
        self._flush_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def sent_any(self) -> bool:
        return self._sent_any

    def on_token(self, token: str) -> None:
        if not token:
            return
        self._buffer += token
        if len(self._buffer) >= self._max_chunk or any(p in token for p in ("\n", "。", "！", "？", ".", "!", "?")):
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_task and not self._flush_task.done():
            return
        loop = asyncio.get_running_loop()
        self._flush_task = loop.create_task(self._flush(force=False))

    def _collect_break_positions(self, text: str, limit: int) -> tuple[list[int], list[int]]:
        hard: list[int] = []
        soft: list[int] = []
        i = 0
        while i < limit:
            ch = text[i]
            if ch == "\n":
                # Prefer paragraph boundaries first.
                if i + 1 < limit and text[i + 1] == "\n":
                    hard.append(i + 2)
                    i += 1
                else:
                    soft.append(i + 1)
            elif ch in "。！？!?":
                j = i + 1
                while j < limit and text[j] in ' "\'”’）】)]':
                    j += 1
                hard.append(j)
            elif ch == ".":
                prev_ch = text[i - 1] if i > 0 else ""
                next_ch = text[i + 1] if i + 1 < len(text) else ""
                # Keep decimal numbers together: e.g. 3.14
                if prev_ch.isdigit() and next_ch.isdigit():
                    i += 1
                    continue
                if i + 1 >= limit or next_ch.isspace() or next_ch in "\"'”’)]}】）":
                    j = i + 1
                    while j < limit and text[j] in ' "\'”’）】)]':
                        j += 1
                    hard.append(j)
            elif ch in "，,；;：:、":
                soft.append(i + 1)
            elif ch.isspace():
                soft.append(i + 1)
            i += 1
        return hard, soft

    def _short_hard_break_min(self) -> int:
        # Allow sentence-level streaming once min_interval is satisfied.
        return 20 if not self._sent_any else 28

    def _find_split_index(self, *, force: bool, allow_short_hard_break: bool = False) -> int:
        if not self._buffer:
            return 0
        min_required = self._first_chunk_min if not self._sent_any else self._min_chunk
        chunk_limit = self._first_chunk_max if not self._sent_any else self._max_chunk
        limit = len(self._buffer) if force else min(len(self._buffer), chunk_limit)
        if not force and len(self._buffer) < min_required:
            return 0

        hard_breaks, soft_breaks = self._collect_break_positions(self._buffer, limit)
        short_floor = self._short_hard_break_min()
        for pos in reversed(hard_breaks):
            if force or pos >= min_required or (allow_short_hard_break and pos >= short_floor):
                return pos

        # Avoid mid-sentence split unless we hit chunk limit or force flush.
        if not force and len(self._buffer) < chunk_limit:
            return 0

        for pos in reversed(soft_breaks):
            if force or pos >= min_required:
                return pos

        if len(self._buffer) >= chunk_limit:
            return limit

        if force:
            return limit
        return 0

    async def _flush(self, *, force: bool) -> None:
        async with self._lock:
            now = time.monotonic()
            min_required = self._first_chunk_min if not self._sent_any else self._min_chunk
            interval_required = 0.2 if not self._sent_any else self._min_interval
            elapsed = now - self._last_flush
            if (
                not force
                and len(self._buffer) < min_required
                and elapsed < interval_required
            ):
                return

            while self._buffer:
                split_idx = self._find_split_index(
                    force=force,
                    allow_short_hard_break=(not force and elapsed >= interval_required),
                )
                if split_idx <= 0:
                    break
                chunk = self._buffer[:split_idx]
                self._buffer = self._buffer[split_idx:]
                chunk = chunk.strip()
                if not chunk:
                    if not force:
                        break
                    continue
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=self._channel,
                        chat_id=self._chat_id,
                        content=chunk,
                    )
                )
                self._sent_any = True
                self._last_flush = time.monotonic()
                elapsed = 0.0
                if not force:
                    break

    async def finish(self) -> None:
        if self._flush_task and not self._flush_task.done():
            await self._flush_task
        await self._flush(force=True)

class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        minimax_mcp_config: Any | None = None,
        minimax_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        memory_adapter: MemoryAdapter | None = None,
        memu_config: Any | None = None,
        stream_config: Any | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        # Normalize common alias models early (e.g. "minimax" → "minimax/MiniMax-M2.5")
        self.model = self._normalize_model_alias(self.model)
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.minimax_mcp_config = minimax_mcp_config
        self.minimax_api_key = minimax_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.stream_enabled = bool(getattr(stream_config, "enabled", False)) if stream_config is not None else False
        stream_channels = getattr(stream_config, "channels", []) if stream_config is not None else []
        self.stream_channels = {c for c in stream_channels if isinstance(c, str)}
        # Disable bot-side chunk pushing; use provider-native streaming instead.
        self.stream_push_enabled = False
        try:
            self.memu_retrieve_timeout_sec = max(0.0, float(os.getenv("NANOBOT_MEMU_RETRIEVE_TIMEOUT_SEC", "1.2")))
        except Exception:
            self.memu_retrieve_timeout_sec = 1.2
        try:
            self.memu_retrieve_timeout_full_sec = max(
                self.memu_retrieve_timeout_sec,
                float(os.getenv("NANOBOT_MEMU_RETRIEVE_TIMEOUT_SEC_FULL", "3.0")),
            )
        except Exception:
            self.memu_retrieve_timeout_full_sec = max(self.memu_retrieve_timeout_sec, 3.0)
        
        self.context = ContextBuilder(workspace)
        memu_enabled = True
        if memu_config is not None:
            memu_enabled = bool(getattr(memu_config, "enabled", True))
        self.memory_adapter = memory_adapter or MemoryAdapter(
            workspace=workspace,
            enable_memory=memu_enabled,
            memu_config=memu_config,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.confirmations = ConfirmationStore(ttl_seconds=300)
        self._current_session_model: str | None = None
        self._confirm_exec = ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            include_exit_code=True,
            deny_patterns=[],
            allow_high_risk=True,
        )
        
        self._running = False
        self._provider_cache: dict[str, LLMProvider] = {}
        self._register_default_tools()

    @staticmethod
    def _get_previous_user_message(messages: list[dict[str, Any]]) -> str | None:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and not content.startswith("[System:"):
                return content
        return None

    @staticmethod
    def _fire_and_forget(coro: Any, label: str) -> None:
        task = asyncio.create_task(coro)

        def _on_done(t: asyncio.Task) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                # Expected on shutdown or Ctrl+C; no need to warn.
                return
            except Exception as exc:
                logger.warning(f"Background task {label} failed: {exc}")

        task.add_done_callback(_on_done)
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Register file tools
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        if self.cron_service:
            from nanobot.agent.tools.cron import CronTool
            self.tools.register(CronTool(self.cron_service))
        
        # Read-only shell tool (delegates when unsafe)
        self.tools.register(ReadOnlyExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            confirmations=self.confirmations,
        ))
        
        # MiniMax MCP-backed tools
        mcp_cfg = self.minimax_mcp_config
        minimax_mcp_enabled = bool(getattr(mcp_cfg, "enabled", False)) if mcp_cfg is not None else False
        minimax_mcp_key = (getattr(mcp_cfg, "api_key", "") or self.minimax_api_key or "").strip() if mcp_cfg is not None else (self.minimax_api_key or "").strip()
        minimax_mcp_host = getattr(mcp_cfg, "api_host", DEFAULT_MINIMAX_MCP_HOST) if mcp_cfg is not None else DEFAULT_MINIMAX_MCP_HOST
        minimax_mcp_timeout = float(getattr(mcp_cfg, "timeout_seconds", 15)) if mcp_cfg is not None else 15.0
        minimax_web_enabled = bool(getattr(mcp_cfg, "enable_web_search", True)) if mcp_cfg is not None else True
        minimax_image_enabled = bool(getattr(mcp_cfg, "enable_image_understanding", True)) if mcp_cfg is not None else True

        # Web tools
        self.tools.register(WebSearchTool(
            api_key=self.brave_api_key,
            minimax_enabled=minimax_mcp_enabled and minimax_web_enabled,
            minimax_api_key=minimax_mcp_key,
            minimax_api_host=minimax_mcp_host,
            minimax_timeout=minimax_mcp_timeout,
        ))
        self.tools.register(WebFetchTool())
        if minimax_mcp_enabled and minimax_image_enabled and minimax_mcp_key:
            self.tools.register(UnderstandImageTool(
                minimax_api_key=minimax_mcp_key,
                minimax_api_host=minimax_mcp_host,
                timeout=minimax_mcp_timeout,
            ))
        logger.info(
            "MiniMax MCP tools: enabled={} key_set={} web={} image={} host={}",
            minimax_mcp_enabled,
            bool(minimax_mcp_key),
            minimax_web_enabled,
            minimax_image_enabled,
            minimax_mcp_host,
        )

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Memory tools (MemU integration)
        if self.memory_adapter and self.memory_adapter.enable_memory:
            from nanobot.agent.tools.memory_tool import MemoryRetrieveTool, MemorySaveTool
            self.tools.register(MemoryRetrieveTool(self.memory_adapter))
            self.tools.register(MemorySaveTool(self.memory_adapter))
            logger.info("Memory tools registered (retrieve_memory, save_memory)")

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg, stream_callback=None)

                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _extract_confirmation_id(self, text: str) -> str | None:
        match = re.match(r"^\\s*确认\\s+(CONFIRM-[A-Z0-9]{4,})\\s*$", text.strip())
        if match:
            return match.group(1)
        return None

    def _get_session_model(self, session: "Session") -> str:
        model = session.metadata.get("model") or self.model
        return self._normalize_model_alias(model)

        return self._normalize_model_alias(model)

    def _truncate(self, text: str, max_len: int = 200) -> str:
        clean = " ".join(text.split())
        return clean if len(clean) <= max_len else clean[:max_len] + "..."

    def _strip_think_for_output(self, text: str | None, *, channel: str, sender_id: str, stage: str) -> str | None:
        if text is None:
            return None
        visible, hidden = _ThinkTagFilter.sanitize_text(str(text))
        hidden_clean = hidden.strip()
        if hidden_clean:
            logger.info(
                "Hidden <think> content stripped (channel={}, sender={}, stage={}, chars={})",
                channel,
                sender_id,
                stage,
                len(hidden_clean),
            )
            logger.debug(
                "Hidden think preview (channel={}, sender={}, stage={}): {}",
                channel,
                sender_id,
                stage,
                self._truncate(hidden_clean, max_len=1200),
            )
            if not visible.strip():
                return "（模型思考过程已隐藏，未生成可展示答案）"
        return visible

    def _split_name_desc(self, payload: str) -> tuple[str, str]:
        if not payload:
            return "", ""
        for sep in ("|", "：", ":"):
            if sep in payload:
                left, right = payload.split(sep, 1)
                return left.strip(), right.strip()
        return payload.strip(), ""

    def _looks_like_side_effect(self, text: str) -> bool:
        if not text:
            return False
        patterns = [
            r"\\b(write|edit|modify|change|delete|remove|rm|install|exec|build|deploy|start|stop)\\b",
            r"\\b(create|update|patch|replace|append|overwrite)\\b",
            r"\\b(upload|download|clone|pull|push)\\b",
            r"\\b(写|改|修改|编辑|删除|移除|安装|编译|部署|启动|停止|替换|追加|覆盖|创建|更新|上传|下载)\\b",
            r"\\b(configure|configuration|set up|setup)\\b",
            r"\\b(order|purchase|buy|pay|transfer|send)\\b",
            r"(下单|点餐|点外卖|点奶茶|购买|付款|支付|转账|发红包|发送|通知|预约)",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            return True
        if re.search(r"\\.(json|ya?ml|toml|ini|env|cfg|conf)\\b", text, re.IGNORECASE):
            return True
        if re.search(r"(配置|设置|环境变量)", text) and re.search(
            r"(改|写|填|设置|更新|添加|替换|修改)", text
        ):
            return True
        if re.search(r"(api|key|token|密钥|令牌)", text, re.IGNORECASE) and re.search(
            r"(填|写|改|更新|设置|添加|替换)", text
        ):
            return True
        return False

    def _looks_like_readonly(self, text: str) -> bool:
        if not text:
            return False
        patterns = [
            r"\\b(read|list|show|view|print|search|find|grep|rg|cat|ls)\\b",
            r"\\b(read_file|list_dir|web_search|web_fetch)\\b",
            r"\\b(查看|读取|列出|搜索|查询|浏览|检索|统计)\\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

        from nanobot.config.loader import load_config
        from nanobot.providers.registry import PROVIDERS

        config = load_config()
        default_model = config.agents.defaults.model
        known_models = self._sanitize_known_models(session.metadata.get("known_models") or [])
        # Persist cleaned lists to avoid showing stale aliases
        session.metadata["known_models"] = known_models
        merged: list[str] = []
        for candidate in [current_model, default_model, *known_models]:
            if candidate and candidate not in merged:
                merged.append(candidate)

        configured_providers: list[str] = []
        for spec in PROVIDERS:
            provider_cfg = getattr(config.providers, spec.name, None)
            if not provider_cfg:
                continue
            has_key = bool(getattr(provider_cfg, "api_key", None))
            has_base = bool(getattr(provider_cfg, "api_base", None))
            if has_key or has_base:
                configured_providers.append(spec.label)

        lines = [
            f"当前模型：{current_model}",
            f"默认模型：{default_model}",
        ]
        if merged:
            lines.append("已使用模型：" + ", ".join(merged))
        if configured_providers:
            lines.append("已配置提供商：" + ", ".join(configured_providers))
        lines.append("用法：/model <模型名> | /model list | /model reset")
        lines.append("清理历史：/model clean")
        return "\n".join(lines)

    def _format_memu_status(self, status: dict[str, Any], run_checks: bool) -> str:
        lines: list[str] = []
        enabled = status.get("enabled", False)
        lines.append(f"MemU: {'启用' if enabled else '禁用'}")
        if not enabled:
            lines.append("用法：/memu status")
            return "\n".join(lines)

        llm = status.get("llm", {}) or {}
        embedding = status.get("embedding", {}) or {}
        db = status.get("db", {}) or {}
        health = status.get("health") or {}

        def _fmt_key(flag: bool | None) -> str:
            if flag is None:
                return "?"
            return "✓" if flag else "✗"

        if llm:
            lines.append(
                "LLM(default): "
                f"provider={llm.get('provider')}, "
                f"base_url={llm.get('base_url')}, "
                f"api_key={_fmt_key(llm.get('api_key_set'))}, "
                f"chat_model={llm.get('chat_model')}"
            )
        if embedding:
            lines.append(
                "Embedding: "
                f"provider={embedding.get('provider')}, "
                f"base_url={embedding.get('base_url')}, "
                f"api_key={_fmt_key(embedding.get('api_key_set'))}, "
                f"embed_model={embedding.get('embed_model')}"
            )
        if db:
            lines.append(f"DB: {db.get('provider')} {db.get('dsn')}")

        if health:
            ok = health.get("ok")
            lines.append(f"Health: {'ok' if ok else 'error'}")
            counts = health.get("counts")
            if counts:
                lines.append(f"Counts: categories={counts.get('categories')}, items={counts.get('items')}")
            if health.get("restart_required"):
                lines.append("Restart required: yes")
            if health.get("error"):
                lines.append(f"Health error: {health.get('error')}")

        checks = status.get("checks", {}) or {}
        if run_checks and checks:
            def _fmt_check(name: str) -> str:
                chk = checks.get(name, {})
                if chk.get("ok") is True:
                    extra = ""
                    if "items" in chk:
                        extra = f" (items={chk.get('items')})"
                    if chk.get("mode") == "full":
                        extra = f"{extra} [full]"
                    return f"{name}: ok{extra}"
                if chk.get("ok") is False:
                    if chk.get("skipped"):
                        return f"{name}: skipped ({chk.get('skipped')})"
                    if chk.get("mode") == "full":
                        return f"{name}: error ({chk.get('error')}) [full]"
                    return f"{name}: error ({chk.get('error')})"
                return f"{name}: unknown"

            lines.append("Checks:")
            lines.append(_fmt_check("embedding"))
            lines.append(_fmt_check("write"))
            lines.append(_fmt_check("retrieve"))
            if "delete" in checks:
                lines.append(_fmt_check("delete"))
        else:
            lines.append("Checks: skipped (use /memu status to run)")

        return "\n".join(lines)

    def _current_memu_tune(self) -> dict[str, Any]:
        tune = self.memory_adapter.get_retrieve_tuning()
        tune["timeout_sec"] = round(float(self.memu_retrieve_timeout_sec), 3)
        tune["timeout_full_sec"] = round(float(self.memu_retrieve_timeout_full_sec), 3)
        return tune

    def _format_memu_tune(self) -> str:
        tune = self._current_memu_tune()
        lines = [
            "MemU 当前调优参数：",
            f"timeout_sec={tune['timeout_sec']}",
            f"timeout_full_sec={tune['timeout_full_sec']}",
            f"top_k={tune['top_k']}",
            f"top_k_full={tune['top_k_full']}",
            f"history_window={tune['history_window']}",
            f"history_window_full={tune['history_window_full']}",
            "",
            "预设：/memu tune speed | balanced | recall",
            "自定义：/memu tune set timeout=1.2 timeout_full=3 top_k=5 top_k_full=12 history=4 history_full=12",
            "说明：本命令即时生效，仅当前进程有效；重启后会回到 config/env。",
        ]
        return "\n".join(lines)

    def _parse_memu_tune_overrides(self, text: str) -> tuple[dict[str, float | int], str | None]:
        aliases = {
            "timeout": "timeout_sec",
            "timeout_sec": "timeout_sec",
            "timeout_full": "timeout_full_sec",
            "timeout_full_sec": "timeout_full_sec",
            "top_k": "top_k",
            "topk": "top_k",
            "top_k_full": "top_k_full",
            "topk_full": "top_k_full",
            "history": "history_window",
            "history_window": "history_window",
            "history_full": "history_window_full",
            "history_window_full": "history_window_full",
        }
        parsed: dict[str, float | int] = {}
        tokens = [tok.strip() for tok in text.split() if tok.strip()]
        if not tokens:
            return parsed, "未提供参数"
        for token in tokens:
            if "=" not in token:
                return {}, f"参数格式错误：{token}（应为 key=value）"
            key_raw, value_raw = token.split("=", 1)
            key = aliases.get(key_raw.strip().lower())
            if not key:
                return {}, f"不支持的参数：{key_raw}"
            value_text = value_raw.strip()
            if key in {"timeout_sec", "timeout_full_sec"}:
                try:
                    value = float(value_text)
                except Exception:
                    return {}, f"{key_raw} 需要数字"
                if value <= 0:
                    return {}, f"{key_raw} 必须 > 0"
                parsed[key] = value
            else:
                try:
                    value = int(value_text)
                except Exception:
                    return {}, f"{key_raw} 需要正整数"
                if value <= 0:
                    return {}, f"{key_raw} 必须是正整数"
                parsed[key] = value
        return parsed, None

    def _apply_memu_tune(self, overrides: dict[str, float | int]) -> tuple[bool, str]:
        timeout_sec = overrides.get("timeout_sec")
        timeout_full_sec = overrides.get("timeout_full_sec")

        if isinstance(timeout_sec, (int, float)):
            self.memu_retrieve_timeout_sec = max(0.05, float(timeout_sec))
        if isinstance(timeout_full_sec, (int, float)):
            self.memu_retrieve_timeout_full_sec = max(0.05, float(timeout_full_sec))

        self.memu_retrieve_timeout_full_sec = max(
            float(self.memu_retrieve_timeout_sec),
            float(self.memu_retrieve_timeout_full_sec),
        )

        update_result = self.memory_adapter.update_retrieve_tuning(
            top_k=int(overrides["top_k"]) if "top_k" in overrides else None,
            top_k_full=int(overrides["top_k_full"]) if "top_k_full" in overrides else None,
            history_window=int(overrides["history_window"]) if "history_window" in overrides else None,
            history_window_full=int(overrides["history_window_full"]) if "history_window_full" in overrides else None,
        )
        if not update_result.get("ok"):
            return False, str(update_result.get("error") or "unknown error")
        return True, self._format_memu_tune()

    async def _handle_memu_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = (msg.content or "").strip()
        if not raw.startswith("/memu"):
            return None

        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else "status"
        arg_lower = arg.lower()

        if arg_lower in {"help", "?"}:
            content = (
                "用法：/memu status [fast|full]\n"
                "分类：/memu category list | add <名称> [| 描述] | update <名称> | <新描述> | delete <名称>\n"
                "调优：/memu tune [show] | speed | balanced | recall | set key=value ..."
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower.startswith("tune"):
            tune_arg = arg[4:].strip() if len(arg) >= 4 else ""
            tune_lower = tune_arg.lower()
            if not tune_arg or tune_lower in {"show", "status", "list"}:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._format_memu_tune())

            presets: dict[str, dict[str, float | int]] = {
                "speed": {
                    "timeout_sec": 0.8,
                    "timeout_full_sec": 2.0,
                    "top_k": 4,
                    "top_k_full": 10,
                    "history_window": 3,
                    "history_window_full": 8,
                },
                "balanced": {
                    "timeout_sec": 1.2,
                    "timeout_full_sec": 3.0,
                    "top_k": 5,
                    "top_k_full": 12,
                    "history_window": 4,
                    "history_window_full": 12,
                },
                "recall": {
                    "timeout_sec": 1.8,
                    "timeout_full_sec": 4.0,
                    "top_k": 8,
                    "top_k_full": 18,
                    "history_window": 8,
                    "history_window_full": 20,
                },
            }
            if tune_lower in presets:
                ok, result = self._apply_memu_tune(presets[tune_lower])
                if not ok:
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"调优失败：{result}")
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"已应用预设：{tune_lower}\n\n{result}",
                )

            set_arg = tune_arg
            if tune_lower.startswith("set "):
                set_arg = tune_arg[4:].strip()
            elif tune_lower in {"set", "update"}:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="用法：/memu tune set timeout=1.2 timeout_full=3 top_k=5 top_k_full=12 history=4 history_full=12",
                )
            parsed, err = self._parse_memu_tune_overrides(set_arg)
            if err:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"调优参数错误：{err}")
            ok, result = self._apply_memu_tune(parsed)
            if not ok:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"调优失败：{result}")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"已更新调优参数。\n\n{result}")

        if arg_lower.startswith("category"):
            sub_parts = arg.split(None, 2)
            if len(sub_parts) < 2:
                content = "用法：/memu category list | add <名称> [| 描述] | update <名称> | <新描述> | delete <名称>"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

            sub = sub_parts[1].lower()
            payload = sub_parts[2].strip() if len(sub_parts) > 2 else ""

            if sub in {"list", "ls"}:
                categories = self.memory_adapter.list_category_config()
                if not categories:
                    content = "当前没有配置分类。"
                else:
                    lines = ["已配置分类："]
                    for idx, cat in enumerate(categories, start=1):
                        name = cat.get("name", "")
                        desc = cat.get("description", "")
                        lines.append(f"{idx}. {name} - {desc}" if desc else f"{idx}. {name}")
                    content = "\n".join(lines)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

            if sub in {"add", "create"}:
                name, desc = self._split_name_desc(payload)
                result = self.memory_adapter.add_category_config(name, desc)
                if result.get("ok"):
                    content = f"已添加分类：{name}\n已写入：{result.get('path')}\n请重启服务生效。"
                else:
                    content = f"添加失败：{result.get('error')}"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

            if sub in {"update", "edit", "modify"}:
                name, desc = self._split_name_desc(payload)
                result = self.memory_adapter.update_category_config(name, desc)
                if result.get("ok"):
                    content = f"已更新分类：{name}\n已写入：{result.get('path')}\n请重启服务生效。"
                else:
                    content = f"更新失败：{result.get('error')}"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

            if sub in {"delete", "del", "remove"}:
                name = payload.strip()
                result = self.memory_adapter.delete_category_config(name)
                if result.get("ok"):
                    content = f"已删除分类：{name}\n已写入：{result.get('path')}\n请重启服务生效。"
                else:
                    content = f"删除失败：{result.get('error')}"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

            content = "用法：/memu category list | add <名称> [| 描述] | update <名称> | <新描述> | delete <名称>"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        run_checks = True
        if "fast" in arg_lower or "quick" in arg_lower:
            run_checks = False
        full_checks = False
        if "full" in arg_lower:
            run_checks = True
            full_checks = True

        status = await self.memory_adapter.memu_status(
            channel=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            run_checks=run_checks,
            full_checks=full_checks,
        )
        content = self._format_memu_status(status, run_checks)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _handle_model_command(
        self,
        msg: InboundMessage,
        session: "Session",
        current_model: str,
    ) -> OutboundMessage | None:
        raw = msg.content.strip()
        if not raw.startswith("/model"):
            return None

        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        arg_lower = arg.lower()

        if not arg:
            content = (
                f"当前模型：{current_model}\n"
                "用法：/model list | /model <模型名> | /model reset\n"
                "清理历史：/model clean"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"list", "ls"}:
            content = self._list_configured_models()
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"help", "?"}:
            content = (
                "用法：/model list | /model <模型名> | /model reset\n"
                "清理历史：/model clean"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"reset", "default", "restart", "reload"}:
            session.metadata.pop("model", None)
            content = f"已恢复默认模型：{self.model}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"clean", "clear", "purge"}:
            session.metadata.pop("known_models", None)
            content = "已清理模型历史列表。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        # Set model override (per-session)
        resolved, error = self._resolve_model_input(arg, session)
        if error:
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=error)
        model_name = resolved or arg

        session.metadata["model"] = model_name
        known_models = session.metadata.get("known_models") or []
        if model_name not in known_models:
            known_models.append(model_name)
            session.metadata["known_models"] = known_models
        content = f"已切换模型：{model_name}"
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _model_is_configured(self, model: str) -> bool:
        from nanobot.config.loader import load_config
        from nanobot.providers.registry import PROVIDERS, find_by_model, find_by_name

        config = load_config()
        if model.startswith("bedrock/"):
            return True

        # If config can resolve a provider for this model, accept it.
        provider_cfg = config.get_provider(model)
        if provider_cfg and (getattr(provider_cfg, "api_key", None) or getattr(provider_cfg, "api_base", None)):
            return True

        model_lower = model.lower()
        spec = find_by_model(model_lower)

        if spec is None and "/" in model_lower:
            prefix = model_lower.split("/", 1)[0]
            spec = find_by_name(prefix)

        if spec is not None:
            provider_cfg = getattr(config.providers, spec.name, None)
            if not provider_cfg:
                return False
            if getattr(provider_cfg, "api_key", None):
                return True
            if getattr(provider_cfg, "api_base", None):
                return True
            return False

        # No direct provider match. Allow if any gateway/local provider is configured.
        for candidate in PROVIDERS:
            if not (candidate.is_gateway or candidate.is_local):
                continue
            provider_cfg = getattr(config.providers, candidate.name, None)
            if not provider_cfg:
                continue
            if getattr(provider_cfg, "api_key", None) or getattr(provider_cfg, "api_base", None):
                return True
        return False

    def _get_provider_for_model(self, model: str) -> LLMProvider:
        if not model:
            return self.provider
        cached = self._provider_cache.get(model)
        if cached:
            return cached

        from nanobot.config.loader import load_config
        from nanobot.providers.litellm_provider import LiteLLMProvider

        config = load_config()
        p = config.get_provider(model)
        if not p and not model.startswith("bedrock/"):
            return self.provider
        provider = LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=config.get_provider_name(model),
        )
        self._provider_cache[model] = provider
        return provider

    @staticmethod
    def _is_minimax_model(model: str | None) -> bool:
        if not model:
            return False
        lowered = model.strip().lower()
        return lowered.startswith("minimax/") or lowered.startswith("minimax-") or lowered == "minimax"

    @staticmethod
    def _is_stepfun_model(model: str | None) -> bool:
        if not model:
            return False
        lowered = model.strip().lower()
        return lowered.startswith("stepfun/") or lowered.startswith("step-") or lowered == "stepfun" or lowered == "step"

    def _should_stream(self, channel: str | None, stream_callback: Any | None, *, model: str | None = None) -> bool:
        # Use provider-native streaming API for StepFun/MiniMax even if outbound chunk pushing is disabled.
        if self._is_minimax_model(model) or self._is_stepfun_model(model):
            return True
        if not stream_callback:
            return False
        if not self.stream_enabled:
            return False
        if not channel:
            return False
        if not self.stream_channels:
            return False
        return channel in self.stream_channels

    def _should_stream_channel(self, channel: str | None) -> bool:
        if not self.stream_enabled:
            return False
        if not channel:
            return False
        if not self.stream_channels:
            return False
        return channel in self.stream_channels

    async def _send_presence(self, msg: InboundMessage, presence: str) -> None:
        if msg.channel != "whatsapp":
            return
        try:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="",
                metadata={"presence": presence},
            ))
        except Exception as e:
            logger.debug(f"Failed to send presence update: {e}")

    async def _presence_loop(self, msg: InboundMessage) -> None:
        while True:
            await self._send_presence(msg, "composing")
            await asyncio.sleep(6)

    async def _handle_confirmation(self, confirm_id: str, msg: InboundMessage) -> OutboundMessage:
        record = self.confirmations.consume(confirm_id)
        if not record:
            content = f"确认码 {confirm_id} 无效或已过期。请重新发起确认。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        if not record.command.strip():
            content = f"确认码 {confirm_id} 未绑定可执行命令，请重新发起确认。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        result = await self._confirm_exec.execute(
            command=record.command,
            working_dir=record.working_dir,
        )
        content = (
            f"已执行确认命令（{confirm_id}）。\\n"
            f"命令：{record.command}\\n"
            f"结果：\\n{result}"
        )

        # Save to session
        session = self.sessions.get_or_create(msg.session_key)
        session.add_message("user", msg.content)
        session.add_message("assistant", content)
        self.sessions.save(session)

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _normalize_model_alias(self, model: str | None) -> str | None:
        if not model:
            return model
        raw = model.strip()
        lowered = raw.lower()
        alias_defaults = {
            "step": "step-3.5-flash",
            "stepfun": "step-3.5-flash",
            "fast": "step-3.5-flash",
            "flash": "step-3.5-flash",
            "minimax": "minimax/MiniMax-M2.5",
        }
        target = alias_defaults.get(lowered)
        if target and self._model_is_configured(target):
            return target
        return model

    def _sanitize_known_models(self, models: list[str]) -> list[str]:
        from nanobot.providers.registry import PROVIDERS

        provider_names = {spec.name for spec in PROVIDERS}
        cleaned: list[str] = []
        for m in models:
            if not m:
                continue
            normalized = self._normalize_model_alias(m) or m
            lowered = normalized.lower()
            if lowered in {"restart", "reset", "fast", "flash"}:
                continue
            if lowered in provider_names:
                continue
            if not self._model_is_configured(normalized):
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    def _collect_model_candidates(self, session: "Session") -> list[str]:
        from nanobot.config.loader import load_config

        config = load_config()
        default_model = config.agents.defaults.model
        known_models = self._sanitize_known_models(session.metadata.get("known_models") or [])
        current_model = self._get_session_model(session)

        merged: list[str] = []
        for candidate in [
            current_model,
            default_model,
            *known_models,
        ]:
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged

    def _match_provider_alias(self, token: str) -> str | None:
        """Map short provider aliases to registry names."""
        t = token.lower()
        alias_map = {
            "step": "stepfun",
            "stepfun": "stepfun",
            "minimax": "minimax",
            "openai": "openai",
            "anthropic": "anthropic",
            "deepseek": "deepseek",
            "gemini": "gemini",
            "moonshot": "moonshot",
            "dashscope": "dashscope",
            "zhipu": "zhipu",
            "groq": "groq",
            "openrouter": "openrouter",
            "aihubmix": "aihubmix",
            "vllm": "vllm",
        }
        return alias_map.get(t)

    def _model_belongs_to_provider(self, model: str, provider_name: str) -> bool:
        from nanobot.providers.registry import find_by_model, find_by_name

        if not model:
            return False
        spec = find_by_model(model.lower())
        if spec is None and "/" in model:
            spec = find_by_name(model.split("/", 1)[0].lower())
        return bool(spec and spec.name == provider_name)

    def _resolve_model_input(self, model_input: str, session: "Session") -> tuple[str | None, str | None]:
        """Resolve short model inputs. Returns (model, error_message)."""
        raw = (model_input or "").strip()
        if not raw:
            return None, "缺少模型名。"

        # Exact and already configured
        if self._model_is_configured(raw):
            return raw, None

        # Known candidates (session + defaults)
        candidates = self._collect_model_candidates(session)

        # Common shorthand: "fast"/"flash" → a Flash model (prefer known matches).
        if raw.lower() in {"fast", "flash"}:
            flash_matches = [
                m for m in candidates
                if "flash" in m.lower() and self._model_is_configured(m)
            ]
            if len(flash_matches) == 1:
                return flash_matches[0], None
            if len(flash_matches) > 1:
                options = " / ".join(flash_matches[:5])
                return None, f"发现多个可选模型，请选择其一：{options}"
            if self._model_is_configured("step-3.5-flash"):
                return "step-3.5-flash", None

        # Provider alias like "step" or "minimax"
        provider_name = self._match_provider_alias(raw)
        if provider_name:
            provider_candidates = [
                m for m in candidates
                if self._model_belongs_to_provider(m, provider_name) and self._model_is_configured(m)
            ]
            # If a candidate is just the provider name (e.g. "minimax"), drop it.
            provider_candidates = [
                m for m in provider_candidates
                if m.lower() != provider_name
            ]
            # Built-in defaults for common providers
            defaults = {
                "stepfun": ["step-3.5-flash"],
                "minimax": ["minimax/MiniMax-M2.5"],
            }
            for m in defaults.get(provider_name, []):
                if m not in provider_candidates and self._model_is_configured(m):
                    provider_candidates.append(m)
            if len(provider_candidates) == 1:
                return provider_candidates[0], None
            if len(provider_candidates) > 1:
                options = " / ".join(provider_candidates[:5])
                return None, f"发现多个可选模型，请选择其一：{options}"
            return None, f"未找到 {raw} 对应的模型或未配置该提供商。"

        # Fuzzy match against candidates
        matches = [m for m in candidates if raw.lower() in m.lower() and self._model_is_configured(m)]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            options = " / ".join(matches[:5])
            return None, f"发现多个可选模型，请选择其一：{options}"

        return None, f"模型不可用或未配置对应提供商：{raw}"


    def _should_force_web_search(self, msg: InboundMessage) -> bool:
        user_text = msg.content or ""
        if not user_text.strip():
            return False
        if user_text.strip().startswith("/"):
            return False
            return False
        patterns = [
            r"\\bsearch\\b",
            r"\\bgoogle\\b",
            r"\\bbing\\b",
            r"搜索",
            r"搜",
            r"搜一下",
            r"查询",
            r"查一下",
            r"查找",
        ]
        return any(re.search(p, user_text, re.IGNORECASE) for p in patterns)

    def _is_weather_intent(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"(天气|气温|温度|weather|forecast)", text, re.IGNORECASE))

    def _extract_weather_location(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.strip()
        m = re.search(r"weather\\s+in\\s+(.+)", lowered, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        cleaned = re.sub(r"(天气|气温|温度|查询|看看|一下|现在|今天|的|吧)", "", lowered)
        cleaned = cleaned.strip(" ,，。?？!！")
        return cleaned.strip()

    def _is_supported_image_source(self, source: str, media_type: str | None) -> bool:
        if not source:
            return False
        if media_type and media_type not in {"image", "photo", "sticker"}:
            return False
        lowered = source.lower()
        if lowered.startswith(("http://", "https://", "data:image/")):
            return True
        path = Path(source).expanduser()
        if path.exists() and path.is_file():
            return True
        return Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    def _collect_image_sources(self, msg: InboundMessage) -> list[str]:
        media_type = str((msg.metadata or {}).get("media_type", "")).lower() or None
        images: list[str] = []
        for src in msg.media or []:
            if isinstance(src, str) and self._is_supported_image_source(src, media_type):
                images.append(src)
        if not images:
            text = msg.content or ""
            for pattern in (
                r"\[(?:Image|图片)\s*:\s*([^\]\n]+)\]",
                r"\[(?:Image|图片)\]\s*([/A-Za-z0-9_.\-~]+)",
            ):
                for match in re.findall(pattern, text, flags=re.IGNORECASE):
                    candidate = str(match).strip()
                    if self._is_supported_image_source(candidate, "image"):
                        images.append(candidate)
                if images:
                    break
        return images[:2]

    def _build_image_analysis_prompt(self, user_text: str) -> str:
        text = (user_text or "").strip()
        if text:
            return f"请根据图片回答用户问题：{text}"
        return "请描述图片主要内容，提取关键实体和可见文字。"

    async def _prepare_multimodal_content(
        self,
        msg: InboundMessage,
    ) -> tuple[str, list[str] | None]:
        images = self._collect_image_sources(msg)
        if not images:
            return msg.content, msg.media if msg.media else None

        tool = self.tools.get("understand_image")
        if not isinstance(tool, UnderstandImageTool) or not tool.is_configured():
            logger.warning(
                "Image received but understand_image tool unavailable (channel={}, sender={})",
                msg.channel,
                msg.sender_id,
            )
            return msg.content, msg.media if msg.media else None

        prompt = self._build_image_analysis_prompt(msg.content or "")
        analyses: list[str] = []
        for idx, image_src in enumerate(images, start=1):
            vision_start = time.perf_counter()
            result = await self.tools.execute(
                "understand_image",
                {"prompt": prompt, "image_source": image_src},
            )
            vision_ms = int(round((time.perf_counter() - vision_start) * 1000))
            if isinstance(result, str) and result.startswith("Error:"):
                logger.warning(
                    "understand_image failed in {}ms (channel={}, sender={}, source={}): {}",
                    vision_ms,
                    msg.channel,
                    msg.sender_id,
                    image_src,
                    result,
                )
                continue
            logger.info(
                "understand_image completed in {}ms (channel={}, sender={}, source={})",
                vision_ms,
                msg.channel,
                msg.sender_id,
                image_src,
            )
            analyses.append(f"[Image {idx}]\n{result}")

        if not analyses:
            return msg.content, msg.media if msg.media else None

        base_text = (msg.content or "").strip()
        merged_analysis = "\n\n".join(analyses)
        if base_text:
            enriched = f"{base_text}\n\n[图像解析结果]\n{merged_analysis}"
        else:
            enriched = f"[图像解析结果]\n{merged_analysis}"

        logger.info(
            "Prepared multimodal context from {} image(s) (channel={}, sender={})",
            len(analyses),
            msg.channel,
            msg.sender_id,
        )
        # Once we have textual analysis, avoid passing raw image blobs to text-only models.
        return enriched, None

    async def _handle_weather(self, msg: InboundMessage) -> OutboundMessage | None:
        if not self._is_weather_intent(msg.content or ""):
            return None
            return None
        location = self._extract_weather_location(msg.content or "")
        if not location:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="请告诉我具体地点，例如：北京天气。",
            )
        url = f"https://wttr.in/{quote_plus(location)}?format=3"
        result = await self.tools.execute(
            "web_fetch",
            {"url": url, "extractMode": "text", "maxChars": 500},
        )
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                text = (data.get("text") or "").strip()
                if text:
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=text)
        except Exception:
            pass
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    def _handle_version_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = (msg.content or "").strip()
        if raw not in {"/version", "/ver"}:
            return None
        try:
            import nanobot
            package_path = Path(nanobot.__file__).resolve()
        except Exception:
            package_path = None
        commit = "unknown"
        if package_path:
            for parent in package_path.parents:
                git_dir = parent / ".git"
                if git_dir.is_dir():
                    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
                    if head.startswith("ref:"):
                        ref = head.split(":", 1)[1].strip()
                        ref_path = git_dir / ref
                        if ref_path.exists():
                            commit = ref_path.read_text(encoding="utf-8").strip()[:8]
                    else:
                        commit = head[:8]
                    break
        content = "nanobot 版本信息："
        if package_path:
            content += f"\n路径：{package_path}"
        content += f"\nGit：{commit}"
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _system_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/system list", "系统命令总览"),
        ]

    def _menu_base_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/menu list", "菜单命令总览"),
            ("/menu all", "全量命令（含分类）"),
            ("/menu categories", "显示所有分类与描述"),
            ("/menu restart now", "重启 agent 进程"),
            ("/menu version", "版本信息（路由到 /version）"),
        ]

    @staticmethod
    def _rewrite_command_prefix(command: str, source: str, target: str) -> str | None:
        if command == source:
            return target
        if command.startswith(source + " "):
            return target + command[len(source) :]
        return None

    @staticmethod
    def _dedupe_command_specs(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for command, desc in specs:
            key = command.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append((command, desc))
        return unique

    def _menu_routed_command_specs(self) -> list[tuple[str, str]]:
        specs: list[tuple[str, str]] = []

        for command, desc in self._model_command_specs():
            mapped = self._rewrite_command_prefix(command, "/model", "/menu model")
            if mapped:
                specs.append((mapped, f"{desc}（路由）"))

        memu_routes = (
            ("/memu status", "/menu status"),
            ("/memu tune", "/menu tune"),
            ("/memu category", "/menu category"),
        )
        for command, desc in self._memu_command_specs():
            for source, target in memu_routes:
                mapped = self._rewrite_command_prefix(command, source, target)
                if mapped:
                    specs.append((mapped, f"{desc}（路由）"))
                    break

            if mapped:
                specs.append((mapped, f"{desc}（路由）"))

        return self._dedupe_command_specs(specs)

    def _menu_command_specs(self) -> list[tuple[str, str]]:
        return self._dedupe_command_specs(
            [*self._menu_base_command_specs(), *self._menu_routed_command_specs()]
        )

        return [
        ]

    def _model_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/model list", "查看可用模型与当前模型"),
            ("/model <模型名>", "切换当前会话模型"),
            ("/model reset", "恢复默认模型"),
            ("/model clean", "清理模型历史"),
        ]

    def _memu_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/memu status [fast|full]", "MemU 健康与读写检查"),
            ("/memu tune [show|speed|balanced|recall|set ...]", "MemU 运行时调优"),
            ("/memu category list", "查看记忆分类"),
            ("/memu category add <名称> [| 描述]", "新增分类"),
            ("/memu category update <名称> | <新描述>", "更新分类"),
            ("/memu category delete <名称>", "删除分类"),
        ]

    def _misc_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/version", "版本信息"),
        ]

    def _command_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        return [
            ("系统命令", self._system_command_specs()),
            ("菜单命令", self._menu_command_specs()),
            ("模型命令", self._model_command_specs()),
            ("MemU 命令", self._memu_command_specs()),
            ("其他命令", self._misc_command_specs()),
        ]

    @staticmethod
    def _format_command_lines(specs: list[tuple[str, str]]) -> list[str]:
        lines: list[str] = []
        for cmd, desc in specs:
            line = f"{cmd} {desc}".strip()
            lines.append(line)
        return lines


    def _format_menu_list(self) -> str:
        lines = ["菜单命令："]
        lines.extend(self._format_command_lines(self._menu_command_specs()))
        return "\n".join(lines)

    def _format_system_list(self) -> str:
        lines: list[str] = []
        for idx, (title, specs) in enumerate(self._command_sections()):
            if idx:
                lines.append("")
            lines.append(f"{title}：")
            lines.extend(self._format_command_lines(specs))
        return "\n".join(lines)

    async def _handle_system_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = (msg.content or "").strip()
        if not raw.startswith("/system"):
            return None
        parts = raw.split(None, 1)
        arg = parts[1].strip().lower() if len(parts) > 1 else "list"
        if arg in {"help", "?", "ls", "list"}:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_system_list(),
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="用法：/system list",
        )

    async def _handle_compact_command(self, msg: InboundMessage) -> OutboundMessage | None:
        """Handle /compact command to compress and save conversation context."""
        raw = (msg.content or "").strip()
        if not raw.startswith("/compact"):
            return None
        
        # Get current session to compress context
        session = self.sessions.get_or_create(msg.session_key)
        history = session.get_history()
        
        if len(history) <= 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="📭 当前对话历史太少，无需压缩。",
            )
        
        compressed_summary = ""
        try:
            # Build conversation text for compression
            conversation_text = []
            for msg_item in history:
                role = msg_item.get("role", "unknown")
                content = msg_item.get("content", "")
                if isinstance(content, str) and content.strip():
                    conversation_text.append(f"{role}: {content[:200]}")  # Limit each message
            
            if not conversation_text:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="📭 没有找到可压缩的对话内容。",
                )
            
            if not self.memory_adapter or not self.memory_adapter.enable_memory:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="❌ MemU 记忆功能未启用，无法保存压缩内容。",
                )
            
            # Use LLM to generate summary
            summary_prompt = (
                "请总结以下对话的关键信息（用户身份、重要事实、待办事项等），"
                "用3-5个要点简要概括：\n\n" + 
                "\n".join(conversation_text[-20:])  # Last 20 messages
            )
            
            # Get current model
            session_model = self._get_session_model(session)
            provider = self._get_provider_for_model(session_model)
            
            response = await provider.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                model=session_model,
                stream=False,
            )
            compressed_summary = (response.content or "").strip()
            
            if compressed_summary:
                # Save compressed summary to memory
                await self.memory_adapter.memorize_turn(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    sender_id=msg.sender_id,
                    user_message="[对话总结] " + compressed_summary[:500],
                    assistant_message="已保存对话摘要到长期记忆",
                    metadata={
                        "session_key": msg.session_key,
                        "compressed": True,
                        "message_count": len(history),
                    },
                )
                
                content = f"📦 已压缩并保存 {len(history)} 条历史记录到长期记忆：\n\n{compressed_summary[:400]}"
                if len(compressed_summary) > 400:
                    content += "..."
                
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                )
            else:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="⚠️ 压缩失败，无法生成摘要。",
                )
                
        except Exception as e:
            logger.warning(f"Context compression failed: {e}")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"❌ 压缩失败: {str(e)}",
            )

    async def _handle_new_command(self, msg: InboundMessage) -> OutboundMessage | None:
        """Handle /new command to archive current conversation and start fresh.
        
        This will:
        1. Archive current conversation to long-term memory (MemU)
        2. Clear current session history
        3. Start a new conversation context
        
        The archived conversation can still be retrieved via memory search.
        """
        raw = (msg.content or "").strip()
        if not raw.startswith("/new"):
            return None
        
        # Get current session
        session = self.sessions.get_or_create(msg.session_key)
        history = session.get_history()
        archived_count = len(history) // 2  # Rough count of user-assistant pairs
        
        # Archive to MemU if there's content to save
        if len(history) > 2 and self.memory_adapter and self.memory_adapter.enable_memory:
            try:
                # Build conversation text
                conversation_text = []
                for msg_item in history:
                    role = msg_item.get("role", "unknown")
                    content = msg_item.get("content", "")
                    if isinstance(content, str) and content.strip():
                        conversation_text.append(f"{role}: {content[:300]}")
                
                # Archive to memory
                await self.memory_adapter.memorize_turn(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    sender_id=msg.sender_id,
                    user_message="[对话归档] " + "\n".join(conversation_text[-30:]),  # Last 30 messages
                    assistant_message=f"已归档 {archived_count} 轮对话到长期记忆",
                    metadata={
                        "session_key": msg.session_key,
                        "archived": True,
                        "message_count": len(history),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to archive conversation: {e}")
        
        # Clear current session
        deleted = self.sessions.delete(msg.session_key)
        
        content = "🆕 已开启新对话！"
        if archived_count > 0:
            content += f"\n📦 已归档 {archived_count} 轮对话到长期记忆"
            content += "\n💡 之前的内容可通过记忆搜索找回"
        else:
            content += "\n（当前没有需要归档的对话）"
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
        )

    async def _handle_menu_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = (msg.content or "").strip()
        if not raw.startswith("/menu"):
            return None
        parts = raw.split(None, 1)
        arg_raw = parts[1].strip() if len(parts) > 1 else "list"
        arg_lower = arg_raw.lower()
        if arg_lower.startswith("restart"):
            restart_parts = arg_raw.split(None, 1)
            restart_flag = restart_parts[1].strip().lower() if len(restart_parts) > 1 else ""
            if restart_flag not in {"now", "confirm", "yes", "ok", "确认"}:
                content = (
                    "将重启当前 nanobot 进程（适用于 systemd 托管）。\n"
                    "用法：/menu restart now"
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            if not self._is_systemd_managed():
                content = (
                    "当前进程未检测到 systemd 托管，已取消自动重启。\n"
                    "请手动执行：sudo systemctl restart nanobot-agent@root"
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            self._schedule_self_restart(delay=1.2)
            content = "收到，正在重启 nanobot 进程（约 3-8 秒恢复）。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        if arg_lower in {"categories", "cats", "cat"}:
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu category list",
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg_lower in {"all", "full"}:
            categories = self.memory_adapter.list_category_config()
            lines = ["全部命令（动态）：", self._format_system_list()]
            if categories:
                lines.append("")
                lines.append("当前分类：")
                for idx, cat in enumerate(categories, start=1):
                    name = cat.get("name", "")
                    desc = cat.get("description", "")
                    lines.append(f"{idx}. {name} - {desc}" if desc else f"{idx}. {name}")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))
        if arg_lower.startswith("model "):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/model " + arg_raw[6:].strip(),
                media=msg.media,
                metadata=msg.metadata,
            )
            session = self.sessions.get_or_create(msg.session_key)
            current_model = self._get_session_model(session)
            return self._handle_model_command(forwarded, session, current_model)
        if arg_lower.startswith("memu "):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg_raw[5:].strip(),
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg_lower.startswith("status"):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg_raw,
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg_lower.startswith("category"):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg_raw,
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg_lower.startswith("tune"):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg_raw,
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg_lower in {"version", "ver"}:
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/version",
                media=msg.media,
                metadata=msg.metadata,
            )
            return self._handle_version_command(forwarded)
        if arg_lower in {"list", "ls"}:
            # /menu list - 显示记忆内容
            items = await self.memory_adapter.query_items(
                channel=msg.channel,
                chat_id=msg.chat_id,
                sender_id=msg.sender_id,
                limit=20,
            )
            if not items:
                content = "暂无记忆内容。用 /menu help 查看命令。"
            else:
                lines = ["记忆内容："]
                for idx, item in enumerate(items, start=1):
                    content_text = item.get("content", "")[:200]
                    memory_type = item.get("memory_type", "unknown")
                    lines.append(f"{idx}. [{memory_type}] {content_text}")
                content = "\n".join(lines)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        if arg_lower in {"help", "?"}:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_menu_list(),
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="用法：/menu list | /menu all | /system list",
        )

    def _is_systemd_managed(self) -> bool:
        # systemd usually injects one of these env vars for managed services.
        return bool(
            os.getenv("INVOCATION_ID")
            or os.getenv("JOURNAL_STREAM")
            or os.getenv("SYSTEMD_EXEC_PID")
        )

    def _schedule_self_restart(self, *, delay: float = 1.0) -> None:
        def _restart_now() -> None:
            logger.warning("Restart requested via /menu restart; exiting with code 1 for supervisor restart.")
            os._exit(1)

        asyncio.get_running_loop().call_later(delay, _restart_now)
    
    async def _process_message(
        self,
        msg: InboundMessage,
        stream_callback: Any | None = None,
    ) -> OutboundMessage | None:
        start = time.perf_counter()
        ingress_wait_ms: int | None = None
        raw_ingress = (msg.metadata or {}).get("_nb_ingress_perf")
        if isinstance(raw_ingress, (int, float)):
            ingress_wait_ms = int(round((start - float(raw_ingress)) * 1000))
        response = await self._process_message_impl(msg, stream_callback=stream_callback)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            content_len = len(msg.content or "")
        except Exception:
            content_len = -1
        logger.info(
            "Message processed in {}ms (channel={}, sender={}, len={}, response={}, ingress_wait={}ms)",
            int(round(elapsed_ms)),
            msg.channel,
            msg.sender_id,
            content_len,
            "yes" if response else "no",
            ingress_wait_ms if ingress_wait_ms is not None else -1,
        )
        return response

    async def _process_message_impl(
        self,
        msg: InboundMessage,
        stream_callback: Any | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        impl_start = time.perf_counter()
        llm_total_ms = 0
        tool_total_ms = 0
        tool_calls = 0

        # Handle system messages
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        # Handle confirmation replies before LLM
        stripped = (msg.content or "").strip().lower()
        if stripped in {"确认", "confirm", "ok", "yes", "是"}:
            pending = self.confirmations.list_pending_ids()
            if len(pending) == 1:
                return await self._handle_confirmation(pending[0], msg)
            if len(pending) > 1:
                content = (
                    "有多个待确认命令，请回复：确认 CONFIRM-XXXX\n"
                    "可用ID: " + ", ".join(pending)
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            content = "当前没有待确认命令。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        confirm_id = self._extract_confirmation_id(msg.content)
        if confirm_id:
            return await self._handle_confirmation(confirm_id, msg)

        # Reset per-turn spawn tracking early

        # Handle /system command before LLM
        system_response = await self._handle_system_command(msg)
        if system_response:
            session = self.sessions.get_or_create(msg.session_key)
            session.add_message("user", msg.content)
            session.add_message("assistant", system_response.content)
            self.sessions.save(session)
            return system_response

        # Handle /new command before LLM
        new_response = await self._handle_new_command(msg)
        if new_response:
            return new_response
        
        # Handle /compact command before LLM
        compact_response = await self._handle_compact_command(msg)
        if compact_response:
            return compact_response

        # Handle /version command before LLM
        menu_response = await self._handle_menu_command(msg)
        if menu_response:
            session = self.sessions.get_or_create(msg.session_key)
            session.add_message("user", msg.content)
            session.add_message("assistant", menu_response.content)
            self.sessions.save(session)
            return menu_response

        # Handle /version command before LLM
        version_response = self._handle_version_command(msg)
        if version_response:
            session = self.sessions.get_or_create(msg.session_key)
            session.add_message("user", msg.content)
            session.add_message("assistant", version_response.content)
            self.sessions.save(session)
            return version_response

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        session_model = self._get_session_model(session)
        self._current_session_model = session_model

        # Handle /model commands before LLM
        memu_response = await self._handle_memu_command(msg)
        if memu_response:
            session.add_message("user", msg.content)
            session.add_message("assistant", memu_response.content)
            self.sessions.save(session)
            self._current_session_model = None
            return memu_response

        # Handle /model commands before LLM
        model_response = self._handle_model_command(msg, session, session_model)
        if model_response:
            # Save to session
            session.add_message("user", msg.content)
            session.add_message("assistant", model_response.content)
            self.sessions.save(session)
            self._current_session_model = None
            return model_response
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        
        exec_tool = self.tools.get("exec")
        if isinstance(exec_tool, ReadOnlyExecTool):
            exec_tool.set_context(msg.channel, msg.chat_id)


        
        # Build initial messages (use get_history for LLM-formatted messages)
        memory_context = ""
        memu_retrieve_ms: int | None = None
        if msg.channel != "system":
            try:
                memu_start = time.perf_counter()
                full_retrieve = self.memory_adapter.should_force_full_retrieve(msg.content or "")
                if full_retrieve:
                    retrieve_timeout_sec = max(
                        float(self.memu_retrieve_timeout_sec),
                        float(self.memu_retrieve_timeout_full_sec),
                    )
                else:
                    retrieve_timeout_sec = float(self.memu_retrieve_timeout_sec)
                retrieve_coro = self.memory_adapter.retrieve_context(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    sender_id=msg.sender_id,
                    history=session.get_history(),
                    current_message=msg.content,
                    full_retrieve=full_retrieve,
                )
                if retrieve_timeout_sec > 0:
                    mem_ctx = await asyncio.wait_for(retrieve_coro, timeout=retrieve_timeout_sec)
                else:
                    mem_ctx = await retrieve_coro
                memory_context = mem_ctx.text
                memu_retrieve_ms = int(round((time.perf_counter() - memu_start) * 1000))
                logger.info(
                    "MemU retrieve in {}ms (channel={}, sender={}, history_len={}, msg_len={}, full_retrieve={}, timeout={}ms)",
                    memu_retrieve_ms,
                    msg.channel,
                    msg.sender_id,
                    len(session.get_history()),
                    len(msg.content or ""),
                    full_retrieve,
                    int(round(retrieve_timeout_sec * 1000)),
                )
            except asyncio.TimeoutError:
                memu_retrieve_ms = int(round((time.perf_counter() - memu_start) * 1000))
                logger.warning(
                    "MemU retrieve timed out in {}ms (timeout={}ms, channel={}, sender={}, full_retrieve={})",
                    memu_retrieve_ms,
                    int(round(retrieve_timeout_sec * 1000)),
                    msg.channel,
                    msg.sender_id,
                    full_retrieve,
                )
            except Exception as exc:
                logger.warning(f"MemU context fetch failed: {exc}")

        effective_content = msg.content
        effective_media = msg.media if msg.media else None
        if msg.channel != "system":
            try:
                effective_content, effective_media = await self._prepare_multimodal_content(msg)
            except Exception as exc:
                logger.warning(f"Multimodal pre-analysis failed: {exc}")

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=effective_content,
            media=effective_media,
            channel=msg.channel,
            chat_id=msg.chat_id,
            memory_context=memory_context,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        typing_task: asyncio.Task[None] | None = None

        if msg.channel == "whatsapp":
            typing_task = asyncio.create_task(self._presence_loop(msg))
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                
                # Call LLM
                llm_start = time.perf_counter()
                use_stream = self._should_stream(msg.channel, stream_callback, model=session_model)
                if use_stream and stream_callback is None and (
                    self._is_minimax_model(session_model) or self._is_stepfun_model(session_model)
                ):
                    provider_name = "MiniMax" if self._is_minimax_model(session_model) else "StepFun"
                    logger.info(
                        "{} native streaming enabled (provider-side, outbound chunk streaming disabled)",
                        provider_name,
                    )
                response = await self._get_provider_for_model(session_model).chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=session_model,
                    stream=use_stream,
                    on_token=stream_callback if use_stream else None,
                )
                llm_elapsed_ms = int(round((time.perf_counter() - llm_start) * 1000))
                llm_total_ms += llm_elapsed_ms
                logger.info(
                    "LLM response in {}ms (channel={}, sender={}, model={}, iter={})",
                    llm_elapsed_ms,
                    msg.channel,
                    msg.sender_id,
                    session_model,
                    iteration,
                )
                
                # Handle tool calls
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)  # Must be JSON string
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts,
                        reasoning_content=response.reasoning_content,
                    )
                    
                    # Execute tools
                    for tool_call in response.tool_calls:
                        tool_calls += 1
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                        tool_start = time.perf_counter()
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_elapsed_ms = int(round((time.perf_counter() - tool_start) * 1000))
                        tool_total_ms += tool_elapsed_ms
                        logger.info(
                            "Tool {} done in {}ms (channel={}, sender={})",
                            tool_call.name,
                            tool_elapsed_ms,
                            msg.channel,
                            msg.sender_id,
                        )
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )

                    # Continue loop after tool execution
                    continue

                # No tool calls
                if self._should_force_web_search(msg):
                    search_tool = self.tools.get("web_search")
                    if isinstance(search_tool, WebSearchTool) and not search_tool.is_configured():
                        final_content = (
                            "搜索未配置：请设置 BRAVE_API_KEY / config.tools.web.search.apiKey，"
                            "或配置 MiniMax MCP（tools.mcp.minimax.apiKey / providers.minimax.apiKey）。"
                        )
                        break
                    result = await self.tools.execute(
                        "web_search",
                        {"query": msg.content, "count": 5},
                    )
                    final_content = result
                    break

                final_content = response.content
                break
        finally:
            if typing_task:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
                await self._send_presence(msg, "paused")
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        final_content = self._strip_think_for_output(
            final_content,
            channel=msg.channel,
            sender_id=msg.sender_id,
            stage="main_response",
        ) or ""

        if (
            "## 我懂你的意思" in final_content
            or "## 我打算怎么做" in final_content
            or "## 我先去干活" in final_content
        ):
            match = re.search(r"你想做的是[:：]\\s*(.*)", final_content)
            summary = match.group(1).strip() if match else (msg.content or "").strip()
            if not summary:
                summary = "已收到你的需求"
            final_content = f"收到：{summary}"

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        impl_elapsed_ms = int(round((time.perf_counter() - impl_start) * 1000))
        logger.info(
            "Stage summary: total={}ms, memu_retrieve={}ms, llm_total={}ms, tool_total={}ms, iterations={}, tool_calls={}",
            impl_elapsed_ms,
            memu_retrieve_ms if memu_retrieve_ms is not None else -1,
            llm_total_ms,
            tool_total_ms,
            iteration,
            tool_calls,
        )
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        self._current_session_model = None

        outbound = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=dict(msg.metadata or {}),  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
        outbound.metadata["_nb_response_ready_perf"] = time.perf_counter()

        # Persist memory via MemU (skip system messages and low-signal content)
        if msg.channel != "system":
            previous_user = self._get_previous_user_message(session.messages[:-2])
            if not self.memory_adapter.should_skip_write(msg.content, previous_user):
                self._fire_and_forget(
                    self.memory_adapter.memorize_turn(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        sender_id=msg.sender_id,
                        user_message=msg.content,
                        assistant_message=final_content,
                        metadata={
                            "session_key": msg.session_key,
                            "channel": msg.channel,
                            "chat_id": msg.chat_id,
                            "sender_id": msg.sender_id,
                        },
                    ),
                    "memorize_turn",
                )

        return outbound
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """Process a system message."""
        logger.info(f"Processing system message from {msg.sender_id}")
        return None

        # Build messages with the announce content
        memory_context = ""
        if msg.channel != "system":
            try:
                memu_start = time.perf_counter()
                memory_context = (
                    await self.memory_adapter.retrieve_context(
                        channel=origin_channel,
                        chat_id=origin_chat_id,
                        sender_id=msg.sender_id,
                        history=session.get_history(),
                        current_message=msg.content,
                    )
                ).text
                memu_elapsed_ms = int(round((time.perf_counter() - memu_start) * 1000))
                logger.info(
                    "MemU retrieve in {}ms (channel={}, sender={}, history_len={}, msg_len={})",
                    memu_elapsed_ms,
                    origin_channel,
                    msg.sender_id,
                    len(session.get_history()),
                    len(msg.content or ""),
                )
            except Exception as exc:
                logger.warning(f"MemU context fetch failed: {exc}")

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
            memory_context=memory_context,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            llm_start = time.perf_counter()
            response = await self._get_provider_for_model(session_model).chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=session_model,
            )
            llm_elapsed_ms = int(round((time.perf_counter() - llm_start) * 1000))
            logger.info(
                "LLM response in {}ms (channel={}, sender={}, model={}, iter={})",
                llm_elapsed_ms,
                origin_channel,
                msg.sender_id,
                session_model,
                iteration,
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        final_content = self._strip_think_for_output(
            final_content,
            channel=origin_channel,
            sender_id=msg.sender_id,
            stage="system_response",
        ) or ""
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        self._current_session_model = None
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        stream_callback: Any | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        safe_stream_callback = stream_callback
        think_filter: _ThinkTagFilter | None = None
        if stream_callback is not None:
            think_filter = _ThinkTagFilter(emit_callback=stream_callback)
            safe_stream_callback = think_filter.feed

        response = await self._process_message(msg, stream_callback=safe_stream_callback)
        if think_filter is not None:
            think_filter.finish()
        return response.content if response else ""
