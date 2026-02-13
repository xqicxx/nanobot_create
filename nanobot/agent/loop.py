"""Agent loop: the core processing engine."""

import asyncio
import json
import time
import re
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
from nanobot.agent.subtask_output import parse_subtask_output
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, ListDirTool, WriteFileTool, EditFileTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.readonly_exec import ReadOnlyExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.delegate import DelegateTool
from nanobot.agent.subagent import SubagentManager, SpawnResult
from nanobot.session.manager import SessionManager


class _StreamBuffer:
    def __init__(
        self,
        *,
        bus: MessageBus,
        channel: str,
        chat_id: str,
        min_chunk: int = 60,
        max_chunk: int = 320,
        min_interval: float = 0.6,
    ) -> None:
        self._bus = bus
        self._channel = channel
        self._chat_id = chat_id
        self._min_chunk = max(1, min_chunk)
        self._max_chunk = max(self._min_chunk, max_chunk)
        self._min_interval = max(0.0, min_interval)
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
        if len(self._buffer) >= self._max_chunk or ("\n" in token and len(self._buffer) >= self._min_chunk):
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_task and not self._flush_task.done():
            return
        loop = asyncio.get_running_loop()
        self._flush_task = loop.create_task(self._flush(force=False))

    async def _flush(self, *, force: bool) -> None:
        async with self._lock:
            now = time.monotonic()
            if (
                not force
                and len(self._buffer) < self._min_chunk
                and (now - self._last_flush) < self._min_interval
            ):
                return

            while self._buffer:
                if not force and len(self._buffer) < self._min_chunk:
                    break
                chunk = self._buffer[: self._max_chunk]
                self._buffer = self._buffer[len(chunk):]
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=self._channel,
                        chat_id=self._chat_id,
                        content=chunk,
                    )
                )
                self._sent_any = True
                self._last_flush = time.monotonic()
                if not force and (time.monotonic() - self._last_flush) < self._min_interval:
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
        subtask_model: str | None = None,
        subtask_timeout: int | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
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
        self.subtask_model = subtask_model
        # Normalize common alias models early (e.g. "minimax" → "minimax/MiniMax-M2.1")
        self.model = self._normalize_model_alias(self.model)
        if self.subtask_model:
            self.subtask_model = self._normalize_model_alias(self.subtask_model)
        self.subtask_timeout = subtask_timeout
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.stream_enabled = bool(getattr(stream_config, "enabled", False)) if stream_config is not None else False
        stream_channels = getattr(stream_config, "channels", []) if stream_config is not None else []
        self.stream_channels = {c for c in stream_channels if isinstance(c, str)}
        
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
        self.active_subtasks: dict[str, dict[str, Any]] = {}
        self.completed_subtasks: dict[str, dict[str, Any]] = {}
        self.completed_subtasks_order: list[str] = []
        self.completed_subtasks_limit = 50
        self._spawned_this_turn: list[dict[str, Any]] = []
        self._current_session_model: str | None = None
        self._current_session_subtask_model: str | None = None
        self._confirm_exec = ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            include_exit_code=True,
            deny_patterns=[],
            allow_high_risk=True,
        )
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.subtask_model or self.model,
            timeout_seconds=self.subtask_timeout,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            cron_service=self.cron_service,
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

        # Delegate side-effect tools to subagents
        write_tool = WriteFileTool(allowed_dir=allowed_dir)
        self.tools.register(DelegateTool(
            name=write_tool.name,
            description=f"{write_tool.description} (delegated to subagent)",
            parameters=write_tool.parameters,
            spawn_cb=self._spawn_subtask,
            task_builder=self._build_subtask_task,
        ))
        edit_tool = EditFileTool(allowed_dir=allowed_dir)
        self.tools.register(DelegateTool(
            name=edit_tool.name,
            description=f"{edit_tool.description} (delegated to subagent)",
            parameters=edit_tool.parameters,
            spawn_cb=self._spawn_subtask,
            task_builder=self._build_subtask_task,
        ))
        if self.cron_service:
            from nanobot.agent.tools.cron import CronTool
            cron_tool = CronTool(self.cron_service)
            self.tools.register(DelegateTool(
                name=cron_tool.name,
                description=f"{cron_tool.description} (delegated to subagent)",
                parameters=cron_tool.parameters,
                spawn_cb=self._spawn_subtask,
                task_builder=self._build_subtask_task,
            ))
        
        # Read-only shell tool (delegates when unsafe)
        self.tools.register(ReadOnlyExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            confirmations=self.confirmations,
            spawn_cb=self._spawn_subtask,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(spawn_cb=self._spawn_subtask)
        self.tools.register(spawn_tool)
    
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
                    stream_buffer: _StreamBuffer | None = None
                    stream_callback: Callable[[str], None] | None = None
                    if self._should_stream_channel(msg.channel) and msg.channel != "cli":
                        stream_buffer = _StreamBuffer(
                            bus=self.bus,
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                        )
                        stream_callback = stream_buffer.on_token

                    response = await self._process_message(msg, stream_callback=stream_callback)

                    if stream_buffer is not None:
                        await stream_buffer.finish()
                        if stream_buffer.sent_any:
                            continue

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

    def _get_session_subtask_model(self, session: "Session") -> str:
        model = session.metadata.get("subtask_model") or self.subtask_model or self._get_session_model(session)
        return self._normalize_model_alias(model)

    def _truncate(self, text: str, max_len: int = 200) -> str:
        clean = " ".join(text.split())
        return clean if len(clean) <= max_len else clean[:max_len] + "..."

    def _split_name_desc(self, payload: str) -> tuple[str, str]:
        if not payload:
            return "", ""
        for sep in ("|", "：", ":"):
            if sep in payload:
                left, right = payload.split(sep, 1)
                return left.strip(), right.strip()
        return payload.strip(), ""

    def _record_completed_subtask(
        self,
        task_id: str,
        info: dict[str, Any] | None,
        status: str,
        result: str,
    ) -> None:
        entry = {
            "task_id": task_id,
            "label": (info or {}).get("label") or "subtask",
            "task": (info or {}).get("task") or "",
            "model": (info or {}).get("model") or "",
            "status": status,
            "result": result,
        }
        self.completed_subtasks[task_id] = entry
        if task_id in self.completed_subtasks_order:
            self.completed_subtasks_order.remove(task_id)
        self.completed_subtasks_order.insert(0, task_id)
        if len(self.completed_subtasks_order) > self.completed_subtasks_limit:
            drop = self.completed_subtasks_order.pop()
            self.completed_subtasks.pop(drop, None)

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

    def _allow_spawn(self, msg: InboundMessage, task: str, label: str | None) -> bool:
        user_text = msg.content or ""
        if user_text.strip().startswith("/") and not re.search(r"\\b(subtask|spawn)\\b", user_text, re.IGNORECASE):
            return False
        if re.search(r"\\b(subtask|delegate|spawn)\\b", user_text, re.IGNORECASE):
            return True
        if "子任务" in user_text or "分派" in user_text or "交给子任务" in user_text:
            return True
        if self._looks_like_readonly(user_text) or self._looks_like_readonly(task):
            return False
        if self._looks_like_side_effect(user_text):
            return True
        if self._looks_like_side_effect(task):
            return True
        if label and self._looks_like_side_effect(label):
            return True
        return False

    def _list_configured_models(self, session: "Session", current_model: str, current_subtask_model: str) -> str:
        from nanobot.config.loader import load_config
        from nanobot.providers.registry import PROVIDERS

        config = load_config()
        default_model = config.agents.defaults.model
        default_subtask_model = config.agents.subtask.model or default_model
        known_models = self._sanitize_known_models(session.metadata.get("known_models") or [])
        known_subtask_models = self._sanitize_known_models(session.metadata.get("known_subtask_models") or [])
        # Persist cleaned lists to avoid showing stale aliases
        session.metadata["known_models"] = known_models
        session.metadata["known_subtask_models"] = known_subtask_models
        merged: list[str] = []
        for candidate in [current_model, default_model, *known_models]:
            if candidate and candidate not in merged:
                merged.append(candidate)
        merged_subtask: list[str] = []
        for candidate in [current_subtask_model, default_subtask_model, *known_subtask_models]:
            if candidate and candidate not in merged_subtask:
                merged_subtask.append(candidate)

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
            f"子任务模型：{current_subtask_model}",
            f"子任务默认模型：{default_subtask_model}",
        ]
        if merged:
            lines.append("已使用模型：" + ", ".join(merged))
        if merged_subtask:
            lines.append("已使用子任务模型：" + ", ".join(merged_subtask))
        if configured_providers:
            lines.append("已配置提供商：" + ", ".join(configured_providers))
        lines.append("用法：/model <模型名> | /model list | /model reset")
        lines.append("子任务：/model sub <模型名> | /model sub reset")
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
                "分类：/memu category list | add <名称> [| 描述] | update <名称> | <新描述> | delete <名称>"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

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

    def _handle_subtask_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = msg.content.strip()
        if not raw.startswith("/subtask"):
            return None

        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        arg_lower = arg.lower()

        if not arg or arg_lower in {"list", "ls"}:
            if not self.active_subtasks and not self.completed_subtasks_order:
                content = (
                    "当前没有在跑的子任务，也没有最近的完成记录。\n"
                    "用法：/subtask run [-m <模型>] <任务> | /subtask list | /subtask recent | /subtask <task_id> | /subtask clear"
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            lines: list[str] = []
            lines.append(f"子任务概览：运行中 {len(self.active_subtasks)} 个，最近完成 {len(self.completed_subtasks_order[:10])} 个")
            if self.active_subtasks:
                lines.append("运行中：")
                for idx, (task_id, info) in enumerate(self.active_subtasks.items(), start=1):
                    label = info.get("label") or "subtask"
                    task = info.get("task") or ""
                    model = info.get("model") or "default"
                    lines.append(f"{idx}. id={task_id} | label={label} | model={model}")
                    lines.append(f"task: {self._truncate(task, 120)}")
            if self.completed_subtasks_order:
                lines.append("最近完成：")
                for idx, task_id in enumerate(self.completed_subtasks_order[:10], start=1):
                    info = self.completed_subtasks.get(task_id, {})
                    label = info.get("label") or "subtask"
                    status = info.get("status") or "ok"
                    model = info.get("model") or "default"
                    lines.append(f"{idx}. id={task_id} | label={label} | model={model} | status={status}")
            lines.append("用法：/subtask run [-m <模型>] <任务> | /subtask <id> | /subtask recent | /subtask clear")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))

        if arg_lower in {"recent", "history"}:
            if not self.completed_subtasks_order:
                content = "没有最近的子任务完成记录。"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            lines = []
            for task_id in self.completed_subtasks_order[:10]:
                info = self.completed_subtasks.get(task_id, {})
                label = info.get("label") or "subtask"
                status = info.get("status") or "ok"
                lines.append(f"- {label} (id: {task_id}): {status}")
            content = "最近完成：\n" + "\n".join(lines)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"clear", "purge"}:
            self.completed_subtasks.clear()
            self.completed_subtasks_order.clear()
            content = "已清空最近子任务记录。正在运行的子任务不受影响。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"help", "?"}:
            content = "用法：/subtask run [-m <模型>] <任务> | /subtask list | /subtask recent | /subtask <task_id> | /subtask clear"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        task_id = arg
        info = self.active_subtasks.get(task_id)
        if info:
            label = info.get("label") or "subtask"
            task = info.get("task") or ""
            content = f"子任务 {label} (id: {task_id})：\n{task}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        info = self.completed_subtasks.get(task_id)
        if info:
            label = info.get("label") or "subtask"
            status = info.get("status") or "ok"
            result = info.get("result") or ""
            content = f"子任务 {label} (id: {task_id}) 已完成，状态：{status}\n结果摘要：{self._truncate(result, 400)}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        content = f"未找到子任务：{task_id}"
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
                f"子任务模型：{self._get_session_subtask_model(session)}\n"
                "用法：/model list | /model <模型名> | /model reset\n"
                "子任务：/model sub <模型名> | /model sub reset\n"
                "清理历史：/model clean"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower.startswith("sub"):
            sub_parts = arg.split(None, 1)
            sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
            sub_lower = sub_arg.lower()
            if not sub_arg:
                content = (
                    f"子任务模型：{self._get_session_subtask_model(session)}\n"
                    "用法：/model sub <模型名> | /model sub reset"
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            if sub_lower in {"reset", "default"}:
                session.metadata.pop("subtask_model", None)
                content = f"已恢复子任务默认模型：{self.subtask_model or self.model}"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            resolved, error = self._resolve_model_input(sub_arg, session)
            if error:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=error)
            model_name = resolved or sub_arg
            session.metadata["subtask_model"] = model_name
            known_subtask = session.metadata.get("known_subtask_models") or []
            if model_name not in known_subtask:
                known_subtask.append(model_name)
                session.metadata["known_subtask_models"] = known_subtask
            content = f"已切换子任务模型：{model_name}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"list", "ls"}:
            content = self._list_configured_models(
                session, current_model, self._get_session_subtask_model(session)
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"help", "?"}:
            content = (
                "用法：/model list | /model <模型名> | /model reset\n"
                "子任务：/model sub <模型名> | /model sub reset\n"
                "清理历史：/model clean"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"reset", "default", "restart", "reload"}:
            session.metadata.pop("model", None)
            content = f"已恢复默认模型：{self.model}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if arg_lower in {"clean", "clear", "purge"}:
            session.metadata.pop("known_models", None)
            session.metadata.pop("known_subtask_models", None)
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

    def _should_stream(self, channel: str | None, stream_callback: Any | None) -> bool:
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

    async def _spawn_subtask(
        self,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
        model_override: str | None = None,
    ) -> SpawnResult:
        effective_model = model_override or self._current_session_subtask_model or self._current_session_model or self.model
        result = await self.subagents.spawn(
            task=task,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            model=effective_model,
        )
        self.active_subtasks[result.task_id] = {
            "task": result.task,
            "label": result.label,
            "model": effective_model,
            "origin": {"channel": origin_channel, "chat_id": origin_chat_id},
        }
        self._spawned_this_turn.append({
            "task_id": result.task_id,
            "label": result.label,
            "task": result.task,
            "model": effective_model,
        })
        return result

    async def _delegate_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        msg: InboundMessage,
        origin_channel: str | None = None,
        origin_chat_id: str | None = None,
    ) -> str:
        task, label = self._build_subtask_task(tool_name, args)
        channel = origin_channel or msg.channel
        chat_id = origin_chat_id or msg.chat_id
        await self._spawn_subtask(task, label, channel, chat_id)
        return f"已将 {tool_name} 交由子任务执行。"

    def _build_subtask_task(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        if tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            task = f"写入文件：{path}\\n内容：\\n{content}"
            label = f"write:{Path(path).name or 'file'}"
            return task, label
        if tool_name == "edit_file":
            path = args.get("path", "")
            old_text = args.get("old_text", "")
            new_text = args.get("new_text", "")
            task = (
                f"编辑文件：{path}\\n"
                f"替换：\\n{old_text}\\n"
                f"为：\\n{new_text}"
            )
            label = f"edit:{Path(path).name or 'file'}"
            return task, label
        if tool_name == "cron":
            task = f"设置定时任务：{json.dumps(args, ensure_ascii=False)}"
            return task, "cron"
        task = f"执行工具 {tool_name}：{json.dumps(args, ensure_ascii=False)}"
        return task, tool_name

    def _build_spawn_ack(self, user_request: str, spawned: list[dict[str, Any]]) -> str:
        def _summarize(text: str, max_len: int = 80) -> str:
            clean = " ".join(text.split())
            return clean if len(clean) <= max_len else clean[:max_len] + "..."

        if not spawned:
            return "未能启动子任务，请重试或补充说明。"
        tasks = "\n".join(
            f"- {item['label']} (id: {item['task_id']}, model: {item.get('model') or 'default'}): {_summarize(item['task'])}"
            for item in spawned
        )
        return (
            "已分派子任务：\n"
            f"{tasks}\n"
            "用 /subtask list 查看进度，用 /subtask <id> 查看详情。"
        )

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
            "minimax": "minimax/MiniMax-M2.1",
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
        default_subtask_model = config.agents.subtask.model or default_model
        known_models = self._sanitize_known_models(session.metadata.get("known_models") or [])
        known_subtask_models = self._sanitize_known_models(session.metadata.get("known_subtask_models") or [])
        current_model = self._get_session_model(session)
        current_subtask = self._get_session_subtask_model(session)

        merged: list[str] = []
        for candidate in [
            current_model,
            default_model,
            default_subtask_model,
            current_subtask,
            *known_models,
            *known_subtask_models,
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
                "minimax": ["minimax/MiniMax-M2.1"],
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

    def _parse_subtask_run_args(self, arg: str) -> tuple[str | None, str, str | None]:
        """Parse /subtask run args. Returns (model_override, task, error)."""
        arg = (arg or "").strip()
        if not arg:
            return None, "", "用法：/subtask run [-m <模型>] <任务内容>"
        tokens = arg.split()
        if tokens[0] in {"-m", "--model"}:
            if len(tokens) < 2:
                return None, "", "缺少模型名。用法：/subtask run -m <模型> <任务内容>"
            model = tokens[1].strip()
            task = " ".join(tokens[2:]).strip()
            if not task:
                return None, "", "缺少任务内容。用法：/subtask run -m <模型> <任务内容>"
            return model, task, None
        if tokens[0].startswith("--model="):
            model = tokens[0].split("=", 1)[1].strip()
            if not model:
                return None, "", "缺少模型名。用法：/subtask run --model=<模型> <任务内容>"
            task = " ".join(tokens[1:]).strip()
            if not task:
                return None, "", "缺少任务内容。用法：/subtask run --model=<模型> <任务内容>"
            return model, task, None
        return None, arg, None

    def _should_force_subtask(self, msg: InboundMessage) -> bool:
        user_text = msg.content or ""
        if not user_text.strip():
            return False
        if user_text.strip().startswith("/"):
            return False
        if self._wants_subtask_spawn(user_text):
            return True
        if self._looks_like_readonly(user_text):
            return False
        if self._looks_like_side_effect(user_text):
            return True
        return False

    def _should_force_web_search(self, msg: InboundMessage) -> bool:
        user_text = msg.content or ""
        if not user_text.strip():
            return False
        if user_text.strip().startswith("/"):
            return False
        if self._wants_subtask_spawn(user_text):
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

    def _wants_subtask_spawn(self, text: str) -> bool:
        if not text:
            return False
        if text.strip().startswith("/"):
            return False
        patterns = [
            r"(创建|开|开启|启动|分派|派|交给|生成).{0,4}子任务",
            r"子任务.*(执行|处理|帮我|代办)",
            r"\\b(create|spawn|delegate)\\s+subtask\\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

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

    async def _handle_weather(self, msg: InboundMessage) -> OutboundMessage | None:
        if not self._is_weather_intent(msg.content or ""):
            return None
        if self._wants_subtask_spawn(msg.content or ""):
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

    async def _handle_menu_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = (msg.content or "").strip()
        if not raw.startswith("/menu"):
            return None
        parts = raw.split(None, 1)
        arg = parts[1].strip().lower() if len(parts) > 1 else "list"
        if arg.startswith("model "):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/model " + arg[6:],
                media=msg.media,
                metadata=msg.metadata,
            )
            session = self.sessions.get_or_create(msg.session_key)
            current_model = self._get_session_model(session)
            return self._handle_model_command(forwarded, session, current_model)
        if arg.startswith("subtask "):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/subtask " + arg[8:],
                media=msg.media,
                metadata=msg.metadata,
            )
            return self._handle_subtask_command(forwarded)
        if arg.startswith("memu "):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg[5:],
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg.startswith("status"):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg,
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg.startswith("category"):
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/memu " + arg,
                media=msg.media,
                metadata=msg.metadata,
            )
            return await self._handle_memu_command(forwarded)
        if arg in {"version", "ver"}:
            forwarded = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content="/version",
                media=msg.media,
                metadata=msg.metadata,
            )
            return self._handle_version_command(forwarded)
        if arg not in {"list", "ls", "help", "?"}:
            arg = "list"
        lines = [
            "可用 /menu 子命令：",
            "/menu list",
            "/menu model list | /menu model <模型名> | /menu model reset",
            "/menu model sub <模型名> | /menu model sub reset",
            "/menu model clean",
            "/menu status [fast|full]",
            "/menu category list | /menu category add <名称> [| 描述]",
            "/menu category update <名称> | <新描述> | /menu category delete <名称>",
            "/menu subtask run [-m <模型>] <任务> | /menu subtask list | /menu subtask recent | /menu subtask <id> | /menu subtask clear",
            "/menu version",
        ]
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))
    
    async def _process_message(
        self,
        msg: InboundMessage,
        stream_callback: Any | None = None,
    ) -> OutboundMessage | None:
        start = time.perf_counter()
        response = await self._process_message_impl(msg, stream_callback=stream_callback)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            content_len = len(msg.content or "")
        except Exception:
            content_len = -1
        logger.info(
            "Message processed in {}ms (channel={}, sender={}, len={}, response={})",
            int(round(elapsed_ms)),
            msg.channel,
            msg.sender_id,
            content_len,
            "yes" if response else "no",
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
        # Handle system messages (subagent announces)
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
        self._spawned_this_turn = []

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

        # Handle /subtask run|spawn before LLM
        raw = (msg.content or "").strip()
        if raw.startswith("/subtask"):
            parts = raw.split(None, 2)
            if len(parts) >= 2 and parts[1].lower() in {"run", "spawn"}:
                arg = parts[2].strip() if len(parts) >= 3 else ""
                session = self.sessions.get_or_create(msg.session_key)
                self._current_session_model = self._get_session_model(session)
                self._current_session_subtask_model = self._get_session_subtask_model(session)
                model_override, task, error = self._parse_subtask_run_args(arg)
                if error:
                    content = error
                else:
                    resolved_override = None
                    if model_override:
                        resolved_override, error = self._resolve_model_input(model_override, session)
                        if error:
                            content = error
                        else:
                            resolved_override = resolved_override or model_override
                    if error:
                        pass
                    else:
                        await self._spawn_subtask(
                            task,
                            "subtask",
                            msg.channel,
                            msg.chat_id,
                            model_override=resolved_override,
                        )
                        content = self._build_spawn_ack(task, self._spawned_this_turn)
                if error:
                    session.add_message("user", msg.content)
                    session.add_message("assistant", content)
                    self.sessions.save(session)
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
                session.add_message("user", msg.content)
                session.add_message("assistant", content)
                self.sessions.save(session)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        # Handle explicit subtask intent before LLM
        if self._wants_subtask_spawn(msg.content or ""):
            session = self.sessions.get_or_create(msg.session_key)
            self._current_session_model = self._get_session_model(session)
            self._current_session_subtask_model = self._get_session_subtask_model(session)
            await self._spawn_subtask(msg.content, "subtask", msg.channel, msg.chat_id)
            content = self._build_spawn_ack(msg.content, self._spawned_this_turn)
            session.add_message("user", msg.content)
            session.add_message("assistant", content)
            self.sessions.save(session)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        # Handle weather intent before LLM
        weather_response = await self._handle_weather(msg)
        if weather_response:
            session = self.sessions.get_or_create(msg.session_key)
            session.add_message("user", msg.content)
            session.add_message("assistant", weather_response.content)
            self.sessions.save(session)
            return weather_response

        # Handle /subtask commands before LLM
        subtask_response = self._handle_subtask_command(msg)
        if subtask_response:
            session = self.sessions.get_or_create(msg.session_key)
            session.add_message("user", msg.content)
            session.add_message("assistant", subtask_response.content)
            self.sessions.save(session)
            return subtask_response
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        session_model = self._get_session_model(session)
        self._current_session_model = session_model
        self._current_session_subtask_model = self._get_session_subtask_model(session)

        # Handle /model commands before LLM
        memu_response = await self._handle_memu_command(msg)
        if memu_response:
            session.add_message("user", msg.content)
            session.add_message("assistant", memu_response.content)
            self.sessions.save(session)
            self._current_session_model = None
            self._current_session_subtask_model = None
            return memu_response

        # Handle /model commands before LLM
        model_response = self._handle_model_command(msg, session, session_model)
        if model_response:
            # Save to session
            session.add_message("user", msg.content)
            session.add_message("assistant", model_response.content)
            self.sessions.save(session)
            self._current_session_model = None
            self._current_session_subtask_model = None
            return model_response
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        exec_tool = self.tools.get("exec")
        if isinstance(exec_tool, ReadOnlyExecTool):
            exec_tool.set_context(msg.channel, msg.chat_id)

        for name in ("write_file", "edit_file", "cron"):
            delegate_tool = self.tools.get(name)
            if isinstance(delegate_tool, DelegateTool):
                delegate_tool.set_context(msg.channel, msg.chat_id)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        memory_context = ""
        memu_retrieve_ms: int | None = None
        if msg.channel != "system":
            try:
                memu_start = time.perf_counter()
                memory_context = (
                    await self.memory_adapter.retrieve_context(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        sender_id=msg.sender_id,
                        history=session.get_history(),
                        current_message=msg.content,
                    )
                ).text
                memu_retrieve_ms = int(round((time.perf_counter() - memu_start) * 1000))
                logger.info(
                    "MemU retrieve in {}ms (channel={}, sender={}, history_len={}, msg_len={})",
                    memu_retrieve_ms,
                    msg.channel,
                    msg.sender_id,
                    len(session.get_history()),
                    len(msg.content or ""),
                )
            except Exception as exc:
                logger.warning(f"MemU context fetch failed: {exc}")

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
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
                use_stream = self._should_stream(msg.channel, stream_callback)
                response = await self._get_provider_for_model(session_model).chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=session_model,
                    stream=use_stream,
                    on_token=stream_callback if use_stream else None,
                )
                llm_elapsed_ms = int(round((time.perf_counter() - llm_start) * 1000))
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
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                        if tool_call.name == "spawn":
                            task = tool_call.arguments.get("task", "")
                            label = tool_call.arguments.get("label")
                            if not self._allow_spawn(msg, task, label):
                                result = "已拒绝子任务分派：仅当明确需要副作用操作或用户明确要求时才允许。"
                            else:
                                result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        else:
                            result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )

                        if self._spawned_this_turn:
                            final_content = self._build_spawn_ack(msg.content, self._spawned_this_turn)
                            break

                    if self._spawned_this_turn:
                        break

                    # Continue loop after tool execution
                    continue

                # No tool calls: decide whether to force a subtask
                if self._should_force_subtask(msg):
                    await self._spawn_subtask(msg.content, "subtask", msg.channel, msg.chat_id)
                    final_content = self._build_spawn_ack(msg.content, self._spawned_this_turn)
                    break

                if self._should_force_web_search(msg):
                    if not self.brave_api_key:
                        final_content = "搜索未配置：请设置 BRAVE_API_KEY 或 config.tools.web.search.apiKey。"
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
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        self._current_session_model = None
        self._current_session_subtask_model = None

        outbound = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

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
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        # Reset per-turn spawn tracking for system messages too
        self._spawned_this_turn = []
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        session_model = self._get_session_model(session)
        self._current_session_model = session_model
        self._current_session_subtask_model = self._get_session_subtask_model(session)

        info: dict[str, Any] | None = None
        task_id = (msg.metadata or {}).get("task_id")
        if task_id:
            info = self.active_subtasks.pop(task_id, None)
            status = (msg.metadata or {}).get("status") or "ok"
            self._record_completed_subtask(task_id, info, status, msg.content or "")
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        exec_tool = self.tools.get("exec")
        if isinstance(exec_tool, ReadOnlyExecTool):
            exec_tool.set_context(origin_channel, origin_chat_id)
        
        # Fast-path failure/high-risk results without LLM
        parsed = parse_subtask_output(msg.content)
        status = parsed.get("status")
        if not status and (msg.metadata or {}).get("status") == "ok":
            status = "成功"
        if status in {"失败", "高风险拦截"}:
            response_text = ""
            if status == "高风险拦截" or "high_risk_command" in (msg.metadata or {}):
                command = (msg.metadata or {}).get("high_risk_command") or ""
                working_dir = (msg.metadata or {}).get("working_dir")
                if command:
                    record = self.confirmations.create(
                        command=command,
                        arguments={"command": command, "working_dir": working_dir},
                        working_dir=working_dir or str(self.workspace),
                    )
                    response_text = (
                        "子任务拦截了高风险命令，需要人工确认。\n"
                        f"Command ID: {record.id}\n"
                        f"请回复：确认 {record.id}\n"
                        "提示：Command ID 仅一次有效，默认 5 分钟过期。"
                    )
                else:
                    response_text = "子任务拦截了高风险命令，但未提供可执行命令。请补充具体命令。"
            else:
                error_cause = parsed.get("error_cause", "未知")
                response_text = (
                    f"子任务失败（错误归因：{error_cause}）。\n"
                    "请确认下一步：修正需求/补充权限/允许重试。"
                )

            session = self.sessions.get_or_create(session_key)
            session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
            session.add_message("assistant", response_text)
            self.sessions.save(session)

            return OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=response_text,
            )

        if status == "成功":
            label = (msg.metadata or {}).get("label") or (info or {}).get("label") or "subtask"
            conclusion = parsed.get("conclusion") or "已完成"
            evidence = parsed.get("evidence") or "无"
            response_text = (
                f"子任务完成：{label} (id: {task_id})\n"
                f"结论：{conclusion}\n"
                f"证据：{evidence}"
            )
            session = self.sessions.get_or_create(session_key)
            session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
            session.add_message("assistant", response_text)
            self.sessions.save(session)
            return OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=response_text,
            )

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
                    if tool_call.name == "spawn":
                        task = tool_call.arguments.get("task", "")
                        label = tool_call.arguments.get("label")
                        if not self._allow_spawn(msg, task, label):
                            result = "已拒绝子任务分派：仅当明确需要副作用操作或用户明确要求时才允许。"
                        else:
                            result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    else:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                if self._spawned_this_turn:
                    final_content = self._build_spawn_ack(msg.content, self._spawned_this_turn)
                    break
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        self._current_session_model = None
        self._current_session_subtask_model = None
        
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
        
        response = await self._process_message(msg, stream_callback=stream_callback)
        return response.content if response else ""
