"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.confirmations import ConfirmationStore
from nanobot.agent.subtask_output import parse_subtask_output
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.readonly_exec import ReadOnlyExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.subagent import SubagentManager, SpawnResult
from nanobot.session.manager import SessionManager


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
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.subtask_model = subtask_model
        self.subtask_timeout = subtask_timeout
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.context = ContextBuilder(workspace)
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
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
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
                    response = await self._process_message(msg)
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
        model = session.metadata.get("model")
        return model or self.model

    def _get_session_subtask_model(self, session: "Session") -> str:
        model = session.metadata.get("subtask_model")
        return model or self.subtask_model or self._get_session_model(session)

    def _truncate(self, text: str, max_len: int = 200) -> str:
        clean = " ".join(text.split())
        return clean if len(clean) <= max_len else clean[:max_len] + "..."

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
        known_models = session.metadata.get("known_models") or []
        known_subtask_models = session.metadata.get("known_subtask_models") or []
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
        return "\n".join(lines)

    def _handle_subtask_command(self, msg: InboundMessage) -> OutboundMessage | None:
        raw = msg.content.strip()
        if not raw.startswith("/subtask"):
            return None

        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        arg_lower = arg.lower()

        if not arg or arg_lower in {"list", "ls"}:
            if not self.active_subtasks and not self.completed_subtasks_order:
                content = "当前没有在跑的子任务，也没有最近的完成记录。"
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            items = []
            for task_id, info in self.active_subtasks.items():
                label = info.get("label") or "subtask"
                task = info.get("task") or ""
                items.append(f"- {label} (id: {task_id}): {self._truncate(task, 80)}")
            completed = []
            for task_id in self.completed_subtasks_order[:10]:
                info = self.completed_subtasks.get(task_id, {})
                label = info.get("label") or "subtask"
                status = info.get("status") or "ok"
                completed.append(f"- {label} (id: {task_id}): {status}")
            content = ""
            if items:
                content += "正在运行的子任务：\n" + "\n".join(items)
            if completed:
                if content:
                    content += "\n"
                content += "最近完成：\n" + "\n".join(completed)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

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

        if arg_lower in {"help", "?"}:
            content = "用法：/subtask list | /subtask recent | /subtask <task_id> | /subtask run <任务>"
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
                "子任务：/model sub <模型名> | /model sub reset"
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
            model_name = sub_arg
            if not self._model_is_configured(model_name):
                content = (
                    f"子任务模型不可用或未配置对应提供商：{model_name}\n"
                    "用法：/model list 查看可用提供商"
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
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

        if arg_lower in {"reset", "default"}:
            session.metadata.pop("model", None)
            content = f"已恢复默认模型：{self.model}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        # Set model override (per-session)
        model_name = arg
        if not self._model_is_configured(model_name):
            content = (
                f"模型不可用或未配置对应提供商：{model_name}\n"
                "用法：/model list 查看可用提供商"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

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
    ) -> SpawnResult:
        result = await self.subagents.spawn(
            task=task,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            model=self._current_session_subtask_model or self._current_session_model or self.model,
        )
        self.active_subtasks[result.task_id] = {
            "task": result.task,
            "label": result.label,
            "origin": {"channel": origin_channel, "chat_id": origin_chat_id},
        }
        self._spawned_this_turn.append({
            "task_id": result.task_id,
            "label": result.label,
            "task": result.task,
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
            f"- {item['label']} (id: {item['task_id']}): {_summarize(item['task'])}"
            for item in spawned
        )
        return (
            "已分派子任务：\n"
            f"{tasks}\n"
            "用 /subtask list 查看进度，用 /subtask <id> 查看详情。"
        )

    def _should_force_subtask(self, msg: InboundMessage) -> bool:
        user_text = msg.content or ""
        if not user_text.strip():
            return False
        if user_text.strip().startswith("/"):
            return False
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
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
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
        confirm_id = self._extract_confirmation_id(msg.content)
        if confirm_id:
            return await self._handle_confirmation(confirm_id, msg)

        # Reset per-turn spawn tracking early
        self._spawned_this_turn = []

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
            if len(parts) >= 3 and parts[1].lower() in {"run", "spawn"}:
                task = parts[2].strip()
                if task:
                    await self._spawn_subtask(task, "subtask", msg.channel, msg.chat_id)
                    content = self._build_spawn_ack(task, self._spawned_this_turn)
                else:
                    content = "用法：/subtask run <任务内容>"
                session = self.sessions.get_or_create(msg.session_key)
                session.add_message("user", msg.content)
                session.add_message("assistant", content)
                self.sessions.save(session)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

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
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
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
                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=session_model
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
                        if tool_call.name in {"write_file", "edit_file", "cron"}:
                            result = await self._delegate_tool_call(
                                tool_call.name,
                                tool_call.arguments,
                                msg,
                            )
                        elif tool_call.name == "spawn":
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
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
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

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=session_model
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
                    if tool_call.name in {"write_file", "edit_file", "cron"}:
                        result = await self._delegate_tool_call(tool_call.name, tool_call.arguments, msg)
                    elif tool_call.name == "spawn":
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
        
        response = await self._process_message(msg)
        return response.content if response else ""
