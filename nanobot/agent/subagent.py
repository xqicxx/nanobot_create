"""Subagent manager for background task execution."""

import asyncio
import difflib
import json
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.guardrails import classify_command
from nanobot.agent.subtask_output import (
    format_subtask_output,
    infer_error_cause,
    render_subtask_result,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import (
    DEFAULT_MINIMAX_MCP_HOST,
    WebSearchTool,
    WebFetchTool,
    UnderstandImageTool,
)
from nanobot.agent.tools.cron import CronTool


@dataclass(frozen=True)
class SpawnResult:
    task_id: str
    label: str
    task: str
    message: str


class SubagentManager:
    """
    Manages background subagent execution.
    
    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        timeout_seconds: int | None = None,
        brave_api_key: str | None = None,
        minimax_mcp_config: Any | None = None,
        minimax_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        cron_service: "CronService | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.timeout_seconds = timeout_seconds or 300
        self.brave_api_key = brave_api_key
        self.minimax_mcp_config = minimax_mcp_config
        self.minimax_api_key = minimax_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.cron_service = cron_service
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
    
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        model: str | None = None,
    ) -> SpawnResult:
        """
        Spawn a subagent to execute a task in the background.
        
        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.
        
        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }
        
        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent_with_timeout(task_id, task, display_label, origin, model=model)
        )
        self._running_tasks[task_id] = bg_task
        
        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))
        
        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return SpawnResult(
            task_id=task_id,
            label=display_label,
            task=task,
            message=f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes.",
        )
    
    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        model: str | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        effective_model = model or self.model
        
        evidence_items: list[str] = []
        exit_code: int | None = None
        status_hint: str | None = None
        error_cause_hint: str | None = None
        risk_hint: str | None = None
        next_step_hint: str | None = None
        blocked_meta: dict[str, Any] = {}
        side_effect_count = 0

        def _truncate(text: str, max_chars: int = 2000, max_lines: int = 20) -> str:
            lines = text.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ["... (truncated)"]
            truncated = "\n".join(lines)
            if len(truncated) > max_chars:
                truncated = truncated[:max_chars] + "\n... (truncated)"
            return truncated

        workspace_root = self.workspace.resolve()

        def _read_text(path: str) -> str:
            resolved = Path(path).expanduser().resolve()
            if self.restrict_to_workspace and workspace_root not in resolved.parents and resolved != workspace_root:
                raise PermissionError("Path outside workspace")
            return resolved.read_text(encoding="utf-8")

        def _record_evidence(label: str, content: str) -> None:
            evidence_items.append(f"{label}:\n{_truncate(content)}")

        def _extract_delete_targets(cmd: str) -> list[str]:
            try:
                tokens = shlex.split(cmd)
            except Exception:
                return []
            if not tokens:
                return []
            head = tokens[0].lower()
            if head not in {"rm", "del", "rmdir"}:
                return []
            targets: list[str] = []
            for tok in tokens[1:]:
                if tok.startswith("-") or tok.startswith("/"):
                    continue
                targets.append(tok)
            return targets

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(EditFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                include_exit_code=True,
            ))
            mcp_cfg = self.minimax_mcp_config
            minimax_mcp_enabled = bool(getattr(mcp_cfg, "enabled", False)) if mcp_cfg is not None else False
            minimax_mcp_key = (getattr(mcp_cfg, "api_key", "") or self.minimax_api_key or "").strip() if mcp_cfg is not None else (self.minimax_api_key or "").strip()
            minimax_mcp_host = getattr(mcp_cfg, "api_host", DEFAULT_MINIMAX_MCP_HOST) if mcp_cfg is not None else DEFAULT_MINIMAX_MCP_HOST
            minimax_mcp_timeout = float(getattr(mcp_cfg, "timeout_seconds", 15)) if mcp_cfg is not None else 15.0
            minimax_web_enabled = bool(getattr(mcp_cfg, "enable_web_search", True)) if mcp_cfg is not None else True
            minimax_image_enabled = bool(getattr(mcp_cfg, "enable_image_understanding", True)) if mcp_cfg is not None else True

            tools.register(WebSearchTool(
                api_key=self.brave_api_key,
                minimax_enabled=minimax_mcp_enabled and minimax_web_enabled,
                minimax_api_key=minimax_mcp_key,
                minimax_api_host=minimax_mcp_host,
                minimax_timeout=minimax_mcp_timeout,
            ))
            tools.register(WebFetchTool())
            if minimax_mcp_enabled and minimax_image_enabled and minimax_mcp_key:
                tools.register(UnderstandImageTool(
                    minimax_api_key=minimax_mcp_key,
                    minimax_api_host=minimax_mcp_host,
                    timeout=minimax_mcp_timeout,
                ))
            if self.cron_service:
                cron_tool = CronTool(self.cron_service)
                cron_tool.set_context(origin["channel"], origin["chat_id"])
                tools.register(cron_tool)
            
            # Build messages with subagent-specific prompt
            task_brief = self._build_subagent_task_brief(task, label)
            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_brief},
            ]
            
            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None
            
            while iteration < max_iterations:
                iteration += 1
                
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=effective_model,
                )
                
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })
                    
                    # Execute tools (enforce single side-effect)
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name} with arguments: {args_str}")
                        _record_evidence("执行步骤", f"iter={iteration}; tool={tool_call.name}; args={args_str}")

                        is_side_effect = tool_call.name in {"write_file", "edit_file", "cron"}
                        if tool_call.name == "exec":
                            cmd = tool_call.arguments.get("command", "")
                            check = classify_command(cmd)
                            if check.high_risk:
                                status_hint = "高风险拦截"
                                error_cause_hint = "拦截"
                                risk_hint = "高风险命令必须由主会话获得人工确认后执行。"
                                next_step_hint = "请主会话向用户发起确认。"
                                blocked_meta = {
                                    "high_risk_command": cmd,
                                    "working_dir": tool_call.arguments.get("working_dir"),
                                }
                                _record_evidence("高风险命令", cmd)
                                final_result = render_subtask_result(
                                    format_subtask_output(
                                        raw="检测到高风险命令，已拦截。",
                                        status_hint=status_hint,
                                        error_cause_hint=error_cause_hint,
                                        evidence="\n\n".join(evidence_items) if evidence_items else "无",
                                        risk=risk_hint,
                                        next_step=next_step_hint,
                                        exit_code=exit_code,
                                        task_id=task_id,
                                    )
                                )
                                break
                            # Treat read-only exec as non-side-effect to avoid false blocking
                            if check.readonly_allowed and not check.dangerous_syntax:
                                is_side_effect = False
                            else:
                                is_side_effect = True
                        if is_side_effect:
                            if side_effect_count >= 1:
                                status_hint = "失败"
                                error_cause_hint = "拦截"
                                risk_hint = "单次子任务原则：一个子任务仅允许一个副作用操作。"
                                next_step_hint = "请由主会话拆分为多个子任务。"
                                _record_evidence("拦截原因", "检测到多个副作用操作尝试，已拦截。")
                                final_result = render_subtask_result(
                                    format_subtask_output(
                                        raw="副作用操作数量超过限制，已拦截。",
                                        status_hint=status_hint,
                                        error_cause_hint=error_cause_hint,
                                        evidence="\n\n".join(evidence_items) if evidence_items else "无",
                                        risk=risk_hint,
                                        next_step=next_step_hint,
                                        exit_code=exit_code,
                                        task_id=task_id,
                                    )
                                )
                                break
                            side_effect_count += 1

                        pre_content: str | None = None
                        if tool_call.name == "edit_file":
                            path = tool_call.arguments.get("path", "")
                            try:
                                pre_content = _read_text(path)
                            except Exception:
                                pre_content = None

                        result = await tools.execute(tool_call.name, tool_call.arguments)

                        if tool_call.name == "write_file":
                            path = tool_call.arguments.get("path", "")
                            try:
                                content = _read_text(path)
                                _record_evidence(f"写入文件 {path}", content)
                            except Exception as e:
                                _record_evidence(f"写入文件 {path}", f"无法读取证据：{e}")

                        if tool_call.name == "edit_file":
                            path = tool_call.arguments.get("path", "")
                            try:
                                post_content = _read_text(path)
                                if pre_content is not None:
                                    diff = "\n".join(
                                        difflib.unified_diff(
                                            pre_content.splitlines(),
                                            post_content.splitlines(),
                                            fromfile="before",
                                            tofile="after",
                                            lineterm="",
                                        )
                                    )
                                    _record_evidence(f"修改文件 {path}", diff or "(no diff)")
                                else:
                                    _record_evidence(f"修改文件 {path}", post_content)
                            except Exception as e:
                                _record_evidence(f"修改文件 {path}", f"无法读取证据：{e}")

                        if tool_call.name == "exec":
                            _record_evidence("命令输出", result)
                            if "Exit code:" in result:
                                try:
                                    exit_code = int(result.split("Exit code:")[-1].strip().split()[0])
                                except Exception:
                                    exit_code = None
                            delete_targets = _extract_delete_targets(tool_call.arguments.get("command", ""))
                            for target in delete_targets:
                                try:
                                    resolved = Path(target).expanduser().resolve()
                                    if self.restrict_to_workspace and workspace_root not in resolved.parents and resolved != workspace_root:
                                        _record_evidence("删除校验", f"{target}: 路径超出工作区，未检查")
                                    else:
                                        exists = resolved.exists()
                                        _record_evidence("删除校验", f"{target}: {'仍存在' if exists else '已删除'}")
                                except Exception as e:
                                    _record_evidence("删除校验", f"{target}: 校验失败 {e}")
                        if tool_call.name == "cron":
                            _record_evidence("定时任务结果", result)

                        if result.startswith("Error:") or result.startswith("Warning:"):
                            status_hint = "失败"
                            error_cause_hint = infer_error_cause(result)
                            risk_hint = "子任务执行失败，需主会话确认下一步。"
                            next_step_hint = "请确认是否修正参数或授权后重试。"
                            final_result = render_subtask_result(
                                format_subtask_output(
                                    raw=result,
                                    status_hint=status_hint,
                                    error_cause_hint=error_cause_hint,
                                    evidence="\n\n".join(evidence_items) if evidence_items else "无",
                                    risk=risk_hint,
                                    next_step=next_step_hint,
                                    exit_code=exit_code,
                                    task_id=task_id,
                                )
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            })
                            break

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                    if final_result is not None:
                        break
                else:
                    final_result = response.content
                    break
            
            if final_result is None:
                final_result = "Task completed but no final response was generated."

            structured = format_subtask_output(
                raw=final_result,
                status_hint=status_hint,
                error_cause_hint=error_cause_hint,
                evidence="\n\n".join(evidence_items) if evidence_items else None,
                risk=risk_hint,
                next_step=next_step_hint,
                exit_code=exit_code,
                task_id=task_id,
            )
            final_result = render_subtask_result(structured)

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok", blocked_meta)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            structured = format_subtask_output(
                raw=error_msg,
                status_hint="失败",
                error_cause_hint=infer_error_cause(error_msg),
                evidence="无",
                risk="子任务异常失败。",
                next_step="请检查异常信息并确认下一步。",
                task_id=task_id,
            )
            await self._announce_result(
                task_id,
                label,
                task,
                render_subtask_result(structured),
                origin,
                "error",
                {},
            )

    async def _run_subagent_with_timeout(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        model: str | None = None,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._run_subagent(task_id, task, label, origin, model=model),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            error_msg = f"Error: Subtask timed out after {self.timeout_seconds}s"
            logger.error(f"Subagent [{task_id}] timed out")
            structured = format_subtask_output(
                raw=error_msg,
                status_hint="失败",
                error_cause_hint="超时",
                evidence="无",
                risk="子任务执行超时。",
                next_step="请缩小任务范围或增加超时时间后重试。",
                task_id=task_id,
            )
            await self._announce_result(
                task_id,
                label,
                task,
                render_subtask_result(structured),
                origin,
                "error",
                {},
            )
    
    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
        meta: dict[str, Any],
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subtask '{label}' {status_text}]

Task ID: {task_id}
Task: {task}

Result:
{result}
"""
        
        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            metadata={"task_id": task_id, "label": label, "status": status, **(meta or {})},
        )
        
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")
    
    def _build_subagent_task_brief(self, task: str, label: str) -> str:
        return f"""任务标题：{label}

原始任务：
{task}

执行要求（MiniMax Coding Plan best practice）：
1. 先识别“目标 + 意图”，只做与目标直接相关的动作。
2. 若是长任务，按最小可完成子步骤串行推进，并保留步骤证据。
3. 优先调用最必要的工具，避免无意义并行与重复调用。
4. 输出必须可验证：命令输出、文件差异或结果片段。"""

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else.
2. Follow instruction + intent from the user task brief before making tool calls.
3. If task is long, decompose into minimal sequential steps and track step evidence.
4. Only one side-effect action per task (single write/edit/exec). If more needed, stop and report.
5. Your final response must be structured with: 状态/错误归因/结论/证据/风险/下一步/Exit Code.
6. Do not initiate conversations or take on side tasks.

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the assigned task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
"""
    
    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
