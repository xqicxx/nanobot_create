"""Read-only exec tool with guardrails and delegation."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from nanobot.agent.confirmations import ConfirmationStore
from nanobot.agent.guardrails import classify_command, has_sensitive_keywords, has_sensitive_paths
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.shell import ExecTool


SpawnCallback = Callable[[str, str | None, str, str], Awaitable[Any]]


class ReadOnlyExecTool(Tool):
    """Execute a read-only shell command with strict guardrails."""

    def __init__(
        self,
        working_dir: str,
        timeout: int = 60,
        restrict_to_workspace: bool = False,
        confirmations: ConfirmationStore | None = None,
        spawn_cb: SpawnCallback | None = None,
    ) -> None:
        self._runner = ExecTool(
            working_dir=working_dir,
            timeout=timeout,
            restrict_to_workspace=restrict_to_workspace,
        )
        self._default_cwd = working_dir
        self._confirmations = confirmations
        self._spawn_cb = spawn_cb
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a read-only shell command from a strict allowlist. "
            "Commands outside the allowlist or with dangerous syntax are delegated."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The read-only shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        check = classify_command(command)
        cwd = working_dir or self._default_cwd

        if check.high_risk:
            if not self._confirmations:
                return "Error: high-risk command requires confirmation, but confirmations are unavailable."
            record = self._confirmations.create(
                command=command,
                arguments={"command": command, "working_dir": working_dir},
                working_dir=cwd,
            )
            return (
                "检测到高风险命令，需人工确认后执行。\n"
                f"Command ID: {record.id}\n"
                f"请回复：确认 {record.id}\n"
                "提示：Command ID 仅一次有效，默认 5 分钟过期。"
            )

        if check.sensitive:
            # Allow sensitive path matches if the command stays within the workspace,
            # but always block explicit sensitive keywords.
            if has_sensitive_keywords(command):
                return await self._delegate(
                    command,
                    "由于涉及敏感关键词，为了安全，我将启动子任务进行受限访问。",
                )
            if has_sensitive_paths(command):
                if self._default_cwd and self._default_cwd in command:
                    pass
                else:
                    return await self._delegate(
                        command,
                        "由于涉及敏感路径，为了安全，我将启动子任务进行受限访问。",
                    )

        if check.dangerous_syntax or not check.readonly_allowed:
            return await self._delegate(
                command,
                "命令不在只读白名单或包含危险语法，已改为子任务执行。",
            )

        return await self._runner.execute(command=command, working_dir=working_dir)

    async def _delegate(self, command: str, notice: str) -> str:
        if not self._spawn_cb:
            return f"Error: {notice}（但子任务不可用）"
        task = f"{notice}\n执行命令：{command}"
        await self._spawn_cb(task, "exec", self._origin_channel, self._origin_chat_id)
        return notice
