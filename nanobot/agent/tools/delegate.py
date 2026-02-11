"""Delegating tool that spawns subagents for side-effect actions."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool

SpawnCallback = Callable[[str, str | None, str, str], Awaitable[Any]]
TaskBuilder = Callable[[str, dict[str, Any]], tuple[str, str]]


class DelegateTool(Tool):
    """Tool wrapper that delegates execution to a subagent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        spawn_cb: SpawnCallback,
        task_builder: TaskBuilder,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._spawn_cb = spawn_cb
        self._task_builder = task_builder
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for delegated execution."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        if not self._spawn_cb:
            return f"Error: delegate tool '{self._name}' is not configured"
        task, label = self._task_builder(self._name, kwargs)
        await self._spawn_cb(task, label, self._origin_channel, self._origin_chat_id)
        return f"已将 {self._name} 交由子任务执行。"
