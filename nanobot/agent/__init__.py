"""Agent core module (lazy imports to avoid heavy dependencies at import time)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.context import ContextBuilder
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.skills import SkillsLoader


def __getattr__(name: str):
    if name == "AgentLoop":
        from nanobot.agent.loop import AgentLoop
        return AgentLoop
    if name == "ContextBuilder":
        from nanobot.agent.context import ContextBuilder
        return ContextBuilder
    if name == "MemoryStore":
        from nanobot.agent.memory import MemoryStore
        return MemoryStore
    if name == "SkillsLoader":
        from nanobot.agent.skills import SkillsLoader
        return SkillsLoader
    raise AttributeError(f"module 'nanobot.agent' has no attribute {name}")
