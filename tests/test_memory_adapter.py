from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nanobot.agent.memory_adapter import MemoryAdapter


class DummyMemU:
    def __init__(self):
        self.retrieve_calls: list[tuple[list[dict], dict]] = []
        self.memorize_calls: list[tuple[str, str, dict]] = []

    async def retrieve(self, *, queries: list[dict], where: dict):
        self.retrieve_calls.append((queries, where))
        return {
            "categories": [{"name": "preferences", "summary": "User prefers concise answers."}],
            "items": [{"memory_type": "profile", "summary": "User likes coffee."}],
            "resources": [{"url": "file://turn.json", "caption": "conversation"}],
        }

    async def memorize(self, *, resource_url: str, modality: str, user: dict):
        self.memorize_calls.append((resource_url, modality, user))
        return {"ok": True}


def test_build_user_id() -> None:
    assert MemoryAdapter.build_user_id("telegram", "123", "456") == "telegram:123:456"
    assert MemoryAdapter.build_user_id(None, None, None) == "default:unknown:system"


@pytest.mark.parametrize(
    ("message", "previous", "expected"),
    [
        ("   ", None, True),
        ("!!!", None, True),
        ("😀😀", None, True),
        ("OK", None, True),
        ("收到", None, True),
        ("same", "same", True),
        ("I like coffee", None, False),
        ("我喜欢咖啡", None, False),
    ],
)
def test_should_skip_write(message: str, previous: str | None, expected: bool) -> None:
    adapter = MemoryAdapter(workspace=Path("."), memu_service=DummyMemU(), resources_dir=Path("."))
    assert adapter.should_skip_write(message, previous) is expected


def test_retrieve_context_formats_output() -> None:
    dummy = DummyMemU()
    adapter = MemoryAdapter(workspace=Path("."), memu_service=dummy, resources_dir=Path("."))
    ctx = asyncio.run(
        adapter.retrieve_context(
            channel="telegram",
            chat_id="c1",
            sender_id="u1",
            history=[{"role": "user", "content": "hi"}],
            current_message="what do I like?",
        )
    )
    assert dummy.retrieve_calls
    assert ctx.text.startswith("# Memory (MemU)")
    assert "## Items" in ctx.text
    assert "User likes coffee" in ctx.text


def test_memorize_turn_writes_resource_and_calls_memu(tmp_path: Path) -> None:
    dummy = DummyMemU()
    adapter = MemoryAdapter(workspace=tmp_path, memu_service=dummy, resources_dir=tmp_path / "resources")
    asyncio.run(
        adapter.memorize_turn(
            channel="telegram",
            chat_id="c1",
            sender_id="u1",
            user_message="I like coffee",
            assistant_message="Noted.",
            metadata={"k": "v"},
        )
    )

    assert len(dummy.memorize_calls) == 1
    resource_url, modality, user = dummy.memorize_calls[0]
    assert modality == "conversation"
    assert user == {"user_id": "telegram:c1:u1"}
    assert Path(resource_url).exists()
