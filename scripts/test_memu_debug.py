#!/usr/bin/env python3
"""Test MemU status and memory reading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "nanoBot_memU" / "nanobot"))

from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter

print("="*60)
print("🔍 MemU 诊断测试")
print("="*60)

# 1. Load config
config = load_config()
print(f"\n1. 配置加载成功")
print(f"   MemU enabled in config: {config.memu.enabled}")

# 2. Create MemoryAdapter
adapter = MemoryAdapter(
    workspace=config.workspace_path,
    enable_memory=True,
    memu_config=config.memu,
)

print(f"\n2. MemoryAdapter 创建成功")
print(f"   enable_memory: {adapter.enable_memory}")
print(f"   _memory_agent: {adapter._memory_agent}")

# 3. Check memory directory
memory_dir = config.workspace_path / ".memu" / "memory"
print(f"\n3. 检查记忆目录")
print(f"   路径: {memory_dir}")
print(f"   存在: {memory_dir.exists()}")

if memory_dir.exists():
    import os
    print(f"\n   目录内容:")
    for root, dirs, files in os.walk(memory_dir):
        level = root.replace(str(memory_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"   {indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"   {subindent}{file}")

# 4. Test retrieve_context
import asyncio

async def test_retrieve():
    print(f"\n4. 测试记忆读取")
    ctx = await adapter.retrieve_context(
        channel="whatsapp",
        chat_id="test",
        sender_id="user",
        history=[],
        current_message="我的长期目标是什么？",
    )
    print(f"   返回的上下文: {ctx.text[:200] if ctx.text else '(空)'}")

asyncio.run(test_retrieve())

# 5. Test query_items
async def test_query():
    print(f"\n5. 测试 query_items")
    items = await adapter.query_items(
        channel="whatsapp",
        chat_id="test",
        sender_id="user",
        limit=10,
    )
    print(f"   返回 {len(items)} 条记忆")
    for i, item in enumerate(items[:3], 1):
        print(f"   {i}. {item.get('memory_type')}: {item.get('content', '')[:50]}...")

asyncio.run(test_query())

print("\n" + "="*60)
print("✅ 诊断完成")
print("="*60)
