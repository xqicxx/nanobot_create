#!/usr/bin/env python3
"""测试 memU 写入和读取"""
import os
import sys
import subprocess

# 使用虚拟环境的 Python
venv_python = "/root/nanobot-venv/bin/python"
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])

# 设置环境变量
os.environ['MEMU_EMBEDDING_MODEL'] = 'BAAI/bge-m3'

sys.path.insert(0, '/root/nanoBot_memU/nanobot')

from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter
import asyncio

def main():
    config = load_config()
    adapter = MemoryAdapter(
        workspace=config.workspace_path,
        memu_config=config.memu,
    )

    print("=" * 50)
    print("测试1: 写入记忆 (减少迭代以加快速度)")
    print("=" * 50)

    # 测试写入 - 直接调用 agent.run 减少迭代
    if adapter._memory_agent:
        conversation = [
            {"role": "user", "content": "我喜欢吃苹果"},
            {"role": "assistant", "content": "好的，我记住了！"},
        ]
        result = adapter._memory_agent.run(
            conversation=conversation,
            character_name="cli:test:user",
            max_iterations=2,  # 减少到2轮，加快测试
        )
        print(f"写入结果: {result.get('success')}, function_calls: {len(result.get('function_calls', []))}")
    print("写入完成，等待处理...")

    # 等待一下让后台任务完成
    import time
    print("等待记忆处理完成...")
    time.sleep(15)  # 等待更长时间

    print("\n" + "=" * 50)
    print("测试2: 读取记忆")
    print("=" * 50)

    async def test_read():
        ctx = await adapter.retrieve_context(
            channel='cli',
            chat_id='test',
            sender_id='user',
            history=[],
            current_message='我喜欢什么？',
        )
        print('读取结果:')
        print(ctx.text)

    asyncio.run(test_read())

if __name__ == "__main__":
    main()
