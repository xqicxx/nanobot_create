#!/usr/bin/env python3
"""
memu-py 性能测试脚本
测试不同操作的响应时间
"""

import os
import sys
import time
from pathlib import Path

# 使用虚拟环境
VENV_PYTHON = "/root/nanobot-venv/bin/python"
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON, __file__] + sys.argv[1:])

sys.path.insert(0, "/root/nanoBot_memU/nanobot")

from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter

def test_llm_speed():
    """测试 LLM 响应速度"""
    print("\n" + "=" * 50)
    print("测试 LLM 响应速度")
    print("=" * 50)

    from memu.llm import DeepSeekClient

    config = load_config()
    memu_cfg = config.memu
    default_cfg = getattr(memu_cfg, "default", None)

    if not default_cfg:
        print("❌ 未找到 MemU 配置")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY", "") or getattr(default_cfg, "api_key", "")
    base_url = getattr(default_cfg, "base_url", "https://api.deepseek.com/v1")
    chat_model = getattr(default_cfg, "chat_model", "deepseek-chat")

    print(f"模型: {chat_model}")
    print(f"API: {base_url}")

    client = DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model_name=chat_model,
    )

    # 测试1: 简单请求
    print("\n📝 测试1: 简单请求 (1句话)")
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
        )
        elapsed = time.perf_counter() - start
        print(f"✅ 耗时: {elapsed:.2f}秒")
        print(f"   回复: {response.choices[0].message.content[:50]}")
    except Exception as e:
        print(f"❌ 失败: {e}")

    # 测试2: 复杂请求（模拟记忆处理）
    print("\n📝 测试2: 复杂请求 (模拟记忆分析)")
    conversation = [
        {"role": "user", "content": "我今天学会了 Python 编程，特别开心！我还认识了新朋友小明，他教我写代码。"},
        {"role": "assistant", "content": "太棒了！Python 是很有用的编程语言，恭喜你学会了！小明也是很好的朋友。"},
    ]
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "分析这段对话，提取关键信息。"},
                *conversation,
            ],
            max_tokens=100,
        )
        elapsed = time.perf_counter() - start
        print(f"✅ 耗时: {elapsed:.2f}秒")
    except Exception as e:
        print(f"❌ 失败: {e}")

def test_memory_agent():
    """测试 MemoryAgent 速度"""
    print("\n" + "=" * 50)
    print("测试 MemoryAgent 速度")
    print("=" * 50)

    config = load_config()
    adapter = MemoryAdapter(
        workspace=config.workspace_path,
        memu_config=config.memu,
    )

    if not adapter._memory_agent:
        print("❌ MemoryAgent 未初始化")
        return

    # 测试记忆处理速度
    print("\n📝 测试记忆处理...")
    conversation = [
        {"role": "user", "content": "测试消息：今天天气真好！"},
        {"role": "assistant", "content": "是啊，很适合出去走走。"},
    ]

    import asyncio
    start = time.perf_counter()
    asyncio.run(adapter.memorize_turn(
        channel="test",
        chat_id="speed-test",
        sender_id="test-user",
        user_message=conversation[0]["content"],
        assistant_message=conversation[1]["content"],
    ))
    elapsed = time.perf_counter() - start
    print(f"✅ 总耗时: {elapsed:.2f}秒")

if __name__ == "__main__":
    test_llm_speed()
    test_memory_agent()
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
