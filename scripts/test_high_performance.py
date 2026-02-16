#!/usr/bin/env python3
"""测试高性能 memU 组件"""

import asyncio
import sys
import os
from pathlib import Path

# 使用虚拟环境的 Python
venv_python = "/root/nanobot-venv/bin/python"
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])

sys.path.insert(0, "/root/nanoBot_memU/nanobot")

from nanobot.agent.memory_decider import MemoryTriggerDecider, TriggerResult
from nanobot.agent.memory_cache import EmbeddingCache, RetrievalCache
from nanobot.agent.memory_throttle import RetrieveThrottler
from nanobot.agent.memory_queue import MemoryWriteQueue
from nanobot.agent.memory_performance import MemUPerformanceConfig


def test_memory_trigger_decider():
    """测试触发决策器"""
    print("\n" + "=" * 50)
    print("测试1: MemoryTriggerDecider")
    print("=" * 50)

    decider = MemoryTriggerDecider()

    # 测试写入触发
    test_cases_memorize = [
        ("记住我的名字叫小明", True, "显式触发"),
        ("我喜欢吃苹果", True, "偏好信息"),
        ("我25岁，在北京工作", True, "个人信息"),
        ("帮我记住这件事", True, "显式触发"),
        ("好的", False, "短回复跳过"),
        ("收到", False, "短回复跳过"),
        ("嗯嗯", False, "短回复跳过"),
        ("今天天气不错", False, "无触发条件"),
    ]

    print("\n--- 写入触发测试 ---")
    for message, expected, desc in test_cases_memorize:
        result = decider.should_memorize(message)
        status = "✅" if result.should_trigger == expected else "❌"
        print(f"{status} '{message[:20]}...' -> should_trigger={result.should_trigger}, reason={result.reason}")

    # 测试检索触发
    test_cases_retrieve = [
        ("我记得你之前说过什么", True, "明确触发"),
        ("什么是向量？", True, "疑问句"),
        ("我的名字是什么？", True, "疑问句"),
        ("今天天气不错", False, "无需检索"),
        ("好的", False, "短回复跳过"),
    ]

    print("\n--- 检索触发测试 ---")
    for query, expected, desc in test_cases_retrieve:
        result = decider.should_retrieve(query)
        status = "✅" if result.should_trigger == expected else "❌"
        print(f"{status} '{query[:20]}...' -> should_trigger={result.should_trigger}, reason={result.reason}")

    # 测试分类推断
    print("\n--- 分类推断测试 ---")
    category_tests = [
        ("我喜欢吃苹果", "interest"),
        ("我叫小明", "profile"),
        ("提醒我明天开会", "reminder"),
        ("今天参加了会议", "event"),
    ]
    for message, expected in category_tests:
        category = decider.get_category_from_message(message)
        status = "✅" if category == expected else "❌"
        print(f"{status} '{message[:15]}...' -> category={category} (expected={expected})")

    print("\n✅ MemoryTriggerDecider 测试完成")


def test_embedding_cache():
    """测试 Embedding 缓存"""
    print("\n" + "=" * 50)
    print("测试2: EmbeddingCache")
    print("=" * 50)

    cache = EmbeddingCache(max_size=3, ttl=10)

    # 测试基本操作
    vector1 = [0.1, 0.2, 0.3]
    vector2 = [0.4, 0.5, 0.6]

    cache.set("hello", vector1)
    cache.set("world", vector2)

    result1 = cache.get("hello")
    result2 = cache.get("world")
    result3 = cache.get("not exist")

    print(f"✅ 缓存 set/get: {result1 == vector1 and result2 == vector2 and result3 is None}")

    # 测试 LRU 淘汰
    cache.set("a", [0.1])
    cache.set("b", [0.2])
    cache.set("c", [0.3])  # 触发淘汰
    cache.set("d", [0.4])  # 触发淘汰

    # "a" 应该被淘汰
    print(f"✅ LRU 淘汰: {cache.get('a') is None}")

    # 测试大小
    print(f"✅ 缓存大小: {cache.size()}")

    print("\n✅ EmbeddingCache 测试完成")


def test_retrieve_throttler():
    """测试检索节流器"""
    print("\n" + "=" * 50)
    print("测试3: RetrieveThrottler")
    print("=" * 50)

    throttler = RetrieveThrottler(
        max_per_step=2,
        max_per_minute=5,
        cooldown_seconds=1,  # 1秒冷却用于测试
    )

    # 测试每轮限制
    print(f"✅ 第1次检索: {throttler.can_retrieve('query1')}")
    print(f"✅ 第2次检索: {throttler.can_retrieve('query2')}")
    print(f"✅ 第3次检索(应被限制): {throttler.can_retrieve('query3')}")

    # 测试冷却
    throttler.record_retrieve("query1")
    throttler.record_retrieve("query2")
    throttler.reset_per_step()

    print(f"✅ 冷却后第1次: {throttler.can_retrieve('query1')}")
    print(f"✅ 冷却中第2次(应被限制): {throttler.can_retrieve('query1')}")

    # 测试统计
    stats = throttler.get_stats()
    print(f"✅ 统计: step_count={stats.step_count}, cooldown_hits={stats.cooldown_hits}")

    print("\n✅ RetrieveThrottler 测试完成")


async def test_write_queue():
    """测试写入队列"""
    print("\n" + "=" * 50)
    print("测试4: MemoryWriteQueue")
    print("=" * 50)

    write_count = []

    async def mock_flush(content, category, user_id, metadata):
        write_count.append((content, category, user_id))
        print(f"   写入: {content[:20]}... ({category})")

    queue = MemoryWriteQueue(batch_size=3, flush_interval=60, max_queue_size=10)
    queue.set_flush_callback(mock_flush)

    # 测试入队
    await queue.enqueue("内容1", "interest", "user1", priority=1)
    await queue.enqueue("内容2", "profile", "user2", priority=2)
    await queue.enqueue("内容3", "activity", "user3", priority=1)

    print(f"✅ 入队3条，队列大小: {queue.size()}")

    # 手动刷新
    await queue.flush()
    print(f"✅ 刷新后，队列大小: {queue.size()}, 写入条数: {len(write_count)}")

    # 测试统计
    stats = queue.get_stats()
    print(f"✅ 统计: enqueued={stats.enqueued}, flushed={stats.flushed}")

    print("\n✅ MemoryWriteQueue 测试完成")


def test_performance_config():
    """测试性能配置"""
    print("\n" + "=" * 50)
    print("测试5: MemUPerformanceConfig")
    print("=" * 50)

    # 默认配置
    config = MemUPerformanceConfig()
    print(f"✅ 默认配置 enable_trigger_rules: {config.enable_trigger_rules}")
    print(f"✅ 默认配置 embedding_cache_size: {config.embedding_cache_size}")
    print(f"✅ 默认配置 write_batch_size: {config.write_batch_size}")

    # 自定义配置
    custom_config = MemUPerformanceConfig(
        enable_trigger_rules=False,
        embedding_cache_size=500,
        write_batch_size=10,
    )
    print(f"✅ 自定义配置 enable_trigger_rules: {custom_config.enable_trigger_rules}")
    print(f"✅ 自定义配置 embedding_cache_size: {custom_config.embedding_cache_size}")
    print(f"✅ 自定义配置 write_batch_size: {custom_config.write_batch_size}")

    # 从字典创建
    config_dict = {
        "enable_trigger_rules": True,
        "embedding_cache_size": 2000,
        "retrieve_throttle_per_step": 2,
    }
    config_from_dict = MemUPerformanceConfig.from_dict(config_dict)
    print(f"✅ 从字典创建: embedding_cache_size={config_from_dict.embedding_cache_size}")

    print("\n✅ MemUPerformanceConfig 测试完成")


def main():
    print("\n" + "=" * 60)
    print("       高性能 memU 组件测试")
    print("=" * 60)

    test_memory_trigger_decider()
    test_embedding_cache()
    test_retrieve_throttler()
    asyncio.run(test_write_queue())
    test_performance_config()

    print("\n" + "=" * 60)
    print("       ✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
