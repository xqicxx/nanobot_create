# 高性能 memU 接入架构文档

> 此文档记录了 nanobot 的高性能 memU 记忆系统架构
> 适用于需要性能优化或自定义修改的场景

---

## 一、架构概述

### 设计目标

| 目标 | 说明 |
|------|------|
| 解耦执行循环 | 记忆操作与主循环分离，异步执行 |
| 按需触发 | 只在必要时调用 embedding/检索 |
| 智能缓存 | 相同 query 不重复 embedding |
| 延迟写入 | 批量写入减少 API 调用 |
| 节流限流 | 防止记忆操作淹没主流程 |

### 架构图

```
用户输入 → Planner / Orchestrator
              ↓
        ┌─────────────────────┐
        │  MemoryTriggerDecider  │ ← 轻量同步决策
        │  should_memorize()      │
        │  should_retrieve()      │
        └───────────┬─────────────┘
                    │
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
 检索管道         主循环          写入管道
  (异步)         执行            (延迟队列)
    ↓
 EmbeddingCache  ←── 缓存
    ↓
 RetrieveThrottler ←── 节流
```

---

## 二、核心组件

### 2.1 文件清单

| 文件 | 位置 | 功能 |
|------|------|------|
| `memory_performance.py` | `nanobot/agent/` | 性能配置类 |
| `memory_decider.py` | `nanobot/agent/` | 触发决策器 |
| `memory_cache.py` | `nanobot/agent/` | Embedding + 检索缓存 |
| `memory_throttle.py` | `nanobot/agent/` | 检索节流器 |
| `memory_queue.py` | `nanobot/agent/` | 延迟写入队列 |
| `memory_adapter.py` | `nanobot/agent/` | 主适配器（已集成） |

---

## 三、配置说明

### 3.1 MemUPerformanceConfig

位置: [memory_performance.py](nanobot/agent/memory_performance.py)

```python
@dataclass
class MemUPerformanceConfig:
    # === 触发规则 ===
    enable_trigger_rules: bool = True           # 启用触发规则
    should_memorize_fn: Callable | None = None  # 自定义写入触发函数
    should_retrieve_fn: Callable | None = None  # 自定义检索触发函数

    # === 缓存策略 ===
    embedding_cache_enabled: bool = True        # 启用 embedding 缓存
    embedding_cache_size: int = 1000           # 缓存最大条目
    embedding_cache_ttl: int = 3600            # 缓存过期时间(秒)

    # === 节流策略 ===
    retrieve_throttle_per_step: int = 1        # 每轮最多检索次数
    retrieve_throttle_per_minute: int = 5      # 每分钟最多检索次数
    retrieve_cooldown_seconds: int = 10        # 同类查询冷却时间

    # === 写入策略 ===
    write_queue_enabled: bool = True           # 启用延迟写入队列
    write_batch_size: int = 5                  # 批量写入大小
    write_flush_interval: int = 30             # 强制刷新间隔(秒)
    write_max_queue_size: int = 20             # 最大队列长度
```

### 3.2 使用自定义配置

```python
from nanobot.agent.memory_performance import MemUPerformanceConfig

config = MemUPerformanceConfig(
    enable_trigger_rules=True,
    embedding_cache_enabled=True,
    retrieve_throttle_per_step=1,
    write_queue_enabled=True,
    write_batch_size=3,
)

adapter = MemoryAdapter(
    workspace=workspace,
    performance_config=config,
    memu_config=config.memu,
)
```

---

## 四、触发规则

### 4.1 写入触发规则

位置: [memory_decider.py:32-60](nanobot/agent/memory_decider.py#L32-L60)

| 类型 | 优先级 | 示例 |
|------|--------|------|
| 显式触发 | 10 | "记住"、"请记住"、"帮我记住" |
| 个人信息 | 9 | "我叫小明"、"我25岁" |
| 偏好信息 | 8 | "我喜欢苹果"、"我讨厌辣" |
| 多轮对话 | 5 | 对话长度 >= 4 轮 |
| 跳过 | 0 | 短回复（"好的"、"收到"） |

**修改触发词**:
```python
class MemoryTriggerDecider:
    EXPLICIT_MEMORIZE = [
        "记住", "请记住", "帮我记住",  # 在这里添加新词
    ]

    PREFERENCE_PATTERNS = [
        r"我喜欢", r"我讨厌",  # 在这里添加正则
    ]
```

### 4.2 检索触发规则

位置: [memory_decider.py:62-75](nanobot/agent/memory_decider.py#L62-L75)

| 类型 | 策略 | 示例 |
|------|------|------|
| 明确触发 | full | "我记得"、"之前" |
| 疑问句 | quick | "什么？"、"为什么？" |
| 历史有记忆 | refresh | 上轮已有记忆 |

**修改检索触发**:
```python
class MemoryTriggerDecider:
    RETRIEVE_TRIGGERS = [
        "我记得", "之前",  # 在这里添加
    ]

    QUESTION_WORDS = ["什么", "哪", "谁", "怎"]  # 在这里添加
```

---

## 五、API 参考

### 5.1 MemoryAdapter 新增方法

```python
# 触发决策
adapter.should_memorize(message, conversation)  # 是否应该写入
adapter.should_retrieve(query, history)         # 是否应该检索

# 节流控制
adapter.record_retrieve(query)    # 记录检索（用于节流计数）
adapter.reset_per_step()          # 每轮重置（在 step 开始时调用）

# 延迟写入队列
adapter.memorize_queued(content, category, user_id, priority=0)  # 加入队列
adapter.flush_write_queue()       # 强制刷新

# 性能统计
adapter.get_performance_stats()   # 获取统计信息
```

### 5.2 获取统计信息

```python
stats = adapter.get_performance_stats()
# 返回:
# {
#     "throttler": {
#         "step_count": 5,
#         "cooldown_hits": 2,
#     },
#     "caches": {
#         "embedding_cache_size": 100,
#         "retrieval_cache_size": 50,
#     },
#     "write_queue": {
#         "size": 3,
#         "enqueued": 10,
#         "flushed": 8,
#         "failed": 0,
#     }
# }
```

---

## 六、集成到执行循环

### 6.1 在 Agent/Executor 中集成

位置: `nanobot/agent/executor.py` 或类似的执行循环文件

```python
class AgentExecutor:
    def __init__(self, ...):
        # 初始化 MemoryAdapter（已在构造函数中完成）
        self.memory = MemoryAdapter(...)

    async def execute_step(self, step, ...):
        # === 1. 每轮开始时重置节流器 ===
        self.memory.reset_per_step()

        # === 2. 获取当前消息 ===
        current_message = step.message

        # === 3. 检索记忆（带触发判断） ===
        context = await self.memory.retrieve_context(
            channel=step.channel,
            chat_id=step.chat_id,
            sender_id=step.sender_id,
            history=step.history,
            current_message=current_message,
        )

        # === 4. 执行步骤 ===
        result = await self._run_step(step, context)

        # === 5. 写入记忆（带触发判断） ===
        await self.memory.memorize_turn(
            channel=step.channel,
            chat_id=step.chat_id,
            sender_id=step.sender_id,
            user_message=step.user_message,
            assistant_message=result.response,
        )

        # === 6. 定期刷新写入队列 ===
        if some_condition:
            await self.memory.flush_write_queue()

        return result
```

---

## 七、性能调优

### 7.1 性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 每轮延迟增加 | <100ms | 触发判断 + 缓存查找 |
| embedding 调用/轮 | 0-1 | 启用缓存后 |
| 检索调用/轮 | 0-1 | 启用节流后 |
| 写入调用/轮 | 0-1 | 启用队列后 |

### 7.2 调优建议

**低延迟场景**:
```python
config = MemUPerformanceConfig(
    enable_trigger_rules=True,
    embedding_cache_enabled=True,     # 必须启用
    embedding_cache_size=2000,        # 增大缓存
    embedding_cache_ttl=7200,        # 延长 TTL
    retrieve_throttle_per_step=1,    # 保持限制
    write_queue_enabled=True,         # 必须启用
    write_batch_size=3,               # 减少批量大小
    write_flush_interval=60,         # 延长刷新间隔
)
```

**高并发场景**:
```python
config = MemUPerformanceConfig(
    enable_trigger_rules=True,
    embedding_cache_enabled=True,
    embedding_cache_size=5000,        # 大缓存
    retrieve_throttle_per_step=1,
    retrieve_throttle_per_minute=3,   # 更严格限制
    retrieve_cooldown_seconds=30,     # 更长冷却
    write_queue_enabled=True,
    write_batch_size=10,             # 大批量
    write_flush_interval=120,        # 长间隔
)
```

**调试场景**（关闭所有优化）:
```python
config = MemUPerformanceConfig(
    enable_trigger_rules=False,       # 关闭触发
    embedding_cache_enabled=False,    # 关闭缓存
    write_queue_enabled=False,        # 关闭队列
)
```

---

## 八、扩展点

### 8.1 自定义触发函数

```python
def custom_should_memorize(message: str, conversation: list) -> TriggerResult:
    """自定义写入触发逻辑"""
    if "重要" in message:
        return TriggerResult(True, "自定义: 包含重要", priority=10)
    return TriggerResult(False, "自定义: 不触发")

config = MemUPerformanceConfig(
    should_memorize_fn=custom_should_memorize,
)
```

### 8.2 自定义缓存

位置: [memory_cache.py](nanobot/agent/memory_cache.py)

```python
class MyEmbeddingCache(EmbeddingCache):
    def __init__(self, ...):
        super().__init__(...)
        # 添加 Redis 或其他分布式缓存

# 然后在 MemoryAdapter 中替换
adapter._embedding_cache = MyEmbeddingCache(...)
```

### 8.3 自定义队列处理

位置: [memory_queue.py](nanobot/agent/memory_queue.py)

```python
class MyWriteQueue(MemoryWriteQueue):
    async def _flush(self):
        # 自定义批量处理逻辑
        # 例如: 批量调用 MemoryAgent.run()
        pass
```

---

## 九、常见问题

### Q1: 触发规则太严格，某些记忆没写入

A: 调整 `memory_decider.py` 中的触发词和正则表达式

### Q2: 检索被频繁节流

A: 调整 `retrieve_throttle_per_step` 和 `retrieve_cooldown_seconds`

### Q3: 写入队列堆积

A: 检查 `_flush_interval` 和 `batch_size`，或手动调用 `flush_write_queue()`

### Q4: 缓存命中率低

A: 检查 `embedding_cache_ttl` 是否太短，或 query 是否太短（短 query 难以命中）

---

## 十、修改记录

| 日期 | 修改内容 |
|------|----------|
| 2026-02-16 | 初始版本：创建 5 个组件并集成到 memory_adapter |

---

## 十一、相关文件链接

- [memory_adapter.py](nanobot/agent/memory_adapter.py) - 主适配器
- [memory_performance.py](nanobot/agent/memory_performance.py) - 配置类
- [memory_decider.py](nanobot/agent/memory_decider.py) - 触发决策器
- [memory_cache.py](nanobot/agent/memory_cache.py) - 缓存实现
- [memory_throttle.py](nanobot/agent/memory_throttle.py) - 节流器
- [memory_queue.py](nanobot/agent/memory_queue.py) - 队列实现
