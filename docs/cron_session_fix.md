# nanobot Cron 模块 Session Context 修复方案

> 分析 nanobot 仓库中 cron 模块的 session context 丢失问题并提供修复方案

---

## 一、当前 Cron 架构分析

### 1.1 文件路径

| 文件 | 作用 |
|------|------|
| `nanobot/cron/types.py` | Cron 数据类型定义 |
| `nanobot/cron/service.py` | CronService 调度服务 |
| `nanobot/agent/tools/cron.py` | Cron 工具（Agent 调用） |
| `nanobot/cli/commands.py:421-437` | Cron 任务执行回调 `on_cron_job` |
| `nanobot/agent/loop.py:2466-2502` | `process_direct()` 函数 |

### 1.2 执行流程

```
用户创建 Cron Job
       ↓
CronService 调度
       ↓
定时触发 → on_cron_job(job)
       ↓
agent.process_direct(
    message=job.payload.message,
    session_key=f"cron:{job.id}",  ← 问题点1
    channel=job.payload.channel,
    chat_id=job.payload.to,
)
       ↓
_process_message_impl()
       ↓
session = sessions.get_or_create(session_key)  ← 新session，无历史
       ↓
memory_adapter.retrieve_context()  ← 可以工作（channel/chat_id正确）
```

### 1.3 Reminder vs Task 区别

| 模式 | deliver | 说明 |
|------|---------|------|
| **Reminder** | `True` | 执行任务并将结果发送给用户 |
| **Task** | `False` | 静默执行，不发送结果给用户 |

两者在 context 恢复层面没有区别。

---

## 二、Session Context 丢失原因

### 问题 1: 独立 Session Key

**位置**: `commands.py:426`

```python
session_key=f"cron:{job.id}",  # 例如: cron:ab523a2
```

**问题**: 每个 cron job 创建全新的 session，无法继承用户历史会话。

**影响**:
- `session.get_history()` 返回空列表
- 无法获取之前的对话上下文
- 每次 cron 执行都是"第一次"对话

### 问题 2: 未绑定原始用户 Context

**位置**: `commands.py:424-429`

```python
response = await agent.process_direct(
    job.payload.message,
    session_key=f"cron:{job.id}",
    channel=job.payload.channel or "cli",
    chat_id=job.payload.to or "direct",
)
```

**问题**: 没有传递以下信息：
- 原始用户的 session_key（如 `whatsapp:+1234567890`）
- 用户的历史会话记录
- 用户的 memory context（虽然 memory 可以通过 channel/chat_id 获取）

### 问题 3: 缺少原始会话历史

**代码追踪**:

1. `loop.py:2071`: `session = self.sessions.get_or_create(msg.session_key)`
2. `loop.py:2072`: `session_model = self._get_session_model(session)`
3. `loop.py:2124`: `history=session.get_history()` - 空！

**结果**:
- LLM 无法看到之前的对话
- 无法理解用户的长期偏好
- 每个 cron 执行都是独立上下文

---

## 三、修复方案设计

### 3.1 核心思路

**目标**: 让 cron Task 执行时能够恢复用户的完整上下文。

```
┌─────────────────────────────────────────────────────────────┐
│                     当前流程 (有问题)                         │
├─────────────────────────────────────────────────────────────┤
│  cron:ab523a2 (新session)                                  │
│       ↓                                                    │
│  session.get_history() = [] ❌                             │
│       ↓                                                    │
│  无历史 → 无上下文                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     修复后流程                                │
├─────────────────────────────────────────────────────────────┤
│  cron:ab523a2 (新session)                                   │
│       ↓                                                    │
│  加载原始用户 session: whatsapp:+1234567890                 │
│       ↓                                                    │
│  session.get_history() = [用户历史...] ✅                   │
│       ↓                                                    │
│  合并 context → 正常执行                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 需要保存的数据

在创建 cron job 时，需要保存：

```python
@dataclass
class CronJob:
    id: str
    payload: CronPayload
    schedule: CronSchedule

@dataclass
class CronPayload:
    message: str
    deliver: bool
    channel: str | None
    to: str | None
    # === 新增字段 ===
    original_session_key: str | None = None  # 原始用户 session_key
    user_id: str | None = None                # 用于 memory 检索
```

### 3.3 修复方案实现

#### 方案 A: 在 CronPayload 中保存原始 session（推荐）

**Step 1: 修改 types.py - 添加字段**

```python
# nanobot/cron/types.py

@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    deliver: bool = False
    channel: str | None = None
    to: str | None = None

    # === 新增：用于恢复 context ===
    original_session_key: str | None = None  # 例如: "whatsapp:+1234567890"
    user_id: str | None = None                # 例如: "whatsapp:+1234567890:user"
```

**Step 2: 修改 commands.py - 传递原始 session**

```python
# nanobot/cli/commands.py

async def on_cron_job(job: CronJob) -> str | None:
    """Execute a cron job through the agent."""

    # === 修改：使用原始 session_key ===
    session_key = job.payload.original_session_key or f"cron:{job.id}"

    response = await agent.process_direct(
        job.payload.message,
        session_key=session_key,  # 使用原始用户 session
        channel=job.payload.channel or "cli",
        chat_id=job.payload.to or "direct",
    )

    if job.payload.deliver and job.payload.to:
        from nanobot.bus.events import OutboundMessage
        await bus.publish_outbound(OutboundMessage(
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to,
            content=response or ""
        ))
    return response
```

**Step 3: 修改 cron.py 工具 - 保存原始 session**

```python
# nanobot/agent/tools/cron.py

class CronTool:
    # ... existing code ...

    async def execute(self, **kwargs) -> str:
        # ... existing validation ...

        # === 新增：传递原始 session 信息 ===
        original_session_key = kwargs.get("original_session_key")
        user_id = kwargs.get("user_id")

        # 构建 payload
        payload = CronPayload(
            message=message,
            deliver=deliver,
            channel=self._channel,
            to=self._chat_id,
            # === 新增字段 ===
            original_session_key=original_session_key or f"{self._channel}:{self._chat_id}",
            user_id=user_id or f"{self._channel}:{self._chat_id}:user",
        )
        # ... rest of code
```

**Step 4: 更新工具描述**

```python
# nanobot/agent/tools/cron.py

@property
def parameters(self) -> dict:
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "..."},
            "deliver": {"type": "boolean", "description": "..."},
            "schedule": {"type": "object", "description": "..."},
            # === 新增参数 ===
            "original_session_key": {
                "type": "string",
                "description": "Original user session key (e.g., 'whatsapp:+1234567890'). If not provided, will use channel:chat_id."
            },
            "user_id": {
                "type": "string",
                "description": "User ID for memory retrieval (e.g., 'whatsapp:+1234567890:user')."
            },
        },
        "required": ["message", "schedule"],
    }
```

---

## 四、Memory Context 修复

### 4.1 当前状态

Memory 实际上**可以工作**，因为：
- `retrieve_context()` 使用 `channel` + `chat_id` + `sender_id` 构建 user_id
- cron job 传递了 `channel` 和 `to`（作为 chat_id）
- 所以 memory 检索路径: `whatsapp:+1234567890:user` ✅

### 4.2 但存在一个问题

在 `process_direct` 中，`sender_id` 被硬编码为 `"user"`:

```python
# loop.py:2486-2491
msg = InboundMessage(
    channel=channel,
    sender_id="user",  # ← 硬编码
    chat_id=chat_id,
    content=content
)
```

**修复**: 在调用 `process_direct` 时传入正确的 sender_id：

```python
# commands.py
sender_id = None
if job.payload.user_id:
    # 从 user_id 提取 sender_id (格式: channel:chat_id:sender_id)
    parts = job.payload.user_id.split(":")
    if len(parts) >= 3:
        sender_id = parts[2]

response = await agent.process_direct(
    job.payload.message,
    session_key=session_key,
    channel=job.payload.channel or "cli",
    chat_id=job.payload.to or "direct",
    sender_id=sender_id,  # ← 新增参数（如果 process_direct 支持）
)
```

**注意**: 需要先检查 `process_direct` 是否支持 `sender_id` 参数。

---

## 五、推荐最终 Cron 架构

### 5.1 修复后数据流

```
用户调用 CronTool
       ↓
保存 job 时携带:
  - original_session_key: "whatsapp:+1234567890"
  - user_id: "whatsapp:+1234567890:user"
       ↓
CronService 定时触发
       ↓
on_cron_job(job)
       ↓
使用 original_session_key 加载用户历史
       ↓
agent.process_direct(
    session_key="whatsapp:+1234567890",  ← 用户原始 session
    channel="whatsapp",
    chat_id="+1234567890",
)
       ↓
session.get_history() = [用户历史...] ✅
       ↓
memory_adapter.retrieve_context() → 正确用户 memory ✅
       ↓
正常执行，返回结果
```

### 5.2 保持 Reminder 轻量

对于 **Reminder 模式**（不需要完整 agent 上下文）：
- 不需要加载历史（纯消息推送）
- 不需要 memory 检索
- 保持原有轻量逻辑

可以通过配置区分：

```python
# 在 CronPayload 中添加
lightweight: bool = False  # True = 不加载历史和 memory

# 在 on_cron_job 中
if job.payload.lightweight:
    # 快速发送消息，不启动完整 agent
    await quick_reminder(job)
else:
    # 完整 agent 流程（加载历史 + memory）
    await agent.process_direct(...)
```

---

## 六、最小可落地修复版本（MVP）

### 6.1 修改清单

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `nanobot/cron/types.py` | CronPayload 添加 `original_session_key`, `user_id` | ~10 |
| `nanobot/agent/tools/cron.py` | 工具参数添加新字段，传递原始 session | ~20 |
| `nanobot/cli/commands.py` | on_cron_job 使用 original_session_key | ~5 |

### 6.2 实施步骤

**Step 1: 修改 types.py**
```python
# 添加 2 个可选字段到 CronPayload
original_session_key: str | None = None
user_id: str | None = None
```

**Step 2: 修改 cron.py 工具**
- 更新 parameters 定义
- 在 execute() 中传递新字段到 payload

**Step 3: 修改 commands.py**
- on_cron_job 使用 `job.payload.original_session_key or f"cron:{job.id}"`

### 6.3 不需要修改的部分

- ✅ `CronService` - 调度逻辑无需改动
- ✅ `memory_adapter` - 已经通过 channel/chat_id 正确工作
- ✅ `SessionManager` - session 加载机制无需改动
- ✅ Reminder 模式 - 保持现有逻辑

---

## 七、测试验证

### 7.1 测试用例

```python
# 测试场景 1: Task 模式恢复上下文
1. 用户正常对话: "我喜欢苹果"
2. 创建 cron: "每天提醒我吃苹果"
3. cron 执行时检查:
   - session.get_history() 应包含 "我喜欢苹果"
   - memory 应检索到用户偏好

# 测试场景 2: Reminder 模式保持轻量
1. 创建 cron (lightweight=True): "每天早上9点提醒我开会"
2. cron 执行时检查:
   - 不加载历史
   - 不检索 memory
   - 直接发送消息
```

---

## 八、总结

| 问题 | 原因 | 修复方案 |
|------|------|----------|
| Session 历史丢失 | cron 使用独立 session_key | 保存 original_session_key 并使用 |
| 无用户上下文 | 每次都是新 session | 加载用户历史到 session |
| Memory 可能不正确 | sender_id 硬编码 | 传递正确的 user_id |

**核心修复**: 在创建 cron job 时保存 `original_session_key`，执行时使用它而非生成新的 session key。

---

## 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| CronPayload 定义 | `nanobot/cron/types.py` | ~30 |
| CronTool 工具 | `nanobot/agent/tools/cron.py` | ~1 |
| on_cron_job 回调 | `nanobot/cli/commands.py` | ~421 |
| process_direct | `nanobot/agent/loop.py` | ~2466 |
| session.get_history | `nanobot/session/manager.py` | ~39 |
| retrieve_context | `nanobot/agent/memory_adapter.py` | ~207 |
