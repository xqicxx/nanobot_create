# Nanobot + MemU Integration

This repository has been modified to remove Nanobot's file-based memory and
replace it with MemU (long-term, structured memory).

## Summary

- File-based memory (`memory/`, `MEMORY.md`, daily notes) is removed.
- MemU is embedded as a Python package.
- Memory is retrieved before each prompt and written after each turn.
- User isolation uses:
  - `user_id = f"{channel}:{chat_id}:{sender_id}"`

## Requirements

- Python 3.13+
- MemU package available (`memu-py`)
- LLM and embedding API keys configured

## Install

```bash
pip install -e .
pip install memu-py
```

## Environment Variables (example)

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_CHAT_MODEL="deepseek-chat"

export SILICONFLOW_API_KEY="..."
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
export SILICONFLOW_EMBED_MODEL="BAAI/bge-m3"

# Optional: override MemU DB dsn (defaults to sqlite at workspace/.memu/memu.db)
# export MEMU_DB_DSN="sqlite:////absolute/path/to/memu.db"
```

## How Memory Works

- **Retrieve**: before each LLM call, MemU retrieves relevant memory and injects
  it into the system prompt.
- **Write**: after each turn, MemU stores the user + assistant exchange as a
  conversation resource.

### Write Filter (skip if any)

- Message is empty after trimming
- Pure emoji or punctuation
- Only greeting/ack (e.g., 好/嗯/OK/收到/谢谢/哈哈)
- Same as previous user message

## Files Changed

- `nanobot/agent/memory_adapter.py` (new MemU adapter)
- `nanobot/agent/context.py` (memory injection now uses MemU context)
- `nanobot/agent/loop.py` (retrieve + write)
- `nanobot/agent/__init__.py` (export MemoryAdapter)
- `nanobot/cli/commands.py` (removed memory file init)
- `workspace/AGENTS.md` (updated instructions)
- `nanobot/agent/memory.py` (removed)

## Troubleshooting

- If MemU is not installed, Nanobot will fail to start.
- If MemU is slow/unavailable, retrieval/write failures are logged and the
  agent continues without memory.
