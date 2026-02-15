#!/usr/bin/env python3
"""Debug MemoryAgent initialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "nanoBot_memU" / "nanobot"))

print("="*60)
print("🔍 MemoryAgent 初始化诊断")
print("="*60)

# 1. 检查 memu-py 是否安装
print("\n1. 检查 memu-py 安装...")
try:
    from memu.memory import MemoryAgent
    from memu.llm import OpenAIClient, DeepSeekClient
    print("✓ memu-py 已安装")
except Exception as e:
    print(f"✗ memu-py 未安装: {e}")
    sys.exit(1)

# 2. 检查配置
print("\n2. 检查配置...")
from nanobot.config.loader import load_config
config = load_config()
print(f"   MemU enabled: {config.memu.enabled}")
print(f"   DeepSeek API Key: {config.memu.default.api_key[:15]}..." if config.memu.default.api_key else "   DeepSeek API Key: 未设置")

# 3. 创建 LLM client
print("\n3. 创建 LLM client...")
try:
    default_cfg = config.memu.default
    provider = getattr(default_cfg, "provider", "openai")
    api_key = getattr(default_cfg, "api_key", "")
    base_url = getattr(default_cfg, "base_url", "") or "https://api.deepseek.com/v1"
    chat_model = getattr(default_cfg, "chat_model", "deepseek-chat")
    
    print(f"   Provider: {provider}")
    print(f"   Base URL: {base_url}")
    print(f"   API Key length: {len(api_key)}")
    print(f"   Chat Model: {chat_model}")
    
    if provider == "deepseek" or "deepseek" in base_url.lower():
        client = DeepSeekClient(
            api_key=api_key,
            base_url=base_url,
            model_name=chat_model,
        )
        print("✓ DeepSeek client 创建成功")
    else:
        client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=chat_model,
        )
        print("✓ OpenAI client 创建成功")
except Exception as e:
    print(f"✗ LLM client 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 创建 MemoryAgent
print("\n4. 创建 MemoryAgent...")
try:
    memory_dir = str(config.workspace_path / ".memu" / "memory")
    agent = MemoryAgent(
        llm_client=client,
        memory_dir=memory_dir,
        enable_embeddings=True,
        agent_id="nanobot",
        user_id="default",
    )
    print("✓ MemoryAgent 创建成功")
    
    # 5. 测试 get_status
    print("\n5. 测试 get_status...")
    status = agent.get_status()
    print(f"   Status: {status}")
    
except Exception as e:
    print(f"✗ MemoryAgent 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有组件初始化成功!")
print("="*60)
