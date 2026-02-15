#!/usr/bin/env python3
"""MemU configuration diagnostic tool for nanobot."""

import json
import os
from pathlib import Path


def check_config():
    """Check nanobot configuration."""
    config_path = Path.home() / ".nanobot" / "config.json"
    
    print("=" * 60)
    print("🔍 Nanobot MemU Configuration Diagnostic")
    print("=" * 60)
    
    # Check if config file exists
    if not config_path.exists():
        print(f"\n❌ Config file not found: {config_path}")
        print("\n💡 Please create config file:")
        print("   cp ~/nanoBot_memU/nanobot/config.example.json ~/.nanobot/config.json")
        return False
    
    print(f"\n✅ Config file found: {config_path}")
    
    # Load config
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"\n❌ Failed to parse config: {e}")
        return False
    
    # Check memu configuration
    memu = config.get("memu", {})
    
    print("\n" + "-" * 60)
    print("📋 MemU Configuration")
    print("-" * 60)
    
    enabled = memu.get("enabled", False)
    print(f"  Enabled: {'✅ Yes' if enabled else '❌ No'}")
    
    if not enabled:
        print("\n⚠️  MemU is disabled. Set 'enabled: true' to enable memory.")
        return False
    
    # Check default LLM config
    default = memu.get("default", {})
    print("\n  📝 Default LLM (DeepSeek):")
    api_key = default.get("apiKey", "")
    base_url = default.get("baseUrl", "")
    chat_model = default.get("chatModel", "")
    
    print(f"    Provider: {default.get('provider', 'not set')}")
    print(f"    Base URL: {base_url or '❌ not set'}")
    print(f"    API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"    Chat Model: {chat_model or 'not set'}")
    
    # Check embedding config
    embedding = memu.get("embedding", {})
    print("\n  🔍 Embedding (SiliconFlow):")
    embed_api_key = embedding.get("apiKey", "")
    embed_base_url = embedding.get("baseUrl", "")
    embed_model = embedding.get("embedModel", "")
    
    print(f"    Provider: {embedding.get('provider', 'not set')}")
    print(f"    Base URL: {embed_base_url or '❌ not set'}")
    print(f"    API Key: {'✅ Set' if embed_api_key else '❌ Not set'}")
    print(f"    Embed Model: {embed_model or 'not set'}")
    
    # Check providers configuration
    providers = config.get("providers", {})
    print("\n" + "-" * 60)
    print("🤖 LLM Providers")
    print("-" * 60)
    
    for name, provider in providers.items():
        api_key = provider.get("apiKey", "")
        print(f"  {name}: {'✅ API Key Set' if api_key else '❌ API Key Missing'}")
    
    # Test memu-py import
    print("\n" + "-" * 60)
    print("🔧 Testing MemU Import")
    print("-" * 60)
    
    try:
        from memu.memory import MemoryAgent
        print("  ✅ memu-py imported successfully")
        print(f"  📦 Version: {getattr(MemoryAgent, '__module__', 'unknown')}")
    except Exception as e:
        print(f"  ❌ Failed to import memu-py: {e}")
        return False
    
    # Test initialization
    print("\n" + "-" * 60)
    print("🚀 Testing MemU Initialization")
    print("-" * 60)
    
    try:
        import sys
        sys.path.insert(0, str(Path.home() / "nanoBot_memU" / "nanobot"))
        
        from nanobot.config.loader import load_config
        from nanobot.agent.memory_adapter import MemoryAdapter
        
        print("  ✅ Config loader imported")
        print("  ✅ MemoryAdapter imported")
        
        config = load_config()
        print(f"  ✅ Config loaded (MemU enabled: {config.memu.enabled})")
        
        # Try to initialize MemoryAdapter
        adapter = MemoryAdapter(
            workspace=config.workspace_path,
            memu_config=config.memu,
        )
        
        if adapter.enable_memory and adapter._memory_agent:
            print("  ✅ MemU initialized successfully")
            status = adapter._memory_agent.get_status()
            print(f"  📊 MemoryAgent Status: {status.get('agent_name', 'unknown')}")
            print(f"  🔢 Actions Available: {status.get('total_actions', 0)}")
            print(f"  🔍 Embeddings Enabled: {status.get('embedding_capabilities', {}).get('embeddings_enabled', False)}")
            return True
        else:
            print("  ⚠️  MemU disabled or failed to initialize")
            print("     Check logs above for error details")
            return False
            
    except Exception as e:
        print(f"  ❌ MemU initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_operations():
    """Test basic memory operations."""
    print("\n" + "=" * 60)
    print("🧪 Testing Memory Operations")
    print("=" * 60)
    
    try:
        import asyncio
        from nanobot.config.loader import load_config
        from nanobot.agent.memory_adapter import MemoryAdapter
        
        config = load_config()
        adapter = MemoryAdapter(
            workspace=config.workspace_path,
            memu_config=config.memu,
        )
        
        if not adapter.enable_memory:
            print("\n❌ MemU not enabled, skipping tests")
            return False
        
        # Test memorize
        print("\n  📝 Testing memorize...")
        asyncio.run(adapter.memorize_turn(
            channel="test",
            chat_id="test-session",
            sender_id="user",
            user_message="你好，我叫测试用户",
            assistant_message="你好测试用户！很高兴认识你。",
        ))
        print("  ✅ Memorize test passed")
        
        # Test status
        print("\n  📊 Checking status...")
        status = asyncio.run(adapter.memu_status())
        print(f"  ✅ Status: {status}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config_ok = check_config()
    
    if config_ok:
        print("\n" + "=" * 60)
        print("🎉 Configuration looks good!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   1. Restart nanobot: sudo systemctl restart nanobot-agent@root")
        print("   2. Test with: nanobot agent -m '你好'")
        
        # Ask if user wants to test memory operations
        response = input("\n🧪 Run memory operation tests? (y/n): ")
        if response.lower() == 'y':
            test_memory_operations()
    else:
        print("\n" + "=" * 60)
        print("⚠️  Configuration has issues")
        print("=" * 60)
        print("\n💡 Please fix the issues above and run again.")
