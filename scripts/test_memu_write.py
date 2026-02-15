#!/usr/bin/env python3
"""Test MemU automatic memory writing."""

import asyncio
import sys
from pathlib import Path

# Add nanobot to path
sys.path.insert(0, str(Path.home() / "nanoBot_memU" / "nanobot"))

from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter


async def test_memory_write():
    """Test that MemU automatically writes memories."""
    print("=" * 60)
    print("🧪 Testing MemU Automatic Memory Writing")
    print("=" * 60)
    
    # Load config
    config = load_config()
    print(f"\n✅ Config loaded")
    print(f"   MemU enabled: {config.memu.enabled}")
    
    # Create memory adapter
    adapter = MemoryAdapter(
        workspace=config.workspace_path,
        memu_config=config.memu,
    )
    
    if not adapter.enable_memory:
        print("\n❌ MemU is not enabled!")
        return False
    
    print(f"✅ MemoryAdapter created")
    print(f"   MemoryAgent initialized: {adapter._memory_agent is not None}")
    
    # Test memorize_turn
    print("\n📝 Testing memorize_turn...")
    print("   Saving: 'User: 你好，我叫测试用户'")
    print("   Response: 'Assistant: 你好测试用户！很高兴认识你。'")
    
    await adapter.memorize_turn(
        channel="test",
        chat_id="test-session",
        sender_id="user",
        user_message="你好，我叫测试用户",
        assistant_message="你好测试用户！很高兴认识你。",
        metadata={"test": True}
    )
    
    print("✅ memorize_turn completed")
    
    # Check if memory files were created
    print("\n🔍 Checking memory files...")
    memory_dir = config.workspace_path / ".memu" / "memory"
    
    if memory_dir.exists():
        print(f"   Memory directory: {memory_dir}")
        
        # Walk through all subdirectories
        import os
        found_files = []
        for root, dirs, files in os.walk(memory_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    found_files.append(file_path)
        
        if found_files:
            print(f"\n✅ Found {len(found_files)} memory files:")
            for file_path in found_files:
                print(f"\n   📄 {file_path.relative_to(memory_dir)}")
                try:
                    content = file_path.read_text(encoding="utf-8")
                    # Show first 200 chars
                    preview = content[:200].replace('\n', ' ')
                    print(f"      Content: {preview}...")
                except Exception as e:
                    print(f"      Error reading: {e}")
            
            print("\n" + "=" * 60)
            print("✅ SUCCESS! MemU is automatically writing memories!")
            print("=" * 60)
            return True
        else:
            print("\n⚠️  No memory files found yet.")
            print("   This might mean:")
            print("   1. MemoryAgent hasn't created files yet (check logs)")
            print("   2. The conversation wasn't processed by MemoryAgent")
            
            # List directory structure
            print(f"\n   Directory structure:")
            for root, dirs, files in os.walk(memory_dir):
                level = root.replace(str(memory_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"   {indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"   {subindent}{file}")
            
            return False
    else:
        print(f"\n❌ Memory directory doesn't exist: {memory_dir}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_memory_write())
    sys.exit(0 if result else 1)
