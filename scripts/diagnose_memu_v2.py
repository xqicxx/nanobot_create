#!/usr/bin/env python3
"""
memU 系统完整诊断脚本 - 修复版
- 环境变量必须在导入前设置
- 使用正确的 API
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# 关键：环境变量必须在导入 memu 之前设置！
# ============================================================================

# 使用虚拟环境
VENV_PYTHON = "/root/nanobot-venv/bin/python"
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON, __file__] + sys.argv[1:])

sys.path.insert(0, "/root/nanoBot_memU/nanobot")

# 检查配置文件获取 API Keys
config_path = Path.home() / ".nanobot" / "config.json"
deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
silicon_key = os.environ.get("SILICONFLOW_API_KEY", "")
silicon_base_url = "https://api.siliconflow.cn/v1"
silicon_embed_model = "BAAI/bge-m3"

if config_path.exists():
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        memu = config.get("memu", {})

        default_cfg = memu.get("default", {})
        if not deepseek_key:
            deepseek_key = default_cfg.get("apiKey", "")

        embedding_cfg = memu.get("embedding", {})
        if not silicon_key:
            silicon_key = embedding_cfg.get("apiKey", "")
        silicon_base_url = embedding_cfg.get("baseUrl", "https://api.siliconflow.cn/v1")
        silicon_embed_model = embedding_cfg.get("embedModel", "BAAI/bge-m3")
    except Exception as e:
        print(f"读取配置失败: {e}")

# ============================================================================
# 关键修复：在导入 memu 之前设置环境变量
# ============================================================================
os.environ["OPENAI_API_KEY"] = silicon_key
os.environ["OPENAI_BASE_URL"] = silicon_base_url
os.environ["OPENAI_EMBED_MODEL"] = silicon_embed_model
os.environ["MEMU_EMBEDDING_MODEL"] = silicon_embed_model
os.environ["MEMU_LLM_API_KEY"] = silicon_key
os.environ["MEMU_LLM_BASE_URL"] = silicon_base_url
os.environ["MEMU_EMBEDDING_PROVIDER"] = "openai"

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(name, status, details=""):
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")

# ============================================================================
# 1. 基础连接检查
# ============================================================================
print_header("1. 基础环境检查")

print_test("配置文件存在", config_path.exists(), str(config_path))

workspace = Path.home() / ".nanobot" / "workspace"
print_test("工作空间存在", workspace.exists(), str(workspace))

print_test("DeepSeek API Key", bool(deepseek_key and len(deepseek_key) > 20))
print_test("SiliconFlow API Key", bool(silicon_key and len(silicon_key) > 20))
print_test("Embedding Model", True, silicon_embed_model)

# ============================================================================
# 2. memU 模块导入
# ============================================================================
print_header("2. memU 模块导入")

try:
    from memu.memory import MemoryAgent
    print_test("MemoryAgent 导入", True)
except Exception as e:
    print_test("MemoryAgent 导入", False, str(e))
    sys.exit(1)

try:
    from memu.llm import DeepSeekClient
    print_test("DeepSeekClient 导入", True)
except Exception as e:
    print_test("DeepSeekClient 导入", False, str(e))

# ============================================================================
# 3. 初始化 MemoryAgent
# ============================================================================
print_header("3. 初始化 MemoryAgent")

try:
    llm_client = DeepSeekClient(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    )
    print_test("LLM Client 创建", True)

    # 关键：检查环境变量是否正确传递
    print(f"   环境变量检查:")
    print(f"     OPENAI_EMBED_MODEL: {os.environ.get('OPENAI_EMBED_MODEL', 'NOT SET')}")
    print(f"     MEMU_EMBEDDING_MODEL: {os.environ.get('MEMU_EMBEDDING_MODEL', 'NOT SET')}")

    agent = MemoryAgent(
        llm_client=llm_client,
        memory_dir=str(workspace / ".memu" / "memory"),
        enable_embeddings=True,
        agent_id="diagnose",
        user_id="default",
    )
    print_test("MemoryAgent 创建", True)

    status = agent.get_status()
    print_test("Agent 状态获取", True, json.dumps(status, ensure_ascii=False)[:100])

except Exception as e:
    print_test("MemoryAgent 初始化", False, str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. 功能测试
# ============================================================================

async def run_tests():
    print_header("4. 功能测试")

    # ------------------------------------------------------------------
    # 4.1 memorize() - 写入记忆
    # ------------------------------------------------------------------
    print("\n## 4.1 memorize() - 写入记忆")
    test_memory = f"测试记忆 - 健身爱好 - {datetime.now().strftime('%H:%M:%S')}"

    try:
        conversation = [
            {"role": "user", "content": "我喜欢健身"},
            {"role": "assistant", "content": "很好，健身有助于健康"},
        ]
        result = agent.run(
            conversation=conversation,
            character_name="diagnose_test",
            max_iterations=3,
        )

        success = result.get("success", False)
        function_calls = result.get("function_calls", [])
        print_test("memorize() 执行", success, f"success={success}, function_calls={len(function_calls)}")

    except Exception as e:
        print_test("memorize() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.2 检索 API 探索
    # ------------------------------------------------------------------
    print("\n## 4.2 检索 API 探索")

    # 检查 MemoryAgent 有哪些方法
    print("   MemoryAgent 可用方法:")
    for attr in dir(agent):
        if not attr.startswith('_') and callable(getattr(agent, attr)):
            print(f"     - {attr}")

    # 检查 core 或其他属性
    print("\n   MemoryAgent 属性:")
    for attr in ['core', 'memory_core', 'storage_manager', 'embeddings']:
        if hasattr(agent, attr):
            print(f"     - {attr}: {type(getattr(agent, attr))}")

    # ------------------------------------------------------------------
    # 4.3 尝试使用正确的检索 API
    # ------------------------------------------------------------------
    print("\n## 4.3 语义检索测试")

    # 方法1: 尝试通过 core 访问
    try:
        if hasattr(agent, 'core') and hasattr(agent.core, 'retrieve'):
            print("   通过 core.retrieve() 检索")
            result = await agent.core.retrieve(
                query="健身",
                user_id="diagnose_test",
            )
            print_test("core.retrieve()", True, str(result)[:100])
    except Exception as e:
        print(f"   core.retrieve 失败: {e}")

    # 方法2: 尝试使用 memory_core
    try:
        if hasattr(agent, 'memory_core'):
            mc = agent.memory_core
            print(f"   memory_core 类型: {type(mc)}")
            if hasattr(mc, 'retrieve'):
                result = await mc.retrieve(
                    query="健身",
                    user_id="diagnose_test",
                )
                print_test("memory_core.retrieve()", True, str(result)[:100])
    except Exception as e:
        print(f"   memory_core.retrieve 失败: {e}")

    # ------------------------------------------------------------------
    # 4.4 文件系统检索 (降级方案)
    # ------------------------------------------------------------------
    print("\n## 4.4 文件系统检索 (降级)")

    try:
        memory_dir = workspace / ".memu" / "memory"
        # 查找诊断测试的记忆
        user_dir = memory_dir / "nanobot" / "diagnose_test"
        if user_dir.exists():
            md_files = list(user_dir.glob("*.md"))
            print_test("找到记忆文件", True, f"共 {len(md_files)} 个")
            for f in md_files[:3]:
                content = f.read_text(encoding="utf-8")[:200]
                print(f"     - {f.name}: {content[:50]}...")
        else:
            print_test("找到记忆文件", False, "目录不存在")
    except Exception as e:
        print_test("文件系统检索", False, str(e))

    # ------------------------------------------------------------------
    # 4.5 数据库检查
    # ------------------------------------------------------------------
    print("\n## 4.5 数据库检查")

    try:
        memory_dir = workspace / ".memu" / "memory"
        md_files = list(memory_dir.rglob("*.md"))
        print_test("记忆文件数量", True, f"共 {len(md_files)} 个 .md 文件")

        for f in md_files[:10]:
            size = f.stat().st_size
            print(f"   - {f.name}: {size} bytes")

    except Exception as e:
        print_test("数据库文件检查", False, str(e))

    # ============================================================================
    # 5. 总结
    # ============================================================================
    print_header("诊断总结")
    print("\n请检查上方测试结果，确保所有功能正常运行。")

if __name__ == "__main__":
    asyncio.run(run_tests())
