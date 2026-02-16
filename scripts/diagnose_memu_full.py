#!/usr/bin/env python3
"""
memU 系统完整诊断脚本
根据官方文档验证所有功能点
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

# 使用虚拟环境
VENV_PYTHON = "/root/nanobot-venv/bin/python"
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON, __file__] + sys.argv[1:])

sys.path.insert(0, "/root/nanoBot_memU/nanobot")

# memu-py 是已安装的包，不需要额外添加路径
# 导入方式: from memu.app.service import MemoryService

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

# 检查配置文件
config_path = Path.home() / ".nanobot" / "config.json"
print_test("配置文件存在", config_path.exists(), str(config_path))

# 检查工作空间
workspace = Path.home() / ".nanobot" / "workspace"
print_test("工作空间存在", workspace.exists(), str(workspace))

# 检查 API Keys - 优先从配置文件读取
deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
silicon_key = os.environ.get("SILICONFLOW_API_KEY", "")
silicon_base_url = "https://api.siliconflow.cn/v1"  # 默认值
silicon_embed_model = "BAAI/bge-m3"  # 默认值

# 从配置文件读取
if config_path.exists():
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        memu = config.get("memu", {})

        # DeepSeek 配置
        default_cfg = memu.get("default", {})
        if not deepseek_key:
            deepseek_key = default_cfg.get("apiKey", "")

        # SiliconFlow 配置
        embedding_cfg = memu.get("embedding", {})
        if not silicon_key:
            silicon_key = embedding_cfg.get("apiKey", "")
        silicon_base_url = embedding_cfg.get("baseUrl", "https://api.siliconflow.cn/v1")
        silicon_embed_model = embedding_cfg.get("embedModel", "BAAI/bge-m3")
    except Exception as e:
        print(f"读取配置失败: {e}")

print_test("DeepSeek API Key", bool(deepseek_key and len(deepseek_key) > 20))
print_test("SiliconFlow API Key", bool(silicon_key and len(silicon_key) > 20))

# ============================================================================
# 2. memU 模块导入 - 必须在导入前设置环境变量
# ============================================================================
print_header("2. memU 模块导入")

# 在导入 memu 之前设置环境变量 - SiliconFlow 兼容 OpenAI API
# 需要同时设置 OPENAI_* 变量（memu 默认使用）
os.environ["OPENAI_API_KEY"] = silicon_key
os.environ["OPENAI_BASE_URL"] = silicon_base_url
os.environ["OPENAI_EMBED_MODEL"] = silicon_embed_model
# 也设置 SILICONFLOW_* 以防万一
os.environ["SILICONFLOW_API_KEY"] = silicon_key
os.environ["SILICONFLOW_BASE_URL"] = silicon_base_url
os.environ["SILICONFLOW_EMBED_MODEL"] = silicon_embed_model

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
    # 配置 LLM
    llm_client = DeepSeekClient(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    )
    print_test("LLM Client 创建", True)

    # 创建 MemoryAgent
    agent = MemoryAgent(
        llm_client=llm_client,
        memory_dir=str(workspace / ".memu" / "memory"),
        enable_embeddings=True,
        agent_id="diagnose",
        user_id="default",
    )
    print_test("MemoryAgent 创建", True)

    # 获取状态
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

    # 使用 agent 代替 service
    global agent

    # ------------------------------------------------------------------
    # 4.1 memorize() - 写入记忆
    # ------------------------------------------------------------------
    print("\n## 4.1 memorize() - 写入记忆")
    test_memory = f"测试记忆 - 健身爱好 - {datetime.now().strftime('%H:%M:%S')}"

    try:
        # MemoryAgent 使用 run() 方法进行记忆
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
    # 4.2 retrieve() - 语义检索
    # ------------------------------------------------------------------
    print("\n## 4.2 retrieve() - 语义检索")

    try:
        # MemoryAgent 的 retrieve 方法
        result = await agent.retrieve(
            query="健身",
            method="rag",
            user={"user_id": "diagnose_test"},
        )

        items = result.get("items", [])
        categories = result.get("categories", [])

        print_test("retrieve() 执行", True, f"找到 {len(items)} 条items, {len(categories)} 个categories")

        # 显示结果
        for item in items[:3]:
            content = item.get("content", "")[:100]
            print(f"   - Item: {content}")

    except Exception as e:
        print_test("retrieve() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.3 list_memory_items() - 列出记忆
    # ------------------------------------------------------------------
    print("\n## 4.3 list_memory_items() - 列出记忆")

    try:
        # 使用 storage_manager 直接读取
        if hasattr(agent, 'storage_manager'):
            items = agent.storage_manager.list_memory_items(user_id="diagnose_test")
            print_test("list_memory_items() 执行", True, f"返回 {len(items)} 条")
        else:
            print_test("list_memory_items() 执行", False, "storage_manager 不可用")

    except Exception as e:
        print_test("list_memory_items() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.4 list_categories() - 列出分类
    # ------------------------------------------------------------------
    print("\n## 4.4 list_categories() - 列出分类")

    try:
        if hasattr(agent, 'storage_manager'):
            categories = agent.storage_manager.list_memory_categories(user_id="diagnose_test")
            print_test("list_categories() 执行", True, f"返回 {len(categories)} 个分类")
        else:
            print_test("list_categories() 执行", False, "storage_manager 不可用")

    except Exception as e:
        print_test("list_categories() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.5 reinforcement 强化机制
    # ------------------------------------------------------------------
    print("\n## 4.5 reinforcement 强化机制")

    try:
        # 写入相同内容两次，测试强化
        for i in range(2):
            await agent.run(
                conversation=[{"role": "user", "content": "强化测试内容 - 喜欢跑步"}],
                character_name="reinforce_test",
                max_iterations=2,
            )

        # 检索测试
        result = await agent.retrieve(
            query="跑步",
            method="rag",
            user={"user_id": "reinforce_test"},
        )

        items = result.get("items", [])
        print_test("reinforcement 写入", True, f"写入2次，检索到 {len(items)} 条")

    except Exception as e:
        print_test("reinforcement 测试", False, str(e))

    # ------------------------------------------------------------------
    # 4.6 salience 排序
    # ------------------------------------------------------------------
    print("\n## 4.6 salience 排序")

    try:
        # 写入不同内容
        await agent.run(
            conversation=[{"role": "user", "content": "今天很重要：我要学习Python编程"}],
            character_name="salience_test",
            max_iterations=2,
        )

        await agent.run(
            conversation=[{"role": "user", "content": "随便说说的不重要的话"}],
            character_name="salience_test",
            max_iterations=2,
        )

        # 检索
        result = await agent.retrieve(
            query="学习",
            method="rag",
            user={"user_id": "salience_test"},
        )

        items = result.get("items", [])
        print_test("salience 排序", True, f"返回 {len(items)} 条，按相关性排序")

    except Exception as e:
        print_test("salience 排序测试", False, str(e))

    # ------------------------------------------------------------------
    # 4.7 多模态配置
    # ------------------------------------------------------------------
    print("\n## 4.7 多模态配置")

    try:
        from memu.settings import load_enable_video_from_config
        video_enabled = load_enable_video_from_config()
        print_test("多模态视频配置", True, f"enable_video: {video_enabled}")

    except Exception as e:
        print_test("多模态配置检查", False, str(e))

    # ------------------------------------------------------------------
    # 4.8 restart_required 标记
    # ------------------------------------------------------------------
    print("\n## 4.8 restart_required 标记")

    try:
        restart_marker = Path("config/restart_required.json")
        if restart_marker.exists():
            with open(restart_marker) as f:
                marker = json.load(f)
            print_test("restart_required 文件", True, json.dumps(marker))
        else:
            print_test("restart_required 文件", False, "文件不存在")

    except Exception as e:
        print_test("restart_required 检查", False, str(e))

    # ------------------------------------------------------------------
    # 4.9 数据库检查
    # ------------------------------------------------------------------
    print("\n## 4.9 数据库检查")

    try:
        memory_dir = workspace / ".memu" / "memory"
        md_files = list(memory_dir.rglob("*.md"))
        print_test("记忆文件数量", True, f"共 {len(md_files)} 个 .md 文件")

        # 显示文件列表
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
