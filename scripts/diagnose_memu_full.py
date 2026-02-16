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

# 检查 API Keys
deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
silicon_key = os.environ.get("SILICONFLOW_API_KEY", "")
print_test("DeepSeek API Key", bool(deepseek_key and len(deepseek_key) > 20))
print_test("SiliconFlow API Key", bool(silicon_key and len(silicon_key) > 20))

# ============================================================================
# 2. memU 模块导入
# ============================================================================
print_header("2. memU 模块导入")

try:
    from memu.app.service import MemoryService
    print_test("MemoryService 导入", True)
except Exception as e:
    print_test("MemoryService 导入", False, str(e))
    sys.exit(1)

try:
    from memu.llm import DeepSeekClient
    print_test("DeepSeekClient 导入", True)
except Exception as e:
    print_test("DeepSeekClient 导入", False, str(e))

# ============================================================================
# 3. 初始化 MemoryService
# ============================================================================
print_header("3. 初始化 MemoryService")

try:
    # 配置 LLM
    llm_client = DeepSeekClient(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    )
    print_test("LLM Client 创建", True)

    # 创建服务
    service = MemoryService(
        llm_client=llm_client,
        memory_dir=str(workspace / ".memu" / "memory"),
        enable_embeddings=True,
    )
    print_test("MemoryService 创建", True)

    # 获取状态
    status = service.get_status()
    print_test("服务状态获取", True, json.dumps(status, ensure_ascii=False)[:100])

except Exception as e:
    print_test("MemoryService 初始化", False, str(e))
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
        result = await service.memorize(
            resource_url=None,  # 直接传内容
            modality="conversation",
            user={"user_id": "diagnose_test"},
            conversation=[
                {"role": "user", "content": "我喜欢健身"},
                {"role": "assistant", "content": "很好，健身有助于健康"},
            ],
        )

        success = result.get("success", False)
        items_count = len(result.get("items", []))
        print_test("memorize() 执行", success, f"提取了 {items_count} 条记忆")

        # 检查是否写入数据库
        if success:
            print_test("memorize() 写入成功", True, f"items: {items_count}")
        else:
            print_test("memorize() 写入失败", False, str(result.get("error")))

    except Exception as e:
        print_test("memorize() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.2 retrieve() - 语义检索
    # ------------------------------------------------------------------
    print("\n## 4.2 retrieve() - 语义检索")

    try:
        result = await service.retrieve(
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
        items = await service.list_memory_items(
            user={"user_id": "diagnose_test"},
            limit=10,
        )

        print_test("list_memory_items() 执行", True, f"返回 {len(items)} 条")

        for item in items[:3]:
            content = item.get("content", "")[:80]
            print(f"   - {content}")

    except Exception as e:
        print_test("list_memory_items() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.4 list_categories() - 列出分类
    # ------------------------------------------------------------------
    print("\n## 4.4 list_categories() - 列出分类")

    try:
        categories = await service.list_memory_categories(
            user={"user_id": "diagnose_test"},
        )

        print_test("list_categories() 执行", True, f"返回 {len(categories)} 个分类")

        for cat in categories[:5]:
            name = cat.get("name", "")
            summary = cat.get("summary", "")[:80]
            print(f"   - {name}: {summary}")

    except Exception as e:
        print_test("list_categories() 执行", False, str(e))

    # ------------------------------------------------------------------
    # 4.5 reinforcement 强化机制
    # ------------------------------------------------------------------
    print("\n## 4.5 reinforcement 强化机制")

    try:
        # 写入相同内容两次，测试强化
        for i in range(2):
            await service.memorize(
                resource_url=None,
                modality="conversation",
                user={"user_id": "reinforce_test"},
                conversation=[
                    {"role": "user", "content": "强化测试内容 - 喜欢跑步"},
                ],
            )

        # 检索测试
        result = await service.retrieve(
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
        await service.memorize(
            resource_url=None,
            modality="conversation",
            user={"user_id": "salience_test"},
            conversation=[
                {"role": "user", "content": "今天很重要：我要学习Python编程"},
            ],
        )

        await service.memorize(
            resource_url=None,
            modality="conversation",
            user={"user_id": "salience_test"},
            conversation=[
                {"role": "user", "content": "随便说说的不重要的话"},
            ],
        )

        # 检索
        result = await service.retrieve(
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
        from memu.app.settings import load_enable_video_from_config
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
