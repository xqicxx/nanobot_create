#!/usr/bin/env python3
"""
nanobot 系统诊断脚本
用于验证 DeepSeek、Embedding 和 memU 记忆系统是否正常工作

使用方法:
    python scripts/diagnose_system.py

或指定 API Key:
    python scripts/diagnose_system.py <deepseek_api_key> [siliconflow_api_key]
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 尝试添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    print(f"\n## {title}")
    print("-" * 50)


def print_result(label: str, status: str, details: str = "") -> None:
    status_icon = "✅" if "正常" in status or "成功" in status or "OK" in status else "❌"
    print(f"  {status_icon} {label}: {status}")
    if details:
        print(f"     {details}")


# ============================================================================
# 1. DeepSeek API 检测
# ============================================================================

def check_deepseek_config() -> dict[str, Any]:
    """检查 DeepSeek 配置"""
    result = {
        "status": "unknown",
        "api_key": None,
        "base_url": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-chat",
        "error": None,
    }

    # 1. 检查配置文件
    config_path = Path.home() / ".nanobot" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            memu = config.get("memu", {})
            default = memu.get("default", {})
            result["api_key"] = default.get("apiKey")
            result["base_url"] = default.get("baseUrl", result["base_url"])
            result["chat_model"] = default.get("chatModel", result["chat_model"])
            print_result("配置文件", "已找到", f"位置: {config_path}")
        except Exception as e:
            result["error"] = f"配置文件读取失败: {e}"
            print_result("配置文件", "读取失败", str(e))
            return result
    else:
        print_result("配置文件", "未找到", str(config_path))

    # 2. 检查环境变量（覆盖配置）
    env_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_api_key:
        result["api_key"] = env_api_key
        print_result("环境变量", "已设置", "DEEPSEEK_API_KEY")

    # 3. 检查 API Key 有效性
    if not result["api_key"]:
        result["error"] = "未配置 API Key"
        result["status"] = "not_configured"
        return result

    if result["api_key"] in ["your-deepseek-api-key", "sk-your-deepseek-api-key", ""]:
        result["error"] = "API Key 未修改（仍为示例值）"
        result["status"] = "not_configured"
        return result

    if len(result["api_key"]) < 20:
        result["error"] = f"API Key 格式不正确: {result['api_key'][:10]}..."
        result["status"] = "invalid"
        return result

    result["status"] = "configured"
    return result


def test_deepseek_api(api_key: str, base_url: str, model: str) -> dict[str, Any]:
    """测试 DeepSeek API 是否可用"""
    result = {
        "status": "unknown",
        "response_time_ms": 0,
        "model": None,
        "error": None,
    }

    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            result["response_time_ms"] = elapsed_ms
            resp_data = json.loads(response.read().decode("utf-8"))
            result["model"] = resp_data.get("model")
            result["status"] = "ok"
            print_result("API 响应", "正常", f"{elapsed_ms}ms")
            print_result("模型", result["model"] or "unknown")
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            error_msg = error_body.get("error", {}).get("message", "Unknown error")
            error_code = error_body.get("error", {}).get("code", "unknown")
        except:
            error_msg = str(e)
            error_code = "http_error"

        result["error"] = f"{error_code}: {error_msg}"
        result["status"] = "error"
        print_result("API 响应", "失败", f"HTTP {e.code}: {error_msg}")

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
        print_result("API 响应", "失败", str(e))

    return result


# ============================================================================
# 2. Embedding 服务检测
# ============================================================================

def check_embedding_config() -> dict[str, Any]:
    """检查 Embedding 配置"""
    result = {
        "status": "unknown",
        "api_key": None,
        "base_url": "https://api.siliconflow.cn/v1",
        "embed_model": "BAAI/bge-m3",
        "error": None,
    }

    # 1. 检查配置文件
    config_path = Path.home() / ".nanobot" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            memu = config.get("memu", {})
            embedding = memu.get("embedding", {})
            result["api_key"] = embedding.get("apiKey")
            result["base_url"] = embedding.get("baseUrl", result["base_url"])
            result["embed_model"] = embedding.get("embedModel", result["embed_model"])
            print_result("配置文件", "已找到")
        except Exception as e:
            result["error"] = f"配置文件读取失败: {e}"
            return result

    # 2. 检查环境变量
    env_api_key = os.environ.get("SILICONFLOW_API_KEY")
    if env_api_key:
        result["api_key"] = env_api_key
        print_result("环境变量", "已设置", "SILICONFLOW_API_KEY")

    if not result["api_key"]:
        result["status"] = "not_configured"
        result["error"] = "未配置 API Key"
        return result

    if result["api_key"] in ["your-siliconflow-api-key", "sk-your-siliconflow-api-key", ""]:
        result["status"] = "not_configured"
        result["error"] = "API Key 未修改"
        return result

    result["status"] = "configured"
    return result


def test_embedding_api(api_key: str, base_url: str, model: str) -> dict[str, Any]:
    """测试 Embedding API 是否可用"""
    result = {
        "status": "unknown",
        "response_time_ms": 0,
        "vector_dimensions": 0,
        "error": None,
    }

    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/embeddings"
    test_text = "Hello world"
    data = json.dumps({
        "model": model,
        "input": test_text
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            result["response_time_ms"] = elapsed_ms
            resp_data = json.loads(response.read().decode("utf-8"))

            embeddings = resp_data.get("data", [])
            if embeddings:
                vector = embeddings[0].get("embedding", [])
                result["vector_dimensions"] = len(vector)
                result["status"] = "ok"
                print_result("API 响应", "正常", f"{elapsed_ms}ms")
                print_result("向量维度", str(result["vector_dimensions"]))
            else:
                result["error"] = "No embeddings returned"
                result["status"] = "error"
                print_result("API 响应", "失败", "No embeddings returned")

    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            error_msg = error_body.get("error", {}).get("message", "Unknown error")
        except:
            error_msg = str(e)
        result["error"] = error_msg
        result["status"] = "error"
        print_result("API 响应", "失败", f"HTTP {e.code}: {error_msg}")

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
        print_result("API 响应", "失败", str(e))

    return result


# ============================================================================
# 3. memU 记忆系统检测
# ============================================================================

def check_memu_system() -> dict[str, Any]:
    """检查 memU 记忆系统"""
    result = {
        "status": "unknown",
        "config_exists": False,
        "workspace_exists": False,
        "memory_dir_exists": False,
        "restart_required": False,
        "import_ok": False,
        "agent_initialized": False,
        "error": None,
    }

    # 1. 检查配置文件
    config_path = Path.home() / ".nanobot" / "config.json"
    result["config_exists"] = config_path.exists()
    if result["config_exists"]:
        print_result("配置文件", "存在", str(config_path))
    else:
        print_result("配置文件", "不存在", str(config_path))

    # 2. 检查工作空间
    workspace = Path.home() / ".nanobot" / "workspace"
    result["workspace_exists"] = workspace.exists()
    if result["workspace_exists"]:
        print_result("工作空间", "存在", str(workspace))
    else:
        print_result("工作空间", "不存在", "将自动创建")

    # 3. 检查 memory 目录
    memory_dir = workspace / ".memu" / "memory"
    result["memory_dir_exists"] = memory_dir.exists()
    if result["memory_dir_exists"]:
        print_result("记忆目录", "存在", str(memory_dir))
    else:
        print_result("记忆目录", "不存在", "将自动创建")

    # 4. 检查 restart_required 标记
    restart_marker = Path("config") / "restart_required.json"
    if restart_marker.exists():
        try:
            with open(restart_marker, encoding="utf-8") as f:
                marker_data = json.load(f)
            result["restart_required"] = marker_data.get("restart_required", False)
            if result["restart_required"]:
                reason = marker_data.get("reason", "unknown")
                print_result("重启标记", "需要重启", f"原因: {reason}")
            else:
                print_result("重启标记", "无需重启")
        except Exception as e:
            print_result("重启标记", "检查失败", str(e))
    else:
        print_result("重启标记", "文件不存在")

    # 5. 测试导入 memu
    try:
        from memu.memory import MemoryAgent
        result["import_ok"] = True
        print_result("memu-py", "导入成功")
    except ImportError as e:
        result["error"] = f"memu-py 导入失败: {e}"
        print_result("memu-py", "导入失败", str(e))
        return result

    # 6. 尝试初始化 MemoryAdapter
    try:
        from nanobot.config.loader import load_config
        from nanobot.agent.memory_adapter import MemoryAdapter

        config = load_config()
        print_result("配置加载", "成功", f"MemU enabled: {config.memu.enabled}")

        adapter = MemoryAdapter(
            workspace=config.workspace_path,
            memu_config=config.memu,
        )

        if adapter._memory_agent is not None:
            result["agent_initialized"] = True
            print_result("MemoryAgent", "已初始化")
        else:
            print_result("MemoryAgent", "未初始化", "将使用文件存储")

        result["status"] = "ok"

    except Exception as e:
        result["error"] = str(e)
        print_result("MemoryAdapter", "初始化失败", str(e))

    return result


async def test_memu_operations() -> dict[str, Any]:
    """测试 memU 记忆操作"""
    result = {
        "write": "unknown",
        "read": "unknown",
        "query": "unknown",
    }

    try:
        from nanobot.config.loader import load_config
        from nanobot.agent.memory_adapter import MemoryAdapter

        config = load_config()
        adapter = MemoryAdapter(
            workspace=config.workspace_path,
            memu_config=config.memu,
        )

        # 测试写入
        print_section("测试写入记忆")
        try:
            await adapter.memorize_turn(
                channel="diagnose",
                chat_id="test-session",
                sender_id="diagnose-user",
                user_message="测试消息：我的名字叫诊断测试用户",
                assistant_message="你好诊断测试用户！我记住你了。",
            )
            result["write"] = "ok"
            print_result("写入", "成功")
        except Exception as e:
            result["write"] = "error"
            print_result("写入", "失败", str(e))

        # 测试读取
        print_section("测试读取记忆")
        try:
            items = await adapter.query_items(
                channel="diagnose",
                chat_id="test-session",
                sender_id="diagnose-user",
                limit=5,
            )
            if items:
                result["read"] = "ok"
                print_result("读取", "成功", f"找到 {len(items)} 条记忆")
            else:
                result["read"] = "empty"
                print_result("读取", "空", "暂无记忆")
        except Exception as e:
            result["read"] = "error"
            print_result("读取", "失败", str(e))

        # 测试状态查询
        print_section("测试状态查询")
        try:
            status = await adapter.memu_status()
            result["query"] = "ok"
            print_result("状态查询", "成功", json.dumps(status, ensure_ascii=False)[:100])
        except Exception as e:
            result["query"] = "error"
            print_result("状态查询", "失败", str(e))

    except Exception as e:
        print_result("操作测试", "初始化失败", str(e))

    return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    print_header("nanobot 系统诊断")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    print(f"用户: {os.environ.get('USER', 'unknown')}")

    results = {
        "deepseek": {},
        "embedding": {},
        "memu": {},
    }

    # =========================================================================
    # 1. DeepSeek 检测
    # =========================================================================
    print_section("1. DeepSeek API 检测")

    deepseek_cfg = check_deepseek_config()
    results["deepseek"]["config"] = deepseek_cfg

    if deepseek_cfg["status"] == "configured":
        deepseek_test = test_deepseek_api(
            api_key=deepseek_cfg["api_key"],
            base_url=deepseek_cfg["base_url"],
            model=deepseek_cfg["chat_model"],
        )
        results["deepseek"]["test"] = deepseek_test
    else:
        print_result("DeepSeek", "未配置", deepseek_cfg.get("error", ""))

    # =========================================================================
    # 2. Embedding 检测
    # =========================================================================
    print_section("2. Embedding 服务检测")

    embedding_cfg = check_embedding_config()
    results["embedding"]["config"] = embedding_cfg

    if embedding_cfg["status"] == "configured":
        embedding_test = test_embedding_api(
            api_key=embedding_cfg["api_key"],
            base_url=embedding_cfg["base_url"],
            model=embedding_cfg["embed_model"],
        )
        results["embedding"]["test"] = embedding_test
    else:
        print_result("Embedding", "未配置", embedding_cfg.get("error", ""))

    # =========================================================================
    # 3. memU 检测
    # =========================================================================
    print_section("3. memU 记忆系统检测")

    memu_check = check_memu_system()
    results["memu"]["check"] = memu_check

    # =========================================================================
    # 输出总结报告
    # =========================================================================
    print_header("诊断总结")

    # DeepSeek 状态
    deepseek_ok = results["deepseek"].get("test", {}).get("status") == "ok"
    if deepseek_ok:
        rt = results["deepseek"]["test"]["response_time_ms"]
        print(f"✅ DeepSeek: 正常 (响应时间: {rt}ms)")
    else:
        error = results["deepseek"]["config"].get("error") or results["deepseek"]["test"].get("error", "未知错误")
        print(f"❌ DeepSeek: 异常 - {error}")

    # Embedding 状态
    embedding_ok = results["embedding"].get("test", {}).get("status") == "ok"
    if embedding_ok:
        dim = results["embedding"]["test"]["vector_dimensions"]
        rt = results["embedding"]["test"]["response_time_ms"]
        print(f"✅ Embedding: 正常 (维度: {dim}, 响应时间: {rt}ms)")
    else:
        error = results["embedding"]["config"].get("error") or results["embedding"]["test"].get("error", "未知错误")
        print(f"❌ Embedding: 异常 - {error}")

    # memU 状态
    memu_ok = results["memu"]["check"].get("status") == "ok"
    restart_required = results["memu"]["check"].get("restart_required", False)
    if memu_ok:
        print(f"✅ memU: 正常")
        if restart_required:
            print(f"   ⚠️  需要重启")
    else:
        error = results["memu"]["check"].get("error", "未知错误")
        print(f"❌ memU: 异常 - {error}")

    # =========================================================================
    # 修复建议
    # =========================================================================
    if not deepseek_ok or not embedding_ok or not memu_ok:
        print_header("修复建议")

        if not deepseek_ok:
            print("\n📌 DeepSeek API:")
            print("   1. 访问 https://platform.deepseek.com/")
            print("   2. 登录后创建 API Key")
            print("   3. 编辑 ~/.nanobot/config.json:")
            print('      "memu": { "default": { "apiKey": "sk-xxx", "baseUrl": "https://api.deepseek.com/v1" } }')
            print("   4. 或设置环境变量: export DEEPSEEK_API_KEY=sk-xxx")

        if not embedding_ok:
            print("\n📌 Embedding:")
            print("   1. 访问 https://siliconflow.cn/")
            print("   2. 创建 API Key")
            print("   3. 编辑 ~/.nanobot/config.json:")
            print('      "memu": { "embedding": { "apiKey": "sk-xxx" } }')
            print("   4. 或设置环境变量: export SILICONFLOW_API_KEY=sk-xxx")

        if not memu_ok:
            print("\n📌 memU:")
            print("   1. 确保配置文件存在: ~/.nanobot/config.json")
            print("   2. 确保 memu.enabled = true")
            print("   3. 重启 nanobot 服务")

    # 保存详细报告
    report_path = Path("diagnose_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细报告已保存: {report_path}")

    return 0 if (deepseek_ok and embedding_ok and memu_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
