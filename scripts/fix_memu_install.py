#!/usr/bin/env python3
"""
memu-py 安装修复脚本
用于排查和修复 memu-py 安装问题
"""

import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd, shell=False):
    """执行命令并返回结果"""
    print(f"\n>>> 执行: {cmd}")
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(),
            capture_output=True,
            text=True,
            shell=shell
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def main():
    print("=" * 60)
    print("  memu-py 安装修复")
    print("=" * 60)

    # 1. 检查当前 Python
    print("\n## 1. 检查当前环境")
    print(f"当前 Python: {sys.executable}")
    print(f"Python 版本: {sys.version}")

    # 2. 检查 pip
    print("\n## 2. 检查 pip")
    run_cmd(f"{sys.executable} -m pip --version", shell=True)

    # 3. 检查 memu-py 是否已安装
    print("\n## 3. 检查 memu-py 安装状态")
    run_cmd(f"{sys.executable} -m pip show memu-py", shell=True)

    # 4. 尝试导入
    print("\n## 4. 尝试导入测试")
    test_code = """
import sys
print(f'Python: {sys.executable}')
try:
    import memu
    print(f'memu 模块: {memu}')
    print(f'memu 路径: {memu.__file__}')
except ImportError as e:
    print(f'导入失败: {e}')

    # 检查是否安装了类似的包
    import pkg_resources
    installed = [pkg.key for pkg in pkg_resources.working_set]
    memu_pkgs = [p for p in installed if 'memu' in p.lower()]
    print(f'已安装的相关包: {memu_pkgs}')
"""
    run_cmd(f'{sys.executable} -c "{test_code}"', shell=True)

    # 5. 尝试安装
    print("\n## 5. 尝试安装 memu-py")

    # 检查虚拟环境
    venv_path = "/root/nanobot-venv"
    if os.path.exists(venv_path):
        print(f"发现虚拟环境: {venv_path}")
        venv_python = os.path.join(venv_path, "bin", "python")
        if os.path.exists(venv_python):
            print(f"使用虚拟环境 Python: {venv_python}")
            print("\n在虚拟环境中安装:")
            run_cmd(f"{venv_python} -m pip install --no-cache-dir memu-py==0.2.2", shell=True)
            print("\n验证虚拟环境中安装:")
            run_cmd(f"{venv_python} -c 'import memu; print(\"OK!\")'", shell=True)
    else:
        print("未发现虚拟环境，直接安装到当前环境")
        run_cmd(f"{sys.executable} -m pip install --no-cache-dir memu-py==0.2.2", shell=True)

    # 6. 最终验证
    print("\n" + "=" * 60)
    print("  最终验证")
    print("=" * 60)

    # 尝试使用项目代码中的 Python
    test_import = f"""
import sys
sys.path.insert(0, '/root/nanoBot_memU/nanobot')
try:
    from memu.memory import MemoryAgent
    print('✅ memu.memory.MemoryAgent 导入成功!')
except ImportError as e:
    print(f'❌ 导入失败: {e}')
"""
    os.chdir("/root/nanoBot_memU/nanobot")
    run_cmd(f'/root/nanobot-venv/bin/python -c "{test_import}"', shell=True)


if __name__ == "__main__":
    main()
