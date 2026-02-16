#!/bin/bash
# 紧急修复 MemU - 强制启用记忆系统

set -e

echo "🚨 紧急修复 MemU 记忆系统"
echo "============================"

# 1. 更新代码
echo "📥 拉取最新代码..."
cd ~/nanoBot_memU/nanobot
git pull origin main

# 2. 检查 Python 环境
echo "🐍 检查 Python 环境..."
source ~/nanobot-venv/bin/activate

# 3. 重新安装
echo "⚙️ 重新安装 nanobot..."
pip install --force-reinstall -e . > /dev/null 2>&1

# 4. 验证代码修改
echo "✅ 验证 MemU 强制启用..."
python3 -c "
from nanobot.agent.memory_adapter import MemoryAdapter
import inspect

# 检查 __init__ 方法源码
src = inspect.getsource(MemoryAdapter.__init__)
if 'self.enable_memory = True' in src:
    print('✓ memory_adapter.py 已强制启用')
else:
    print('✗ memory_adapter.py 未强制启用')
    exit(1)
"

# 5. 重启服务
echo "🔄 重启服务..."
sudo systemctl restart nanobot-agent@root

# 6. 等待服务启动
sleep 3

# 7. 检查状态
echo "📊 检查服务状态..."
if sudo systemctl is-active --quiet nanobot-agent@root; then
    echo "✅ 服务运行中"
else
    echo "❌ 服务启动失败"
    sudo systemctl status nanobot-agent@root --no-pager -l
    exit 1
fi

# 8. 测试 MemU
echo "🧪 测试 MemU..."
python3 -c "
import sys
sys.path.insert(0, '/root/nanoBot_memU/nanobot')
from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter

config = load_config()
adapter = MemoryAdapter(
    workspace=config.workspace_path,
    enable_memory=True,
    memu_config=config.memu,
)

print(f'MemU enabled: {adapter.enable_memory}')
print(f'MemoryAgent: {adapter._memory_agent}')

# 检查记忆目录
import os
memory_dir = config.workspace_path / '.memu' / 'memory'
if memory_dir.exists():
    print(f'✓ 记忆目录存在: {memory_dir}')
    files = list(memory_dir.rglob('*.md'))
    print(f'✓ 找到 {len(files)} 个记忆文件')
    for f in files[:3]:
        print(f'  - {f}')
else:
    print(f'✗ 记忆目录不存在: {memory_dir}')
"

echo ""
echo "============================"
echo "✅ 修复完成！"
echo ""
echo "测试命令:"
echo "  /root/nanobot-venv/bin/nanobot agent -m '/memu status'"
