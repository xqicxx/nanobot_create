#!/bin/bash
# 清理并重新安装正确的 memu-py

echo "🧹 清理错误的 memu 包..."

# 1. 卸载所有 memu 相关的包
/root/nanobot-venv/bin/pip uninstall memu memu-py memu_py -y 2>/dev/null || true

# 2. 删除残留文件
rm -rf /root/nanobot-venv/lib/python3.13/site-packages/memu*
rm -rf /root/nanobot-venv/lib/python3.13/site-packages/*memu*

echo "📦 安装正确的 memu-py..."

# 3. 安装正确的包（注意是 memu-py，不是 memu_py）
/root/nanobot-venv/bin/pip install --no-cache-dir "memu-py==0.2.2"

echo "✅ 验证安装..."

# 4. 验证
/root/nanobot-venv/bin/python -c "
from memu.memory import MemoryAgent
from memu.llm import DeepSeekClient
print('✓ memu-py 0.2.2 安装成功！')
"

echo "🔄 重启服务..."
sudo systemctl restart nanobot-agent@root

echo "✅ 完成！"
