#!/bin/bash
# 服务器测试脚本 - 一键拉取并测试高性能 memU 组件

set -e

echo "============================================"
echo "  高性能 memU 组件服务器测试"
echo "============================================"

# 1. 拉取最新代码
echo ""
echo "[1/4] 拉取最新代码..."
cd /root/nanoBot_memU
git pull origin main

# 2. 检查依赖
echo ""
echo "[2/4] 检查依赖..."
source /root/nanobot-venv/bin/activate

python3 -c "
import sys
sys.path.insert(0, '/root/nanoBot_memU/nanobot')
from nanobot.agent.memory_decider import MemoryTriggerDecider
from nanobot.agent.memory_cache import EmbeddingCache
from nanobot.agent.memory_throttle import RetrieveThrottler
from nanobot.agent.memory_queue import MemoryWriteQueue
from nanobot.agent.memory_performance import MemUPerformanceConfig
print('✅ 所有组件导入成功')
"

# 3. 运行测试
echo ""
echo "[3/4] 运行测试..."
python3 /root/nanoBot_memU/nanobot/scripts/test_high_performance.py

# 4. 测试 MemoryAdapter 集成
echo ""
echo "[4/4] 测试 MemoryAdapter 集成..."
python3 -c "
import sys
import os
sys.path.insert(0, '/root/nanoBot_memU/nanobot')

# 设置环境变量
os.environ['MEMU_EMBEDDING_MODEL'] = 'BAAI/bge-m3'

from nanobot.config.loader import load_config
from nanobot.agent.memory_adapter import MemoryAdapter
from nanobot.agent.memory_performance import MemUPerformanceConfig

config = load_config()

# 测试默认配置
adapter1 = MemoryAdapter(
    workspace=config.workspace_path,
    memu_config=config.memu,
)
print('✅ 默认配置 MemoryAdapter 创建成功')

# 测试自定义配置
custom_config = MemUPerformanceConfig(
    enable_trigger_rules=True,
    embedding_cache_enabled=True,
    write_queue_enabled=True,
)
adapter2 = MemoryAdapter(
    workspace=config.workspace_path,
    memu_config=config.memu,
    performance_config=custom_config,
)
print('✅ 自定义配置 MemoryAdapter 创建成功')

# 测试触发决策
result = adapter2.should_memorize('我喜欢吃苹果')
print(f'✅ 触发决策 should_memorize: {result.should_trigger}, reason={result.reason}')

result2 = adapter2.should_retrieve('什么是向量？')
print(f'✅ 触发决策 should_retrieve: {result2.should_trigger}, reason={result2.reason}')

# 测试性能统计
stats = adapter2.get_performance_stats()
print(f'✅ 性能统计: {stats}')

print('')
print('============================================')
print('  ✅ 所有测试通过!')
print('============================================')
"

echo ""
echo "测试完成！"
