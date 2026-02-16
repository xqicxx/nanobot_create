"""高性能 memU 配置"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MemUPerformanceConfig:
    """memU 性能配置"""

    # === 触发规则 ===
    enable_trigger_rules: bool = True
    should_memorize_fn: Callable | None = None  # 自定义触发函数
    should_retrieve_fn: Callable | None = None

    # === 缓存策略 ===
    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 1000
    embedding_cache_ttl: int = 3600  # 秒

    # === 节流策略 ===
    retrieve_throttle_per_step: int = 1
    retrieve_throttle_per_minute: int = 5
    retrieve_cooldown_seconds: int = 10

    # === 写入策略 ===
    write_queue_enabled: bool = True
    write_batch_size: int = 5
    write_flush_interval: int = 30
    write_max_queue_size: int = 20

    # === 预取策略 ===
    prefetch_enabled: bool = True
    prefetch_on_start: bool = True  # 启动时预取

    # === 性能目标 ===
    max_retrieve_latency_ms: int = 500
    max_embedding_latency_ms: int = 300

    @classmethod
    def from_dict(cls, data: dict) -> "MemUPerformanceConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
