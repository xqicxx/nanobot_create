"""检索节流器"""

import time
from dataclasses import dataclass


@dataclass
class ThrottleStats:
    """节流统计"""
    step_count: int = 0
    minute_count: int = 0
    cooldown_hits: int = 0


class RetrieveThrottler:
    """检索节流器"""

    def __init__(
        self,
        max_per_step: int = 1,       # 每轮最多1次检索
        max_per_minute: int = 5,     # 每分钟最多5次
        cooldown_seconds: int = 10,  # 同类查询冷却10秒
    ):
        self._max_per_step = max_per_step
        self._max_per_minute = max_per_minute
        self._cooldown = cooldown_seconds

        self._step_count = 0
        self._minute_history: list[float] = []
        self._query_history: dict[str, float] = {}
        self._stats = ThrottleStats()

    def can_retrieve(self, query: str) -> bool:
        """检查是否可以检索"""
        now = time.time()

        # 1. 每轮限制
        if self._step_count >= self._max_per_step:
            self._stats.cooldown_hits += 1
            return False

        # 2. 每分钟限制
        self._minute_history = [t for t in self._minute_history if now - t < 60]
        if len(self._minute_history) >= self._max_per_minute:
            self._stats.cooldown_hits += 1
            return False

        # 3. 同query冷却（简化key：前50字符）
        query_key = query.strip()[:50].lower()
        if query_key in self._query_history:
            if now - self._query_history[query_key] < self._cooldown:
                self._stats.cooldown_hits += 1
                return False

        return True

    def record_retrieve(self, query: str) -> None:
        """记录一次检索"""
        now = time.time()
        self._step_count += 1
        self._minute_history.append(now)
        self._stats.step_count += 1
        self._stats.minute_count += 1

        query_key = query.strip()[:50].lower()
        self._query_history[query_key] = now

        # 清理过期的query历史
        self._query_history = {
            k: v for k, v in self._query_history.items()
            if now - v < self._cooldown * 2
        }

    def reset_per_step(self) -> None:
        """每轮重置"""
        self._step_count = 0

    def get_stats(self) -> ThrottleStats:
        """获取统计信息"""
        return self._stats

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = ThrottleStats()
