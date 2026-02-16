"""Embedding 向量缓存"""

import hashlib
import time
from collections import OrderedDict
from typing import Any


class EmbeddingCache:
    """Embedding 向量缓存 - LRU + TTL"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl  # 秒

    def get(self, text: str) -> list[float] | None:
        """获取缓存的 embedding"""
        key = self._hash(text)
        if key in self._cache:
            vector, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # 移到末尾（LRU）
                self._cache.move_to_end(key)
                return vector
            else:
                # 过期删除
                del self._cache[key]
        return None

    def set(self, text: str, vector: list[float]) -> None:
        """设置缓存"""
        key = self._hash(text)

        # 如果已存在，删除旧的
        if key in self._cache:
            del self._cache[key]

        # LRU 淘汰：如果满了，删除最老的
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (vector, time.time())

    def invalidate(self, text: str) -> None:
        """使缓存失效"""
        key = self._hash(text)
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)

    @staticmethod
    def _hash(text: str) -> str:
        """简单哈希"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()


class RetrievalCache:
    """检索结果缓存"""

    def __init__(self, max_size: int = 100, ttl: int = 300):
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl  # 5分钟

    def get(self, query: str) -> dict | None:
        """获取缓存的检索结果"""
        key = self._hash(query)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                return result
            else:
                del self._cache[key]
        return None

    def set(self, query: str, result: dict) -> None:
        """设置缓存"""
        key = self._hash(query)

        if key in self._cache:
            del self._cache[key]

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)

    @staticmethod
    def _hash(query: str) -> str:
        """哈希查询"""
        # 简化：取前100字符
        normalized = query.strip()[:100].lower()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()
