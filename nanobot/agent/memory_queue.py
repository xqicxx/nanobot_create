"""延迟写入队列 - 批量写入减少API调用"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueuedItem:
    """队列项"""
    content: str
    category: str
    user_id: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


@dataclass
class WriteStats:
    """写入统计"""
    enqueued: int = 0
    flushed: int = 0
    failed: int = 0


class MemoryWriteQueue:
    """延迟写入队列 - 批量写入减少API调用"""

    def __init__(
        self,
        batch_size: int = 5,           # 批量大小
        flush_interval: int = 30,     # 强制刷新间隔(秒)
        max_queue_size: int = 20,     # 最大队列长度
    ):
        self._queue: list[QueuedItem] = []
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_queue_size = max_queue_size

        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        self._flush_callback: Any = None  # 实际的写入函数
        self._worker_task: asyncio.Task | None = None
        self._stats = WriteStats()

        # 启动后台 worker
        self._running = True

    def set_flush_callback(self, callback: Any) -> None:
        """设置实际写入回调函数"""
        self._flush_callback = callback

    async def start(self) -> None:
        """启动后台 worker"""
        if self._worker_task is None or self._worker_task.done():
            self._running = True
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """停止后台 worker"""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            await self._worker_task
        # 最后刷新一次
        await self.flush()

    async def _worker(self) -> None:
        """后台 worker - 定期刷新"""
        while self._running:
            await asyncio.sleep(5)  # 每5秒检查一次
            if time.time() - self._last_flush > self._flush_interval:
                await self.flush()

    async def enqueue(
        self,
        content: str,
        category: str,
        user_id: str,
        metadata: dict | None = None,
        priority: int = 0,
    ) -> None:
        """加入写入队列"""
        async with self._lock:
            item = QueuedItem(
                content=content,
                category=category,
                user_id=user_id,
                metadata=metadata or {},
                priority=priority,
            )
            self._queue.append(item)
            self._stats.enqueued += 1

            # 检查是否需要立即刷新
            should_flush = (
                len(self._queue) >= self._batch_size or
                len(self._queue) >= self._max_queue_size
            )

        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """批量写入"""
        async with self._lock:
            if not self._queue:
                return

            # 按优先级排序
            self._queue.sort(key=lambda x: -x.priority)

            # 取出一批
            batch = self._queue[:self._batch_size]
            self._queue = self._queue[self._batch_size:]
            self._last_flush = time.time()

        if not batch:
            return

        # 执行批量写入
        if self._flush_callback:
            try:
                for item in batch:
                    await self._flush_callback(
                        content=item.content,
                        category=item.category,
                        user_id=item.user_id,
                        metadata=item.metadata,
                    )
                self._stats.flushed += len(batch)
            except Exception as e:
                self._stats.failed += len(batch)
                # 可以加入重试逻辑

    def size(self) -> int:
        """返回队列大小"""
        return len(self._queue)

    def get_stats(self) -> WriteStats:
        """获取统计信息"""
        return self._stats

    def is_empty(self) -> bool:
        """队列是否为空"""
        return len(self._queue) == 0
