"""Queue-based pipeline orchestration with thread workers."""
from __future__ import annotations

import queue
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class _Sentinel(Enum):
    DONE = auto()
    ERROR = auto()


@dataclass
class QueueItem:
    data: Any
    metadata: dict = field(default_factory=dict)
    error: Optional[Exception] = None

    @property
    def is_sentinel(self) -> bool:
        return isinstance(self.data, _Sentinel)

    @property
    def is_error(self) -> bool:
        return self.data is _Sentinel.ERROR


SENTINEL = QueueItem(data=_Sentinel.DONE)


class PipelineQueue:
    """Typed wrapper around queue.Queue with named stages."""

    def __init__(self, name: str, maxsize: int = 100):
        self.name = name
        self._q: queue.Queue[QueueItem] = queue.Queue(maxsize=maxsize)

    def put(self, item: QueueItem, timeout: float = 60.0) -> None:
        self._q.put(item, timeout=timeout)

    def get(self, timeout: float = 30.0) -> QueueItem:
        return self._q.get(timeout=timeout)

    def task_done(self) -> None:
        self._q.task_done()

    def join(self) -> None:
        self._q.join()

    def qsize(self) -> int:
        return self._q.qsize()


ProcessorFn = Callable[[Any, dict], Optional[Any]]


class PipelineStage:
    """
    Single stage in the pipeline.

    Reads QueueItem from `input_q`, calls `processor(data, metadata)`,
    writes result to `output_q`. Passes SENTINEL downstream when done.
    Multiple input sentinels (one per upstream worker) are counted so
    multi-worker upstream stages work correctly.
    """

    def __init__(
        self,
        name: str,
        processor: ProcessorFn,
        input_q: PipelineQueue,
        output_q: PipelineQueue,
        num_workers: int = 1,
        expected_sentinels: int = 1,
    ):
        self.name = name
        self.processor = processor
        self.input_q = input_q
        self.output_q = output_q
        self.num_workers = num_workers
        self._expected_sentinels = expected_sentinels
        self._sentinel_count = 0
        self._sentinel_lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self.errors: list[Exception] = []
        self._error_lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self) -> None:
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._loop,
                name=f"{self.name}-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        log.debug("Stage '%s' started %d worker(s)", self.name, self.num_workers)

    def join(self) -> None:
        for t in self._workers:
            t.join()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            try:
                item = self.input_q.get(timeout=30)
            except queue.Empty:
                continue

            try:
                if item.is_sentinel:
                    with self._sentinel_lock:
                        self._sentinel_count += 1
                        received_all = self._sentinel_count >= self._expected_sentinels
                    if received_all:
                        self.output_q.put(SENTINEL)
                    self.input_q.task_done()
                    break

                result = self.processor(item.data, item.metadata)
                if result is not None:
                    self.output_q.put(QueueItem(data=result, metadata=item.metadata))
                self.input_q.task_done()

            except Exception as exc:
                log.error("Stage '%s' error: %s", self.name, exc, exc_info=True)
                with self._error_lock:
                    self.errors.append(exc)
                err_item = QueueItem(data=_Sentinel.ERROR, error=exc)
                self.output_q.put(err_item)
                self.input_q.task_done()
