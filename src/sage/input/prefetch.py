"""One bounded producer thread between frame assembly and the optimizer.

Frame assembly is CPU and IO work; mapping is GPU work. A bounded queue lets
them overlap without letting the producer run arbitrarily far ahead, and an
incomplete run must be aborted rather than closed so a partial result can never
be mistaken for a complete one.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import resource
import time
from queue import Full, Queue
from threading import Event, Thread
from typing import Callable, Generic, Iterable, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class _StreamError:
    error: BaseException


class BoundedResultStream(Generic[T]):
    """Expose one ordered producer through a bounded, lossless iterator.

    The producer is deliberately kept behind this small interface.  It may do sensor
    decoding and projection, while the consumer controls how quickly the producer can
    advance by filling the bounded queue.  A normal close is only valid after the
    iterator has consumed the producer's completion sentinel; early termination must
    use :meth:`abort` so an incomplete receipt cannot be mistaken for a complete run.
    """

    def __init__(
        self,
        producer: Callable[[], Iterable[T]],
        *,
        identity: Mapping[str, object],
        queue_capacity: int = 1,
    ) -> None:
        if not callable(producer):
            raise ValueError("BoundedResultStream producer must be callable")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("BoundedResultStream queue_capacity must be a positive integer")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("BoundedResultStream identity must be a non-empty object")
        self._producer = producer
        self._identity = deepcopy(dict(identity))
        self._queue_capacity = queue_capacity
        self._queue: Queue[object] = Queue(maxsize=queue_capacity)
        self._stop = Event()
        self._thread: Thread | None = None
        self._started = False
        self._exhausted = False
        self._aborted = False
        self._error: BaseException | None = None
        self._items_emitted = 0
        self._max_queue_depth = 0
        self._producer_stall_seconds = 0.0
        self._consumer_wait_seconds = 0.0

    def start_identity(self) -> dict[str, object]:
        if self._aborted:
            raise RuntimeError("BoundedResultStream has already been aborted")
        return deepcopy(self._identity)

    @property
    def items_emitted(self) -> int:
        return self._items_emitted

    def _offer(self, item: object) -> bool:
        stalled_at = None
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                if stalled_at is not None:
                    self._producer_stall_seconds += time.monotonic() - stalled_at
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
                return True
            except Full:
                if stalled_at is None:
                    stalled_at = time.monotonic()
                continue
        return False

    def _run_producer(self) -> None:
        try:
            for item in self._producer():
                if not self._offer(item):
                    return
            self._offer(None)
        except BaseException as exc:  # surfaced on the consumer thread
            self._error = exc
            self._offer(_StreamError(exc))

    def frames(self):
        if self._started:
            raise RuntimeError("BoundedResultStream.frames() is single-use")
        if self._aborted:
            raise RuntimeError("BoundedResultStream has already been aborted")
        self._started = True
        self._thread = Thread(target=self._run_producer, name="bounded-result-producer", daemon=True)
        self._thread.start()
        while True:
            waiting_at = time.monotonic()
            item = self._queue.get()
            self._consumer_wait_seconds += time.monotonic() - waiting_at
            if item is None:
                self._exhausted = True
                break
            if isinstance(item, _StreamError):
                self._error = item.error
                raise item.error
            self._items_emitted += 1
            yield item

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("BoundedResultStream producer did not stop")

    def close(self) -> dict[str, object]:
        if not self._started:
            raise RuntimeError("BoundedResultStream must be iterated before close()")
        if not self._exhausted:
            raise RuntimeError("BoundedResultStream.close() requires a fully consumed iterator")
        self._join()
        if self._error is not None:
            raise RuntimeError("BoundedResultStream producer failed") from self._error
        return {
            **deepcopy(self._identity),
            "complete": True,
            "aborted": False,
            "items_emitted": self._items_emitted,
            "queue_capacity": self._queue_capacity,
            "max_queue_depth": self._max_queue_depth,
            "producer_stall_seconds": self._producer_stall_seconds,
            "consumer_wait_seconds": self._consumer_wait_seconds,
            "peak_cpu_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "error": None,
        }

    def abort(self, error: BaseException | str) -> dict[str, object]:
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        else:
            message = str(error)
        if not message:
            raise ValueError("BoundedResultStream abort error must be non-empty")
        self._aborted = True
        self._stop.set()
        self._join()
        return {
            **deepcopy(self._identity),
            "complete": False,
            "aborted": True,
            "items_emitted": self._items_emitted,
            "queue_capacity": self._queue_capacity,
            "max_queue_depth": self._max_queue_depth,
            "producer_stall_seconds": self._producer_stall_seconds,
            "consumer_wait_seconds": self._consumer_wait_seconds,
            "peak_cpu_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "error": message,
        }

__all__ = ["BoundedResultStream"]
