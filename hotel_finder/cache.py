import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import Event, Lock


@dataclass(frozen=True, slots=True)
class CacheResult[V]:
    value: V
    hit: bool


@dataclass(slots=True)
class _Entry[V]:
    value: V
    expires_at: float


class TTLCache[K: Hashable, V]:
    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[K, _Entry[V]] = {}
        self._inflight: dict[K, Event] = {}
        self._lock = Lock()

    def get(self, key: K) -> CacheResult[V] | None:
        with self._lock:
            return self._get_locked(key, self._clock())

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        ttl = self._validated_ttl(ttl_seconds)
        with self._lock:
            now = self._clock()
            self._evict_expired_locked(now)
            if key not in self._entries and len(self._entries) >= self._max_entries:
                earliest_key = min(
                    self._entries,
                    key=lambda candidate: self._entries[candidate].expires_at,
                )
                del self._entries[earliest_key]
            self._entries[key] = _Entry(value=value, expires_at=now + ttl)

    def get_or_load(
        self,
        key: K,
        loader: Callable[[], V],
        ttl_seconds: float | None = None,
    ) -> CacheResult[V]:
        ttl = self._validated_ttl(ttl_seconds)

        while True:
            with self._lock:
                cached = self._get_locked(key, self._clock())
                if cached is not None:
                    return cached

                pending = self._inflight.get(key)
                if pending is None:
                    pending = Event()
                    self._inflight[key] = pending
                    owner = True
                else:
                    owner = False

            if not owner:
                pending.wait()
                continue

            try:
                value = loader()
                self.set(key, value, ttl_seconds=ttl)
                return CacheResult(value=value, hit=False)
            finally:
                with self._lock:
                    completed = self._inflight.pop(key, None)
                    if completed is not None:
                        completed.set()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _get_locked(self, key: K, now: float) -> CacheResult[V] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            del self._entries[key]
            return None
        return CacheResult(value=entry.value, hit=True)

    def _evict_expired_locked(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]

    def _validated_ttl(self, ttl_seconds: float | None) -> float:
        ttl = self._ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        return ttl
