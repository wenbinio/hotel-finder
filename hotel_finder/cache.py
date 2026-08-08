import math
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock, get_ident


@dataclass(frozen=True, slots=True)
class CacheResult[V]:
    value: V
    hit: bool


@dataclass(slots=True)
class _Entry[V]:
    value: V
    expires_at: float


@dataclass(slots=True)
class _Flight[V]:
    future: Future[V]
    owner_thread_id: int


class TTLCache[K: Hashable, V]:
    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ensure_valid_ttl(ttl_seconds)

        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[K, _Entry[V]] = {}
        self._inflight: dict[K, _Flight[V]] = {}
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

                flight = self._inflight.get(key)
                if flight is None:
                    flight = _Flight(future=Future(), owner_thread_id=get_ident())
                    self._inflight[key] = flight
                    owner = True
                else:
                    if flight.owner_thread_id == get_ident():
                        raise RuntimeError(f"recursive get_or_load for key {key!r}")
                    owner = False

            if not owner:
                return CacheResult(value=flight.future.result(), hit=True)

            try:
                value = loader()
                self.set(key, value, ttl_seconds=ttl)
            except BaseException as exc:
                with self._lock:
                    if self._inflight.get(key) is flight:
                        del self._inflight[key]
                flight.future.set_exception(exc)
                raise
            else:
                flight.future.set_result(value)
                return CacheResult(value=value, hit=False)
            finally:
                with self._lock:
                    if self._inflight.get(key) is flight:
                        del self._inflight[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            self._evict_expired_locked(self._clock())
            return len(self._entries)

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
        self._ensure_valid_ttl(ttl)
        return ttl

    @staticmethod
    def _ensure_valid_ttl(ttl: float) -> None:
        if not math.isfinite(ttl):
            raise ValueError("ttl_seconds must be finite")
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
