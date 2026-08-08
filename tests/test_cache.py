from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from hotel_finder.cache import CacheResult, TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_get_or_load_coalesces_same_key() -> None:
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    started = Event()
    release = Event()
    calls = 0

    def loader() -> list[str]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        return ["hotel"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get_or_load, "same", loader) for _ in range(4)]
        assert started.wait(1)
        release.set()
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert [result.value for result in results] == [["hotel"]] * 4
    assert sum(not result.hit for result in results) == 1


def test_loader_exception_is_not_cached_and_wakes_waiters() -> None:
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    first_started = Event()
    release_failure = Event()
    calls_lock = Lock()
    calls = 0
    active = 0
    max_active = 0

    def loader() -> str:
        nonlocal active, calls, max_active
        with calls_lock:
            calls += 1
            attempt = calls
            active += 1
            max_active = max(max_active, active)
        try:
            if attempt == 1:
                first_started.set()
                assert release_failure.wait(2)
                raise RuntimeError("temporary failure")
            return "recovered"
        finally:
            with calls_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get_or_load, "same", loader) for _ in range(4)]
        assert first_started.wait(1)
        release_failure.set()

        failures = 0
        values = []
        for future in futures:
            try:
                values.append(future.result(timeout=2).value)
            except RuntimeError as exc:
                assert str(exc) == "temporary failure"
                failures += 1

    assert failures == 1
    assert values == ["recovered"] * 3
    assert calls == 2
    assert max_active == 1
    assert cache.get("same") == CacheResult(value="recovered", hit=True)


def test_different_keys_load_concurrently() -> None:
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    first_started = Event()
    second_started = Event()
    release = Event()

    def load_first() -> str:
        first_started.set()
        assert second_started.wait(1)
        assert release.wait(2)
        return "first"

    def load_second() -> str:
        second_started.set()
        assert first_started.wait(1)
        assert release.wait(2)
        return "second"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_load, "first", load_first)
        second = pool.submit(cache.get_or_load, "second", load_second)
        assert first_started.wait(1)
        assert second_started.wait(1)
        release.set()
        assert first.result(timeout=2).value == "first"
        assert second.result(timeout=2).value == "second"


def test_expired_entry_is_a_miss_and_reloads() -> None:
    clock = FakeClock()
    cache = TTLCache(max_entries=2, ttl_seconds=10, clock=clock)
    cache.set("hotel", "old")

    clock.advance(10)

    assert cache.get("hotel") is None
    assert cache.get_or_load("hotel", lambda: "new") == CacheResult(value="new", hit=False)


def test_capacity_evicts_expired_entries_before_live_entries() -> None:
    clock = FakeClock()
    cache = TTLCache(max_entries=2, ttl_seconds=20, clock=clock)
    cache.set("short", "gone", ttl_seconds=2)
    cache.set("long", "kept", ttl_seconds=20)

    clock.advance(3)
    cache.set("new", "also kept")

    assert cache.get("short") is None
    assert cache.get("long") == CacheResult(value="kept", hit=True)
    assert cache.get("new") == CacheResult(value="also kept", hit=True)


def test_capacity_evicts_live_entry_with_earliest_expiry() -> None:
    clock = FakeClock()
    cache = TTLCache(max_entries=2, ttl_seconds=30, clock=clock)
    cache.set("early", 1, ttl_seconds=10)
    cache.set("late", 2, ttl_seconds=20)

    cache.set("new", 3, ttl_seconds=30)

    assert cache.get("early") is None
    assert cache.get("late") == CacheResult(value=2, hit=True)
    assert cache.get("new") == CacheResult(value=3, hit=True)


def test_clear_removes_cached_values() -> None:
    cache = TTLCache(max_entries=2, ttl_seconds=10)
    cache.set("hotel", "cached")

    cache.clear()

    assert cache.get("hotel") is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_entries": 0, "ttl_seconds": 1}, "max_entries"),
        ({"max_entries": 1, "ttl_seconds": 0}, "ttl_seconds"),
    ],
)
def test_cache_rejects_non_positive_bounds(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TTLCache(**kwargs)


def test_set_rejects_non_positive_ttl_override() -> None:
    cache = TTLCache(max_entries=2, ttl_seconds=10)

    with pytest.raises(ValueError, match="ttl_seconds"):
        cache.set("hotel", "value", ttl_seconds=0)
