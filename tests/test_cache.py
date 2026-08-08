import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier, Event, Lock, Thread

import pytest

import hotel_finder.cache as cache_module
from hotel_finder.cache import CacheDeadlineExceeded, CacheResult, TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.values)


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


def test_get_or_load_not_after_allows_boundary_with_one_publication_sample() -> None:
    clock = SequenceClock([0.0, 10.0, 11.0])
    cache = TTLCache(max_entries=8, ttl_seconds=60, clock=clock)

    loaded = cache.get_or_load("same", lambda: "on-time", not_after=10.0)

    assert loaded == CacheResult(value="on-time", hit=False)
    assert clock.calls == 2
    assert cache.get("same") == CacheResult(value="on-time", hit=True)


def test_deadline_after_uses_the_cache_clock_domain() -> None:
    clock = FakeClock()
    clock.advance(57_000)
    cache = TTLCache(max_entries=8, ttl_seconds=60, clock=clock)

    assert cache.deadline_after(15) == 57_015


def test_get_or_load_deadline_failure_is_shared_then_retryable(monkeypatch) -> None:
    waiter_joined = Event()

    class WaiterAwareFuture(Future):
        def result(self, timeout=None):
            waiter_joined.set()
            return super().result(timeout=timeout)

    monkeypatch.setattr(cache_module, "Future", WaiterAwareFuture)
    clock = FakeClock()
    cache = TTLCache(max_entries=8, ttl_seconds=60, clock=clock)
    owner_started = Event()
    release_owner = Event()
    calls = 0

    def late_loader() -> str:
        nonlocal calls
        calls += 1
        owner_started.set()
        assert release_owner.wait(2)
        return "late"

    def capture_deadline() -> CacheDeadlineExceeded:
        try:
            cache.get_or_load("same", late_loader, not_after=0.0)
        except CacheDeadlineExceeded as error:
            return error
        raise AssertionError("late cache publication succeeded")

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(capture_deadline)
        assert owner_started.wait(1)
        waiter = pool.submit(capture_deadline)
        assert waiter_joined.wait(1)
        clock.advance(1)
        release_owner.set()
        errors = [owner.result(timeout=2), waiter.result(timeout=2)]

    assert calls == 1
    assert errors[0] is errors[1]
    assert errors[0].not_after == 0.0
    assert errors[0].observed_at == 1.0
    assert cache.get("same") is None
    assert cache.get_or_load(
        "same", lambda: "recovered", not_after=1.0
    ) == CacheResult(value="recovered", hit=False)


def test_persistent_failure_is_shared_once_by_the_waiting_cohort_then_retryable() -> None:
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    owner_started = Event()
    release_failure = Event()
    calls_lock = Lock()
    calls = 0

    def loader() -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
            attempt = calls
        owner_started.set()
        assert release_failure.wait(2)
        raise RuntimeError(f"failure {attempt}")

    def capture_failure() -> RuntimeError:
        try:
            cache.get_or_load("same", loader)
        except RuntimeError as exc:
            return exc
        raise AssertionError("loader failure was not propagated")

    with ThreadPoolExecutor(max_workers=8) as pool:
        owner = pool.submit(capture_failure)
        assert owner_started.wait(1)
        waiters_ready = Barrier(8)

        def wait_for_owner() -> RuntimeError:
            waiters_ready.wait(timeout=1)
            return capture_failure()

        waiters = [pool.submit(wait_for_owner) for _ in range(7)]
        waiters_ready.wait(timeout=1)
        time.sleep(0.05)
        release_failure.set()
        errors = [owner.result(timeout=2)] + [future.result(timeout=2) for future in waiters]

    assert calls == 1
    assert len({id(error) for error in errors}) == 1
    assert str(errors[0]) == "failure 1"
    assert cache.get("same") is None
    assert cache.get_or_load("same", lambda: "recovered") == CacheResult(
        value="recovered", hit=False
    )


def test_failed_flight_is_removed_before_a_woken_waiter_retries(monkeypatch) -> None:
    failure_published = Event()
    allow_owner_finish = Event()
    waiter_joined = Event()

    class PausingFuture(Future):
        def result(self, timeout=None):
            waiter_joined.set()
            return super().result(timeout=timeout)

        def set_exception(self, exception):
            super().set_exception(exception)
            failure_published.set()
            assert allow_owner_finish.wait(2)

    monkeypatch.setattr(cache_module, "Future", PausingFuture)
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    owner_started = Event()
    release_failure = Event()
    retry_called = Event()

    def failing_loader() -> str:
        owner_started.set()
        assert release_failure.wait(2)
        raise RuntimeError("owner failed")

    def retry_loader() -> str:
        retry_called.set()
        return "recovered"

    def wait_then_retry() -> str:
        try:
            cache.get_or_load("same", failing_loader)
        except RuntimeError as exc:
            assert str(exc) == "owner failed"
        return cache.get_or_load("same", retry_loader).value

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cache.get_or_load, "same", failing_loader)
        assert owner_started.wait(1)
        waiter = pool.submit(wait_then_retry)
        assert waiter_joined.wait(1)

        try:
            release_failure.set()
            assert failure_published.wait(1)
            retry_observed_before_owner_finished = retry_called.wait(0.5)
        finally:
            allow_owner_finish.set()

        with pytest.raises(RuntimeError, match="owner failed"):
            owner.result(timeout=2)
        waiter_error = None
        try:
            waiter_result = waiter.result(timeout=2)
        except RuntimeError as exc:
            waiter_result = None
            waiter_error = exc

    assert retry_observed_before_owner_finished
    assert waiter_error is None
    assert waiter_result == "recovered"


def test_recursive_same_key_load_raises_instead_of_deadlocking() -> None:
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    errors: list[BaseException] = []

    def recurse() -> None:
        try:
            cache.get_or_load(
                "same",
                lambda: cache.get_or_load("same", lambda: "nested").value,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=recurse, daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    assert not thread.is_alive(), "recursive load deadlocked"
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "recursive" in str(errors[0]).lower()


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


def test_size_purges_expired_entries() -> None:
    clock = FakeClock()
    cache = TTLCache(max_entries=3, ttl_seconds=20, clock=clock)
    cache.set("short", 1, ttl_seconds=2)
    cache.set("long", 2, ttl_seconds=20)

    clock.advance(3)

    assert cache.size() == 1
    assert cache.get("short") is None
    assert cache.get("long") == CacheResult(value=2, hit=True)


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


@pytest.mark.parametrize("ttl_seconds", [float("nan"), float("inf"), float("-inf")])
def test_cache_rejects_non_finite_default_ttl(ttl_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TTLCache(max_entries=2, ttl_seconds=ttl_seconds)


@pytest.mark.parametrize("ttl_seconds", [float("nan"), float("inf"), float("-inf")])
def test_cache_rejects_non_finite_ttl_override(ttl_seconds: float) -> None:
    cache = TTLCache(max_entries=2, ttl_seconds=10)
    loader_called = Event()

    with pytest.raises(ValueError, match="finite"):
        cache.set("hotel", "value", ttl_seconds=ttl_seconds)
    with pytest.raises(ValueError, match="finite"):
        cache.get_or_load("hotel", lambda: loader_called.set(), ttl_seconds=ttl_seconds)
    assert not loader_called.is_set()
