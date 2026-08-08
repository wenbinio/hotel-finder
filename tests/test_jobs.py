import json
from threading import Event, Lock

import pytest

from hotel_finder.jobs import JobConflict, JobNotFound, SweepJobManager


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_only_one_active_job_and_cancel_is_cooperative() -> None:
    entered = Event()
    release = Event()

    def runner(job, payload):
        entered.set()
        assert release.wait(2)
        if job.cancel_requested:
            return None
        return {"ok": True, "payload": payload}

    manager = SweepJobManager(max_retained=8, retention_seconds=1800)
    first = manager.start({"query": 1}, runner)
    assert entered.wait(1)

    with pytest.raises(JobConflict):
        manager.start({"query": 2}, runner)

    manager.cancel(first.id)
    release.set()

    assert manager.wait(first.id, timeout=2).status == "cancelled"


def test_cancel_request_wins_over_runner_result() -> None:
    entered = Event()
    release = Event()

    def runner(_job, _payload):
        entered.set()
        assert release.wait(2)
        return {"published": "too late"}

    manager = SweepJobManager()
    job = manager.start({}, runner)
    assert entered.wait(1)
    manager.cancel(job.id)
    release.set()

    snapshot = manager.wait(job.id, timeout=2)
    assert snapshot.status == "cancelled"
    assert snapshot.result is None


def test_failed_job_releases_active_slot_and_records_json_safe_error() -> None:
    def fail(_job, _payload):
        raise RuntimeError("upstream broke")

    manager = SweepJobManager()
    failed = manager.start({}, fail)
    failed_snapshot = manager.wait(failed.id, timeout=2)

    assert failed_snapshot.status == "failed"
    assert failed_snapshot.error == {
        "message": "upstream broke",
        "type": "RuntimeError",
    }

    replacement = manager.start({}, lambda _job, _payload: {"ok": True})
    assert manager.wait(replacement.id, timeout=2).status == "completed"


def test_non_json_runner_result_fails_cleanly_and_releases_active_slot() -> None:
    manager = SweepJobManager()
    invalid = manager.start({}, lambda _job, _payload: object())

    snapshot = manager.wait(invalid.id, timeout=2)

    assert snapshot.status == "failed"
    assert snapshot.error == {
        "message": "job data is not JSON-safe: object",
        "type": "TypeError",
    }
    replacement = manager.start({}, lambda _job, _payload: None)
    assert manager.wait(replacement.id, timeout=2).status == "completed"


def test_snapshot_is_deeply_immutable_detached_and_json_safe() -> None:
    returned_result = {"rows": [{"price": 120}], "dates": ["2026-08-09"]}

    def runner(job, _payload):
        job.set_progress(completed=1, total=2)
        job.add_partial({"date": "2026-08-09", "price": 120})
        job.add_warning({"code": "partial", "location": "Phuket"})
        return returned_result

    manager = SweepJobManager()
    job = manager.start({}, runner)
    snapshot = manager.wait(job.id, timeout=2)
    returned_result["rows"][0]["price"] = 999

    assert snapshot.progress == {"completed": 1, "total": 2}
    assert snapshot.partial == ({"date": "2026-08-09", "price": 120},)
    assert snapshot.warnings == ({"code": "partial", "location": "Phuket"},)
    assert snapshot.result == {"rows": ({"price": 120},), "dates": ("2026-08-09",)}
    assert json.loads(json.dumps(snapshot.to_dict()))["result"]["rows"][0]["price"] == 120

    with pytest.raises(TypeError):
        snapshot.progress["completed"] = 99
    with pytest.raises(TypeError):
        snapshot.result["rows"][0]["price"] = 99
    with pytest.raises(AttributeError):
        snapshot.partial.append({"date": "other"})
    with pytest.raises((AttributeError, TypeError)):
        snapshot.status = "running"


def test_replace_requests_cancellation_without_overlapping_runners() -> None:
    first_entered = Event()
    release_first = Event()
    active_lock = Lock()
    active = 0
    max_active = 0

    def runner(job, payload):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if payload["query"] == 1:
                first_entered.set()
                assert release_first.wait(2)
            if job.cancel_requested:
                return None
            return {"query": payload["query"]}
        finally:
            with active_lock:
                active -= 1

    manager = SweepJobManager()
    first = manager.start({"query": 1}, runner)
    assert first_entered.wait(1)

    replacement = manager.start({"query": 2}, runner, replace=True)
    assert first.cancel_requested
    with pytest.raises(JobConflict):
        manager.start({"query": 3}, runner)

    release_first.set()

    assert manager.wait(first.id, timeout=2).status == "cancelled"
    replacement_snapshot = manager.wait(replacement.id, timeout=2)
    assert replacement_snapshot.status == "completed"
    assert replacement_snapshot.result == {"query": 2}
    assert max_active == 1


def test_cancelling_queued_replacement_prevents_its_runner() -> None:
    first_entered = Event()
    release_first = Event()
    calls = []

    def runner(job, payload):
        calls.append(payload["query"])
        if payload["query"] == 1:
            first_entered.set()
            assert release_first.wait(2)
        if job.cancel_requested:
            return None
        return payload

    manager = SweepJobManager()
    first = manager.start({"query": 1}, runner)
    assert first_entered.wait(1)
    queued = manager.start({"query": 2}, runner, replace=True)

    queued_snapshot = manager.cancel(queued.id)
    release_first.set()

    assert queued_snapshot.status == "cancelled"
    assert manager.wait(first.id, timeout=2).status == "cancelled"
    assert manager.wait(queued.id, timeout=2).status == "cancelled"
    assert calls == [1]


def test_cleanup_enforces_count_and_retention_bounds() -> None:
    clock = FakeClock()
    manager = SweepJobManager(max_retained=2, retention_seconds=10, clock=clock)
    job_ids = []

    for index in range(3):
        job = manager.start({"index": index}, lambda _job, payload: payload)
        assert manager.wait(job.id, timeout=2).status == "completed"
        job_ids.append(job.id)
        clock.advance(1)

    with pytest.raises(JobNotFound):
        manager.get(job_ids[0])
    assert manager.get(job_ids[1]).status == "completed"
    assert manager.get(job_ids[2]).status == "completed"

    clock.advance(11)
    assert manager.cleanup() == 2
    with pytest.raises(JobNotFound):
        manager.get(job_ids[1])
    with pytest.raises(JobNotFound):
        manager.get(job_ids[2])


def test_cleanup_never_removes_running_job() -> None:
    clock = FakeClock()
    entered = Event()
    release = Event()

    def runner(_job, _payload):
        entered.set()
        assert release.wait(2)
        return None

    manager = SweepJobManager(max_retained=1, retention_seconds=1, clock=clock)
    job = manager.start({}, runner)
    assert entered.wait(1)
    clock.advance(10)

    assert manager.cleanup() == 0
    assert manager.get(job.id).status == "running"
    release.set()
    assert manager.wait(job.id, timeout=2).status == "completed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retained": 0}, "max_retained"),
        ({"max_retained": 9}, "max_retained"),
        ({"retention_seconds": 0}, "retention_seconds"),
        ({"retention_seconds": 1801}, "retention_seconds"),
    ],
)
def test_manager_rejects_retention_outside_personal_runtime_bounds(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SweepJobManager(**kwargs)


def test_unknown_job_operations_are_explicit() -> None:
    manager = SweepJobManager()

    with pytest.raises(JobNotFound):
        manager.get("missing")
    with pytest.raises(JobNotFound):
        manager.cancel("missing")
    with pytest.raises(JobNotFound):
        manager.wait("missing", timeout=0)
