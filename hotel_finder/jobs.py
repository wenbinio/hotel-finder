import math
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Event, RLock

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_MAX_RETAINED = 8
_MAX_RETENTION_SECONDS = 30 * 60


class JobConflict(RuntimeError):
    def __init__(self, active_job_id: str) -> None:
        self.active_job_id = active_job_id
        super().__init__(f"sweep job {active_job_id} is already active")


class JobNotFound(LookupError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"unknown sweep job: {job_id}")


class _FrozenJSONDict(dict[str, object]):
    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("job snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo: dict[int, object]) -> "_FrozenJSONDict":
        return self


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("job data must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("job data must use string object keys")
        return _FrozenJSONDict((key, _freeze_json(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"job data is not JSON-safe: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SweepJobSnapshot:
    id: str
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    progress: _FrozenJSONDict
    partial: tuple[object, ...]
    result: object | None
    warnings: tuple[object, ...]
    error: object | None
    cancel_requested: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "jobId": self.id,
            "status": self.status,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "progress": _thaw_json(self.progress),
            "partial": _thaw_json(self.partial),
            "result": _thaw_json(self.result),
            "warnings": _thaw_json(self.warnings),
            "error": _thaw_json(self.error),
            "cancelRequested": self.cancel_requested,
        }


@dataclass(slots=True)
class SweepJob:
    id: str
    created_at: float
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict[str, object] = field(
        default_factory=lambda: {"completed": 0, "total": 0}
    )
    partial: list[object] = field(default_factory=list)
    result: object | None = None
    warnings: list[object] = field(default_factory=list)
    error: object | None = None
    _cancel_event: Event = field(default_factory=Event, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def set_progress(self, **progress: object) -> None:
        frozen = _freeze_json(progress)
        if not isinstance(frozen, Mapping):
            raise TypeError("progress must be a JSON object")
        with self._lock:
            self.progress = dict(frozen)

    def add_partial(self, value: object) -> None:
        frozen = _freeze_json(value)
        with self._lock:
            self.partial.append(frozen)

    def add_warning(self, value: object) -> None:
        frozen = _freeze_json(value)
        with self._lock:
            self.warnings.append(frozen)

    def snapshot(self) -> SweepJobSnapshot:
        with self._lock:
            progress = _freeze_json(self.progress)
            if not isinstance(progress, _FrozenJSONDict):
                raise TypeError("progress must be a JSON object")
            partial = _freeze_json(self.partial)
            warnings = _freeze_json(self.warnings)
            if not isinstance(partial, tuple) or not isinstance(warnings, tuple):
                raise TypeError("job collections must be JSON arrays")
            return SweepJobSnapshot(
                id=self.id,
                status=self.status,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                progress=progress,
                partial=partial,
                result=_freeze_json(self.result),
                warnings=warnings,
                error=_freeze_json(self.error),
                cancel_requested=self.cancel_requested,
            )

    def _request_cancel(self) -> None:
        self._cancel_event.set()

    def _mark_running(self, now: float) -> bool:
        with self._lock:
            if self.cancel_requested:
                self.status = "cancelled"
                self.finished_at = now
                return False
            self.status = "running"
            self.started_at = now
            return True

    def _mark_cancelled(self, now: float) -> None:
        with self._lock:
            self._cancel_event.set()
            self.status = "cancelled"
            self.result = None
            self.finished_at = now

    def _mark_completed(self, result: object, now: float) -> None:
        frozen_result = _freeze_json(result)
        with self._lock:
            if self.cancel_requested:
                self.status = "cancelled"
                self.result = None
            else:
                self.status = "completed"
                self.result = frozen_result
            self.finished_at = now

    def _mark_failed(self, exc: Exception, now: float) -> None:
        with self._lock:
            if self.cancel_requested:
                self.status = "cancelled"
                self.result = None
            else:
                self.status = "failed"
                self.error = _FrozenJSONDict(
                    {"message": str(exc), "type": type(exc).__name__}
                )
            self.finished_at = now


class SweepJobManager:
    def __init__(
        self,
        max_retained: int = _MAX_RETAINED,
        retention_seconds: float = _MAX_RETENTION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= max_retained <= _MAX_RETAINED:
            raise ValueError(f"max_retained must be between 1 and {_MAX_RETAINED}")
        if not 0 < retention_seconds <= _MAX_RETENTION_SECONDS:
            raise ValueError(
                f"retention_seconds must be between 0 and {_MAX_RETENTION_SECONDS}"
            )

        self._max_retained = max_retained
        self._retention_seconds = retention_seconds
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hotel-sweep")
        self._jobs: dict[str, SweepJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._active_id: str | None = None
        self._lock = RLock()

    def start[P](
        self,
        payload: P,
        runner: Callable[[SweepJob, P], object],
        *,
        replace: bool = False,
    ) -> SweepJob:
        with self._lock:
            self._cleanup_locked(reserve=1)
            active = self._active_job_locked()
            if active is not None:
                if not replace:
                    raise JobConflict(active.id)
                active._request_cancel()

            job_id = uuid.uuid4().hex
            while job_id in self._jobs:
                job_id = uuid.uuid4().hex
            job = SweepJob(id=job_id, created_at=self._clock())
            self._jobs[job_id] = job
            self._active_id = job_id
            self._futures[job_id] = self._executor.submit(
                self._run_job, job, payload, runner
            )
            return job

    def get(self, job_id: str) -> SweepJobSnapshot:
        with self._lock:
            self._cleanup_locked()
            return self._job_locked(job_id).snapshot()

    def cancel(self, job_id: str) -> SweepJobSnapshot:
        with self._lock:
            self._cleanup_locked()
            job = self._job_locked(job_id)
            snapshot = job.snapshot()
            if snapshot.status in _TERMINAL_STATUSES:
                return snapshot

            job._request_cancel()
            future = self._futures[job_id]
            if future.cancel():
                job._mark_cancelled(self._clock())
                if self._active_id == job_id:
                    self._active_id = None
            return job.snapshot()

    def wait(self, job_id: str, timeout: float | None = None) -> SweepJobSnapshot:
        with self._lock:
            job = self._job_locked(job_id)
            future = self._futures[job_id]
        with suppress(CancelledError):
            future.result(timeout=timeout)
        return job.snapshot()

    def cleanup(self) -> int:
        with self._lock:
            return self._cleanup_locked()

    def _run_job[P](
        self,
        job: SweepJob,
        payload: P,
        runner: Callable[[SweepJob, P], object],
    ) -> None:
        try:
            if not job._mark_running(self._clock()):
                return
            try:
                result = runner(job, payload)
                job._mark_completed(result, self._clock())
            except Exception as exc:
                job._mark_failed(exc, self._clock())
        finally:
            with self._lock:
                if self._active_id == job.id:
                    self._active_id = None
                self._cleanup_locked()

    def _active_job_locked(self) -> SweepJob | None:
        if self._active_id is None:
            return None
        active = self._jobs.get(self._active_id)
        if active is None or active.snapshot().status in _TERMINAL_STATUSES:
            self._active_id = None
            return None
        return active

    def _job_locked(self, job_id: str) -> SweepJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFound(job_id) from exc

    def _cleanup_locked(self, reserve: int = 0) -> int:
        now = self._clock()
        removed = 0
        removable = [
            job_id
            for job_id, job in self._jobs.items()
            if job_id != self._active_id
            and (snapshot := job.snapshot()).status in _TERMINAL_STATUSES
            and snapshot.finished_at is not None
            and now - snapshot.finished_at >= self._retention_seconds
        ]
        for job_id in removable:
            self._drop_job_locked(job_id)
            removed += 1

        target = max(0, self._max_retained - reserve)
        while len(self._jobs) > target:
            candidates = [
                (snapshot.finished_at, snapshot.created_at, job_id)
                for job_id, job in self._jobs.items()
                if job_id != self._active_id
                and (snapshot := job.snapshot()).status in _TERMINAL_STATUSES
            ]
            if not candidates:
                break
            _, _, oldest_id = min(candidates)
            self._drop_job_locked(oldest_id)
            removed += 1

        return removed

    def _drop_job_locked(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._futures.pop(job_id, None)
