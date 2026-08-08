from __future__ import annotations

import logging
import threading
import time
from datetime import date

import pytest

from app import UpstreamError, create_app


def hotel(location, checkin, checkout, price=100.0):
    return {
        "name": f"{location} Test",
        "location": location,
        "url": None,
        "checkin": checkin,
        "checkout": checkout,
        "price": price,
        "rating": 4.7,
        "star_class": 5,
        "confirmation": "html",
        "amenities": ["Pool"],
        "category": "beachfront" if location == "Phuket" else "non_beachfront",
        "flight_cost": 150.0 if location == "Phuket" else 126.0,
    }


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def app_factory():
    applications = []

    def factory(**config):
        application = create_app(
            {
                "TESTING": True,
                "TODAY_PROVIDER": lambda: date(2026, 8, 9),
                **config,
            }
        )
        applications.append(application)
        return application

    yield factory

    for application in applications:
        application.extensions["hotel_finder"]["job_manager"].shutdown(
            wait=True, cancel_futures=True
        )


def start_sweep(client, **overrides):
    payload = {
        "locations": ["Bangkok", "Phuket"],
        "startDate": "2026-08-10",
        "endDate": "2026-08-20",
        "nights": 2,
        "sampleCount": 2,
        "minStars": 5,
        **overrides,
    }
    return client.post("/api/cheapest-dates", json=payload)


def test_sweep_post_returns_202_and_job_contract_immediately(app_factory):
    entered = threading.Event()
    release = threading.Event()

    def fake_search(location, checkin, checkout, min_stars=5):
        del min_stars
        entered.set()
        assert release.wait(2)
        return [hotel(location, checkin, checkout)]

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()

    response = start_sweep(client)
    assert entered.wait(1)

    assert response.status_code == 202
    assert response.json == {
        "jobId": response.json["jobId"],
        "status": "queued",
        "statusUrl": f"/api/sweeps/{response.json['jobId']}",
    }
    release.set()


def test_completed_sweep_snapshot_has_json_safe_progress_partials_and_legacy_result(
    app_factory,
):
    def fake_search(location, checkin, checkout, min_stars=5):
        del min_stars
        price = 80.0 if location == "Phuket" and checkin == "2026-08-20" else 120.0
        return [hotel(location, checkin, checkout, price)]

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    response = start_sweep(client)
    job_id = response.json["jobId"]
    manager = application.extensions["hotel_finder"]["job_manager"]

    assert manager.wait(job_id, timeout=2).status == "completed"
    snapshot = client.get(f"/api/sweeps/{job_id}")

    assert snapshot.status_code == 200
    assert snapshot.json["status"] == "completed"
    assert {
        key: snapshot.json["progress"][key]
        for key in (
            "completed",
            "currentDate",
            "destinationsCompleted",
            "total",
            "totalDates",
            "totalDestinations",
        )
    } == {
        "completed": 4,
        "currentDate": 2,
        "destinationsCompleted": 2,
        "total": 4,
        "totalDates": 2,
        "totalDestinations": 2,
    }
    assert snapshot.json["progress"]["currentLocation"] in {"Bangkok", "Phuket"}
    assert [partial["checkin"] for partial in snapshot.json["partial"]] == [
        "2026-08-10",
        "2026-08-20",
    ]
    result = snapshot.json["result"]
    assert result["locations"] == ["Bangkok", "Phuket"]
    assert result["location"] is None
    assert result["dateRange"] == {"start": "2026-08-10", "end": "2026-08-20"}
    assert result["nights"] == 2
    assert result["cheapestDate"] == {
        "checkin": "2026-08-20",
        "checkout": "2026-08-22",
        "cheapest_price": 80.0,
        "cheapest_hotel": "Phuket Test",
        "location": "Phuket",
    }
    assert result["totalBeachfront"] == 1
    assert result["totalNonBeachfront"] == 1


def test_sweep_parallelizes_at_most_four_destinations_per_date(app_factory):
    lock = threading.Lock()
    release = threading.Event()
    four_started = threading.Event()
    active = 0
    maximum = 0

    def fake_search(location, checkin, checkout, min_stars=5):
        del min_stars
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_started.set()
        assert release.wait(2)
        with lock:
            active -= 1
        return [hotel(location, checkin, checkout)]

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    response = start_sweep(
        client,
        locations=["Bangkok", "Phuket", "Bali", "Hanoi", "Colombo", "Jakarta"],
        sampleCount=1,
    )
    job_id = response.json["jobId"]

    assert four_started.wait(1)
    time.sleep(0.05)
    assert maximum == 4
    release.set()
    manager = application.extensions["hotel_finder"]["job_manager"]
    assert manager.wait(job_id, timeout=2).status == "completed"


def test_sweep_updates_progress_after_each_destination_completion(app_factory):
    bangkok_release = threading.Event()
    phuket_release = threading.Event()

    def fake_search(location, checkin, checkout, min_stars=5):
        del min_stars
        release = bangkok_release if location == "Bangkok" else phuket_release
        assert release.wait(2)
        return [hotel(location, checkin, checkout)]

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    response = start_sweep(client, sampleCount=1)
    job_id = response.json["jobId"]
    bangkok_release.set()

    deadline = time.monotonic() + 1
    progress = {"completed": 0}
    while time.monotonic() < deadline:
        progress = client.get(f"/api/sweeps/{job_id}").json["progress"]
        if progress["completed"] == 1:
            break
        time.sleep(0.01)

    assert progress["completed"] == 1
    assert progress["currentLocation"] == "Bangkok"
    phuket_release.set()


def test_sweep_cancellation_is_cooperative_and_delete_returns_202(app_factory):
    entered = threading.Event()
    release = threading.Event()

    def fake_search(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    created = start_sweep(client)
    job_id = created.json["jobId"]
    assert entered.wait(1)

    cancelled = client.delete(f"/api/sweeps/{job_id}")
    progress_at_cancel = cancelled.json["progress"]
    release.set()
    manager = application.extensions["hotel_finder"]["job_manager"]
    final = manager.wait(job_id, timeout=2)

    assert cancelled.status_code == 202
    assert cancelled.json["jobId"] == job_id
    assert cancelled.json["cancelRequested"] is True
    assert final.status == "cancelled"
    assert final.result is None
    assert final.to_dict()["progress"] == progress_at_cancel


def test_second_sweep_conflicts_while_one_is_active(app_factory):
    entered = threading.Event()
    release = threading.Event()

    def fake_search(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    first = start_sweep(client)
    assert entered.wait(1)
    second = start_sweep(client)

    assert second.status_code == 409
    assert second.json["error"]["code"] == "sweep_conflict"
    assert second.json["error"]["fields"] == {"activeJobId": first.json["jobId"]}
    release.set()


def test_unknown_sweep_get_and_delete_return_structured_404(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    client = application.test_client()

    for method in (client.get, client.delete):
        response = method("/api/sweeps/missing")
        assert response.status_code == 404
        assert response.json["error"] == {
            "code": "sweep_not_found",
            "message": "Sweep job was not found.",
            "fields": {"jobId": "missing"},
        }


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"sampleCount": 0}, "sampleCount"),
        (
            {
                "locations": [
                    "Phuket",
                    "Khao Lak",
                    "Koh Samui",
                    "Da Nang",
                    "Phu Quoc",
                    "Bali",
                    "Lombok",
                    "Langkawi",
                    "Sihanoukville",
                    "Bentota",
                    "Nha Trang",
                    "Hoi An",
                    "Bintan",
                    "Kuala Lumpur",
                    "Bangkok",
                    "Jakarta",
                    "Ho Chi Minh City",
                    "Hanoi",
                    "Colombo",
                    "Siem Reap",
                    "Phnom Penh",
                ],
                "sampleCount": 10,
            },
            "sampleCount",
        ),
        ({"startDate": "2026-08-08"}, "startDate"),
    ],
)
def test_sweep_validation_rejects_zero_excessive_work_and_past_dates(
    app_factory, overrides, field
):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    response = start_sweep(application.test_client(), **overrides)

    assert response.status_code == 400
    assert field in response.json["error"]["fields"]


def test_sweep_records_partial_failure_without_losing_success(app_factory):
    def fake_search(location, checkin, checkout, min_stars=5):
        del min_stars
        if location == "Phuket":
            raise UpstreamError("timeout", source="google", retryable=True)
        return [hotel(location, checkin, checkout)]

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    created = start_sweep(client, sampleCount=1)
    job_id = created.json["jobId"]
    manager = application.extensions["hotel_finder"]["job_manager"]
    assert manager.wait(job_id, timeout=2).status == "completed"

    snapshot = client.get(f"/api/sweeps/{job_id}").json
    assert snapshot["result"]["dates"][0]["hotel_count"] == 1
    assert snapshot["warnings"] == [
        {
            "checkin": "2026-08-10",
            "code": "timeout",
            "location": "Phuket",
            "source": "google",
        }
    ]


def test_sweep_empty_success_plus_timeout_finishes_as_failed_job(app_factory):
    def fake_search(location, *_args, **_kwargs):
        if location == "Phuket":
            raise UpstreamError("timeout", source="google", retryable=True)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    created = start_sweep(application.test_client(), sampleCount=1)
    manager = application.extensions["hotel_finder"]["job_manager"]

    snapshot = manager.wait(created.json["jobId"], timeout=2)

    assert snapshot.status == "failed"
    assert snapshot.error["code"] == "timeout"
    assert snapshot.result is None


def test_sweep_all_timeouts_finishes_as_failed_job(app_factory):
    def timeout(*_args, **_kwargs):
        raise UpstreamError("timeout", source="google", retryable=True)

    application = app_factory(SEARCH_HOTELS=timeout)
    created = start_sweep(application.test_client(), sampleCount=1)
    manager = application.extensions["hotel_finder"]["job_manager"]

    snapshot = manager.wait(created.json["jobId"], timeout=2)

    assert snapshot.status == "failed"
    assert snapshot.error["code"] == "timeout"
    assert snapshot.result is None


def test_sweep_all_genuine_empty_finishes_completed(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    created = start_sweep(application.test_client(), sampleCount=1)
    manager = application.extensions["hotel_finder"]["job_manager"]

    snapshot = manager.wait(created.json["jobId"], timeout=2)

    assert snapshot.status == "completed"
    assert snapshot.result["dates"][0]["hotel_count"] == 0
    assert snapshot.warnings == ()


def test_sweep_background_logs_preserve_initiating_request_and_job_ids(
    app_factory, caplog
):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    client = application.test_client()

    with caplog.at_level(logging.INFO, logger=application.logger.name):
        created = client.post(
            "/api/cheapest-dates",
            json={
                "locations": ["Bangkok"],
                "startDate": "2026-08-10",
                "endDate": "2026-08-11",
                "sampleCount": 1,
            },
            headers={"X-Request-ID": "sweep-request-42"},
        )
        manager = application.extensions["hotel_finder"]["job_manager"]
        assert manager.wait(created.json["jobId"], timeout=2).status == "completed"

    lifecycle = [
        record
        for record in caplog.records
        if record.getMessage() in {"sweep_started", "sweep_finished"}
    ]
    assert [record.getMessage() for record in lifecycle] == [
        "sweep_started",
        "sweep_finished",
    ]
    assert {record.request_id for record in lifecycle} == {"sweep-request-42"}
    assert {record.job_id for record in lifecycle} == {created.json["jobId"]}


def test_legacy_sweep_progress_route_reflects_manager_snapshot(app_factory):
    entered = threading.Event()
    release = threading.Event()

    def fake_search(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()
    created = start_sweep(client, sampleCount=1)
    assert entered.wait(1)

    progress = client.get("/api/sweep-progress")

    assert progress.status_code == 200
    assert progress.json["active"] is True
    assert progress.json["jobId"] == created.json["jobId"]
    assert progress.json["total_dates"] == 1
    assert progress.json["total_dests"] == 2
    release.set()


def test_job_cleanup_removes_expired_snapshot_and_health_reports_no_active_job(app_factory):
    clock = FakeClock()
    application = app_factory(
        SEARCH_HOTELS=lambda *_args, **_kwargs: [],
        CLOCK=clock,
    )
    manager = application.extensions["hotel_finder"]["job_manager"]
    client = application.test_client()
    created = start_sweep(client, sampleCount=1, locations=["Bangkok"])
    job_id = created.json["jobId"]
    assert manager.wait(job_id, timeout=2).status == "completed"

    clock.advance(1801)
    assert manager.cleanup() == 1

    assert client.get(f"/api/sweeps/{job_id}").status_code == 404
    health = client.get("/api/health")
    assert health.json["activeSweep"] is None
