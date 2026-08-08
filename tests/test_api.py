from __future__ import annotations

import json
import logging
from datetime import date
from threading import Event

import pytest

import app as app_module
from app import UpstreamError, create_app, run_bounded

VALID_HOTEL = {
    "name": "Bangkok Test",
    "location": "Bangkok",
    "url": "https://www.google.com/travel/hotels/entity/test",
    "checkin": "2026-08-10",
    "checkout": "2026-08-14",
    "price": 100.0,
    "rating": 4.7,
    "star_class": 5,
    "confirmation": "html",
    "amenities": ["Pool", "Spa"],
    "category": "non_beachfront",
    "flight_cost": 126.0,
}

DEFAULT_UPSTREAM_HTML = r"""
<html><body><div class="uaTTDe">
  <h2 class="BgYkof">Default Boundary Hotel</h2>
  <span class="KFi5wf lA0BZ">4.7</span>
  <span class="ne5qie Ih19Ad">5-star hotel</span>
  <span class="LtjZ2d">Pool</span><span>$220</span>
  <a href="/travel/hotels/entity/default-boundary">open</a>
</div>Agoda \u0024180</body></html>
"""


class StaticHTMLResponse:
    status_code = 200
    text = DEFAULT_UPSTREAM_HTML
    headers = {"content-type": "text/html; charset=utf-8"}


class StaticLowLevelClient:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return StaticHTMLResponse()


class StaticJSONResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    def __init__(self):
        self.payload = {
            "result": {
                "rates": [
                    {"name": "Agoda", "code": "Agoda", "rate": 88.0, "tax": 0}
                ]
            }
        }
        self.body = json.dumps(self.payload).encode("utf-8")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=1):
        size = (len(self.body) or 1) if chunk_size is None else chunk_size
        for offset in range(0, len(self.body), size):
            yield self.body[offset : offset + size]

    def close(self):
        return None


class StaticXoteloClient:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return StaticJSONResponse()


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


@pytest.fixture
def client(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    return application.test_client()


def test_malformed_json_returns_structured_400(client):
    response = client.post(
        "/api/search",
        data='{"location":',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json == {
        "error": {
            "code": "invalid_json",
            "message": "Request body must be valid JSON.",
            "fields": {"body": "must be a JSON object"},
        }
    }


def test_search_uses_injected_current_day_and_explicit_iso_serialization(app_factory):
    calls = []

    def fake_search(location, checkin, checkout, min_stars=5):
        calls.append((location, checkin, checkout, min_stars))
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    client = application.test_client()

    past = client.post(
        "/api/search",
        json={
            "location": "Bangkok",
            "checkin": "2026-08-08",
            "checkout": "2026-08-10",
        },
    )
    current = client.post(
        "/api/search",
        json={
            "location": "Bangkok",
            "checkin": "2026-08-09",
            "checkout": "2026-08-14",
            "minStars": 4,
        },
    )

    assert past.status_code == 400
    assert past.json["error"]["fields"] == {"checkin": "must not be in the past"}
    assert current.status_code == 200
    assert current.json["checkin"] == "2026-08-09"
    assert current.json["checkout"] == "2026-08-14"
    assert calls == [("Bangkok", "2026-08-09", "2026-08-14", 4)]


def test_search_all_returns_partial_results_and_warning(app_factory, monkeypatch):
    def fake_search(location, *_args, **_kwargs):
        if location == "Phuket":
            raise UpstreamError("timeout", source="google", retryable=True)
        return [
            {
                **VALID_HOTEL,
                "name": f"{location} Test",
                "location": location,
                "category": (
                    "beachfront" if location == "Phuket" else "non_beachfront"
                ),
            }
        ]

    monkeypatch.setattr(app_module, "search_hotels", fake_search)
    application = app_factory()
    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-09",
            "checkout": "2026-08-14",
            "destinations": ["Bangkok", "Phuket"],
            "minStars": 5,
            "maxFlight": 300,
        },
    )

    assert response.status_code == 200
    assert response.json["totalNonBeachfront"] == 1
    assert response.json["totalBeachfront"] == 0
    assert response.json["failedDestinations"] == [
        {"location": "Phuket", "code": "timeout"}
    ]
    assert response.json["warnings"] == [
        {"code": "timeout", "location": "Phuket", "source": "google"}
    ]


def test_search_all_default_boundary_works_in_executor_without_flask_context(
    app_factory,
):
    upstream = StaticLowLevelClient()
    application = app_factory(CLIENT_FACTORY=lambda: upstream)

    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
            "destinations": ["Bangkok"],
        },
    )

    assert response.status_code == 200
    assert response.json["totalNonBeachfront"] == 1
    assert response.json["results"]["non_beachfront"][0]["name"] == (
        "Default Boundary Hotel"
    )
    assert response.json["warnings"] == []


def test_compare_default_boundary_works_in_executor_without_flask_context(
    app_factory,
):
    upstream = StaticLowLevelClient()
    xotelo = StaticXoteloClient()
    application = app_factory(
        CLIENT_FACTORY=lambda: upstream,
        XOTELO_CLIENT=xotelo,
    )
    hotel = {
        **VALID_HOTEL,
        "name": "Pullman Bangkok Hotel G",
        "url": "https://www.google.com/travel/hotels/entity/default-boundary",
    }

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": [hotel],
        },
    )

    assert response.status_code == 200
    assert response.json["hotels"][0]["providers"] == {
        "agoda": {
            "rate": 88.0,
            "url": (
                "https://www.agoda.com/search?q=Pullman+Bangkok+Hotel+G"
                "&checkIn=2026-08-10&los=1"
            ),
        }
    }
    assert response.json["failedHotels"] == []
    assert response.json["warnings"] == []
    assert len(upstream.calls) == 1
    assert len(xotelo.calls) == 1


def test_search_all_filters_requested_destinations_by_flight_budget(app_factory):
    calls = []

    def fake_search(location, *_args, **_kwargs):
        calls.append(location)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
            "destinations": ["Bangkok", "Phuket"],
            "maxFlight": 130,
        },
    )

    assert response.status_code == 200
    assert calls == ["Bangkok"]


def test_search_all_empty_success_plus_timeout_is_not_reported_as_empty_inventory(
    app_factory,
):
    def fake_search(location, *_args, **_kwargs):
        if location == "Phuket":
            raise UpstreamError("timeout", source="google", retryable=True)
        return []

    application = app_factory(SEARCH_HOTELS=fake_search)
    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
            "destinations": ["Bangkok", "Phuket"],
        },
    )

    assert response.status_code == 504
    assert response.json["error"]["code"] == "timeout"


def test_search_all_all_genuine_empty_is_a_success(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
            "destinations": ["Bangkok", "Phuket"],
        },
    )

    assert response.status_code == 200
    assert response.json["totalBeachfront"] == 0
    assert response.json["totalNonBeachfront"] == 0
    assert response.json["failedDestinations"] == []


def test_search_all_all_timeouts_returns_gateway_timeout(app_factory):
    def timeout(*_args, **_kwargs):
        raise UpstreamError("timeout", source="google", retryable=True)

    application = app_factory(SEARCH_HOTELS=timeout)
    response = application.test_client().post(
        "/api/search-all",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
            "destinations": ["Bangkok", "Phuket"],
        },
    )

    assert response.status_code == 504
    assert response.json["error"]["code"] == "timeout"


def test_compare_requires_top_level_dates_and_preserves_validated_metadata(app_factory):
    application = app_factory(
        PROVIDER_PRICES=lambda *_args, **_kwargs: {"agoda": 90.0},
        XOTELO_PRICES=lambda *_args, **_kwargs: {},
    )
    client = application.test_client()

    missing_dates = client.post("/api/compare-prices", json={"hotels": [VALID_HOTEL]})
    response = client.post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": [VALID_HOTEL],
        },
    )

    assert missing_dates.status_code == 400
    assert missing_dates.json["error"]["fields"] == {
        "checkin": "must be a non-empty string"
    }
    assert response.status_code == 200
    hotel = response.json["hotels"][0]
    assert hotel["name"] == "Bangkok Test"
    assert hotel["checkin"] == "2026-08-10"
    assert hotel["checkout"] == "2026-08-14"
    assert hotel["amenities"] == ["Pool", "Spa"]
    assert hotel["providers"] == {"agoda": {"rate": 90.0, "url": ""}}


def test_compare_isolates_one_failed_future_and_keeps_input_hotel(app_factory):
    def fake_provider(url):
        if url.endswith("/broken"):
            raise UpstreamError("timeout", source="google_provider", retryable=True)
        return {"booking.com": 95.0}

    application = app_factory(
        PROVIDER_PRICES=fake_provider,
        XOTELO_PRICES=lambda *_args, **_kwargs: {},
    )
    hotels = [
        {**VALID_HOTEL, "name": "Good", "url": VALID_HOTEL["url"] + "/good"},
        {**VALID_HOTEL, "name": "Broken", "url": VALID_HOTEL["url"] + "/broken"},
    ]

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": hotels,
        },
    )

    assert response.status_code == 200
    by_name = {hotel["name"]: hotel for hotel in response.json["hotels"]}
    assert by_name["Good"]["providers"]["booking.com"]["rate"] == 95.0
    assert by_name["Broken"]["providers"] == {}
    assert response.json["failedHotels"] == [
        {"name": "Broken", "code": "timeout", "source": "google_provider"}
    ]


def test_compare_isolates_google_and_xotelo_failures_within_each_hotel(app_factory):
    def google_prices(url):
        if url.endswith("/xotelo-only"):
            raise UpstreamError(
                "timeout", source="google_provider", retryable=True
            )
        return {"booking.com": 91.0}

    def xotelo_prices(_key, hotel_name, *_dates):
        if hotel_name == "Google Only":
            raise UpstreamError("timeout", source="xotelo", retryable=True)
        return {"Agoda": {"rate": 88.0, "tax": 0, "url": "https://agoda.test"}}

    application = app_factory(
        PROVIDER_PRICES=google_prices,
        XOTELO_PRICES=xotelo_prices,
        RESOLVE_TRIPADVISOR=lambda *_args: "ta-key",
    )
    hotels = [
        {
            **VALID_HOTEL,
            "name": "Google Only",
            "url": VALID_HOTEL["url"] + "/google-only",
        },
        {
            **VALID_HOTEL,
            "name": "Xotelo Only",
            "url": VALID_HOTEL["url"] + "/xotelo-only",
        },
    ]

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": hotels,
        },
    )

    assert response.status_code == 200
    by_name = {hotel["name"]: hotel for hotel in response.json["hotels"]}
    assert by_name["Google Only"]["providers"] == {
        "booking.com": {"rate": 91.0, "url": ""}
    }
    assert by_name["Xotelo Only"]["providers"] == {
        "agoda": {"rate": 88.0, "url": "https://agoda.test"}
    }
    assert response.json["warnings"] == [
        {"name": "Google Only", "code": "timeout", "source": "xotelo"},
        {
            "name": "Xotelo Only",
            "code": "timeout",
            "source": "google_provider",
        },
    ]


def test_compare_preserves_google_rates_when_xotelo_key_lookup_fails(app_factory):
    def fail_lookup(*_args):
        raise RuntimeError("lookup unavailable")

    application = app_factory(
        PROVIDER_PRICES=lambda _url: {"booking.com": 91.0},
        RESOLVE_TRIPADVISOR=fail_lookup,
    )

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": [VALID_HOTEL],
        },
    )

    assert response.status_code == 200
    assert response.json["hotels"][0]["providers"] == {
        "booking.com": {"rate": 91.0, "url": ""}
    }
    assert response.json["failedHotels"] == []
    assert response.json["warnings"] == [
        {
            "name": "Bangkok Test",
            "code": "upstream_failure",
            "source": "xotelo",
        }
    ]


def test_compare_top_level_dates_override_stale_hotel_dates(app_factory):
    google_calls = []
    xotelo_calls = []

    def google_prices(_url, *, checkin, checkout):
        google_calls.append((checkin, checkout))
        return {}

    def xotelo_prices(_key, _name, checkin, checkout):
        xotelo_calls.append((checkin, checkout))
        return {}

    application = app_factory(
        PROVIDER_PRICES=google_prices,
        XOTELO_PRICES=xotelo_prices,
        RESOLVE_TRIPADVISOR=lambda *_args: "ta-key",
    )
    stale = {
        **VALID_HOTEL,
        "checkin": "2026-09-01",
        "checkout": "2026-09-02",
    }

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": [stale],
        },
    )

    assert response.status_code == 200
    assert google_calls == [("2026-08-10", "2026-08-14")]
    assert xotelo_calls == [("2026-08-10", "2026-08-14")]
    assert response.json["hotels"][0]["checkin"] == "2026-08-10"
    assert response.json["hotels"][0]["checkout"] == "2026-08-14"


def test_compare_rejects_unsupported_google_url_before_provider_call(app_factory):
    provider_called = Event()
    application = app_factory(
        PROVIDER_PRICES=lambda *_args, **_kwargs: provider_called.set(),
        XOTELO_PRICES=lambda *_args, **_kwargs: {},
    )
    hotel = {**VALID_HOTEL, "url": "https://evil.example/travel/hotels/entity/x"}

    response = application.test_client().post(
        "/api/compare-prices",
        json={
            "checkin": "2026-08-10",
            "checkout": "2026-08-14",
            "hotels": [hotel],
        },
    )

    assert response.status_code == 400
    assert response.json["error"]["fields"] == {
        "hotels[0].url": "must be an HTTPS www.google.com hotel entity URL"
    }
    assert not provider_called.is_set()


def test_cors_is_disabled_by_default_and_exact_when_configured(app_factory):
    disabled = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    configured = app_factory(
        SEARCH_HOTELS=lambda *_args, **_kwargs: [],
        ALLOWED_ORIGINS="https://hotel.lan, https://private.example",
    )

    no_cors = disabled.test_client().get(
        "/api/destinations", headers={"Origin": "https://hotel.lan"}
    )
    rejected = configured.test_client().get(
        "/api/destinations", headers={"Origin": "https://evil.example"}
    )
    allowed = configured.test_client().get(
        "/api/destinations", headers={"Origin": "https://hotel.lan"}
    )
    static = configured.test_client().get(
        "/", headers={"Origin": "https://hotel.lan"}
    )

    assert "Access-Control-Allow-Origin" not in no_cors.headers
    assert "Access-Control-Allow-Origin" not in rejected.headers
    assert allowed.headers["Access-Control-Allow-Origin"] == "https://hotel.lan"
    assert "Access-Control-Allow-Origin" not in static.headers


def test_request_id_is_bounded_validated_and_returned_on_errors(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    client = application.test_client()

    accepted = client.get(
        "/api/destinations", headers={"X-Request-ID": "private-search_42"}
    )
    replaced = client.post(
        "/api/search",
        data="{",
        content_type="application/json",
        headers={"X-Request-ID": "bad id " + "x" * 100},
    )

    assert accepted.headers["X-Request-ID"] == "private-search_42"
    generated = replaced.headers["X-Request-ID"]
    assert generated != "bad id " + "x" * 100
    assert 1 <= len(generated) <= 64


def test_application_logger_emits_structured_info_records_by_default(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    capture = CaptureHandler(level=logging.INFO)
    application.logger.addHandler(capture)
    try:
        application.test_client().get(
            "/api/destinations",
            headers={"X-Request-ID": "local-info-42"},
        )
    finally:
        application.logger.removeHandler(capture)

    assert application.logger.getEffectiveLevel() == logging.INFO
    request_record = next(
        record for record in records if record.getMessage() == "request_complete"
    )
    formatter = application.logger.handlers[0].formatter
    assert formatter is not None
    rendered = json.loads(formatter.format(request_record))
    assert rendered["event"] == "request_complete"
    assert rendered["request_id"] == "local-info-42"
    assert rendered["response_status"] == 200


def test_health_uses_cache_and_job_public_snapshots(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    response = application.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.json == {
        "status": "ok",
        "version": "0.1.0",
        "primp": {
            "profile": "chrome_146",
            "verify": True,
            "connectTimeoutSeconds": 5,
            "readTimeoutSeconds": 12,
            "totalTimeoutSeconds": 15,
            "followRedirects": False,
        },
        "caches": {"search": 0, "providers": 0, "xotelo": 0},
        "activeSweep": None,
        "runtime": {"workers": 1, "upstreamConcurrency": 4},
    }


def test_upstream_error_handler_uses_504_for_timeouts(app_factory):
    def fail(*_args, **_kwargs):
        raise UpstreamError("timeout", source="google", retryable=True)

    application = app_factory(SEARCH_HOTELS=fail)
    response = application.test_client().post(
        "/api/search",
        json={
            "location": "Bangkok",
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
        },
    )

    assert response.status_code == 504
    assert response.json["error"] == {
        "code": "timeout",
        "message": "The google request timed out.",
        "fields": {"retryable": True, "source": "google"},
    }


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_search_never_serializes_nonfinite_upstream_numbers(app_factory, nonfinite):
    application = app_factory(
        SEARCH_HOTELS=lambda *_args, **_kwargs: [
            {**VALID_HOTEL, "rating": nonfinite}
        ]
    )

    response = application.test_client().post(
        "/api/search",
        json={
            "location": "Bangkok",
            "checkin": "2026-08-10",
            "checkout": "2026-08-11",
        },
    )

    assert response.status_code == 502
    assert response.json["error"]["code"] == "unexpected_content"
    assert b"NaN" not in response.data
    assert b"Infinity" not in response.data


def test_spa_and_unknown_api_routes_are_not_shadowed_by_flask_static(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    client = application.test_client()

    spa = client.get("/saved-searches/private")
    missing = client.get("/api/nope")
    wrong_method = client.post("/api/destinations")

    assert spa.status_code == 200
    assert b'<div id="root"></div>' in spa.data
    assert missing.status_code == 404
    assert missing.is_json
    assert missing.json["error"]["code"] == "not_found"
    assert wrong_method.status_code == 405
    assert wrong_method.is_json
    assert wrong_method.json["error"]["code"] == "method_not_allowed"


def test_exact_api_root_never_falls_through_to_spa(app_factory):
    application = app_factory(SEARCH_HOTELS=lambda *_args, **_kwargs: [])
    client = application.test_client()

    missing = client.get("/api")
    wrong_method = client.post("/api")

    assert missing.status_code == 404
    assert missing.is_json
    assert missing.json["error"]["code"] == "not_found"
    assert wrong_method.status_code == 405
    assert wrong_method.is_json
    assert wrong_method.json["error"]["code"] == "method_not_allowed"


def test_run_bounded_preserves_input_order_and_reports_upstream_failures():
    def operation(item):
        if item == 2:
            raise UpstreamError("blocked", source="google", retryable=False)
        return item * 10

    successes, failures = run_bounded([3, 2, 1], operation, max_workers=2)

    assert successes == [(3, 30), (1, 10)]
    assert [(item, error.code) for item, error in failures] == [(2, "blocked")]
