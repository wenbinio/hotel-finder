from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

import app as app_module
from app import UpstreamError, call_with_retry, create_app, fetch_provider_prices, search_hotels

HOTEL_HTML = """
<html><body><div class="uaTTDe">
  <h2 class="BgYkof">Test Grand Hotel</h2>
  <span class="KFi5wf lA0BZ">4.7</span>
  <span class="ne5qie Ih19Ad">5-star hotel</span>
  <span class="LtjZ2d">Pool</span><span>$220</span>
  <a href="/travel/hotels/entity/test">open</a>
</div></body></html>
"""


class FakeResponse:
    def __init__(self, status_code=200, text=HOTEL_HTML, headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        next_value = self.responses.pop(0)
        if isinstance(next_value, BaseException):
            raise next_value
        return next_value


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class DeterministicRng:
    def __init__(self, value=0.1):
        self.value = value
        self.calls = []

    def uniform(self, minimum, maximum):
        self.calls.append((minimum, maximum))
        return self.value


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


def test_make_client_uses_thread_local_hardened_primp_configuration(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(app_module, "Client", FakeClient)
    monkeypatch.setattr(app_module, "_client_local", threading.local())

    first = app_module.make_client()
    second = app_module.make_client()

    assert first is second
    assert created == [
        {
            "impersonate": "chrome_146",
            "verify": True,
            "connect_timeout": 5,
            "read_timeout": 12,
            "timeout": 15,
            "follow_redirects": False,
        }
    ]


def test_search_parses_cards_and_nulls_non_entity_links(app_factory):
    html = HOTEL_HTML.replace(
        'href="/travel/hotels/entity/test"', 'href="https://evil.example/hotel"'
    )
    client = SequenceClient([FakeResponse(text=html)])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        hotels = search_hotels(" Bangkok ", "2026-08-10", "2026-08-11", 5)

    assert hotels[0]["name"] == "Test Grand Hotel"
    assert hotels[0]["url"] is None
    assert hotels[0]["location"] == "Bangkok"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_calls"),
    [(404, "upstream_response", 1), (429, "rate_limited", 2), (503, "upstream_unavailable", 2)],
)
def test_search_non_200_responses_raise_and_only_transient_statuses_retry(
    app_factory, status_code, expected_code, expected_calls
):
    responses = [FakeResponse(status_code=status_code)] * expected_calls
    client = SequenceClient(responses)
    sleeps = []
    application = app_factory(
        CLIENT_FACTORY=lambda: client,
        SLEEP=sleeps.append,
        RNG=DeterministicRng(0.0),
    )

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == expected_code
    assert len(client.calls) == expected_calls
    assert sleeps == ([0.25] if expected_calls == 2 else [])


def test_transport_timeout_is_not_converted_to_empty_inventory(app_factory):
    client = SequenceClient([TimeoutError("slow"), TimeoutError("still slow")])
    application = app_factory(
        CLIENT_FACTORY=lambda: client,
        SLEEP=lambda _seconds: None,
        RNG=DeterministicRng(0.0),
    )

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == "timeout"
    assert caught.value.retryable is True
    assert len(client.calls) == 2


def test_search_rejects_unexpected_content(app_factory):
    client = SequenceClient(
        [FakeResponse(text='{"blocked": true}', headers={"content-type": "application/json"})]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == "unexpected_content"


def test_search_does_not_treat_google_challenge_html_as_empty_inventory(app_factory):
    client = SequenceClient(
        [
            FakeResponse(
                text=(
                    "<html><title>Sorry...</title>"
                    "<p>Our systems have detected unusual traffic from your network.</p>"
                    "</html>"
                )
            )
        ]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == "unexpected_content"


def test_search_cache_uses_full_key_and_returns_detached_copies(app_factory):
    clock = FakeClock()
    client = SequenceClient([FakeResponse(), FakeResponse(), FakeResponse()])
    application = app_factory(CLOCK=clock, CLIENT_FACTORY=lambda: client)

    with application.app_context():
        first = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        first[0]["amenities"].append("MUTATED")
        cached = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        different_stars = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 4)
        clock.advance(601)
        expired = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert "MUTATED" not in cached[0]["amenities"]
    assert different_stars
    assert expired
    assert len(client.calls) == 3


def test_genuine_empty_search_is_cached_for_only_sixty_seconds(app_factory):
    clock = FakeClock()
    empty_html = "<html><body>No available properties</body></html>"
    client = SequenceClient([FakeResponse(text=empty_html), FakeResponse(text=empty_html)])
    application = app_factory(CLOCK=clock, CLIENT_FACTORY=lambda: client)

    with application.app_context():
        assert search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5) == []
        clock.advance(59)
        assert search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5) == []
        clock.advance(2)
        assert search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5) == []

    assert len(client.calls) == 2


def test_provider_rejects_bad_url_before_constructing_request(app_factory):
    client = SequenceClient([])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context(), pytest.raises(Exception) as caught:
        fetch_provider_prices("https://127.0.0.1/latest/meta-data")

    assert type(caught.value).__name__ == "ValidationProblem"
    assert client.calls == []


def test_provider_rejects_redirect_to_non_allowlisted_target(app_factory):
    client = SequenceClient(
        [FakeResponse(status_code=302, headers={"location": "http://127.0.0.1/private"})]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        fetch_provider_prices("https://www.google.com/travel/hotels/entity/test")

    assert caught.value.code == "unsafe_redirect"
    assert len(client.calls) == 1


def test_provider_manually_follows_one_revalidated_google_redirect(app_factory):
    provider_html = r"<html>Agoda \u0024180 Booking.com \x24190</html>"
    client = SequenceClient(
        [
            FakeResponse(
                status_code=302,
                headers={"location": "/travel/hotels/entity/canonical"},
            ),
            FakeResponse(text=provider_html),
        ]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        providers = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/test"
        )

    assert providers == {"agoda": 180.0, "booking.com": 190.0}
    assert [call[0] for call in client.calls] == [
        "https://www.google.com/travel/hotels/entity/test",
        "https://www.google.com/travel/hotels/entity/canonical",
    ]


def test_provider_cache_expires_after_nine_hundred_seconds(app_factory):
    clock = FakeClock()
    html = r"<html>Agoda \u0024180</html>"
    client = SequenceClient([FakeResponse(text=html), FakeResponse(text=html)])
    application = app_factory(CLOCK=clock, CLIENT_FACTORY=lambda: client)

    with application.app_context():
        first = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/test"
        )
        first["agoda"] = 1
        clock.advance(899)
        cached = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/test"
        )
        clock.advance(2)
        expired = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/test"
        )

    assert cached == {"agoda": 180.0}
    assert expired == {"agoda": 180.0}
    assert len(client.calls) == 2


def test_call_with_retry_waits_once_with_jitter_and_never_after_success(app_factory):
    sleeps = []
    rng = DeterministicRng(0.1)
    application = app_factory(SLEEP=sleeps.append, RNG=rng)
    attempts = 0

    def transient_then_success():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise UpstreamError("rate_limited", source="google", retryable=True)
        return "ok"

    with application.app_context():
        assert call_with_retry(transient_then_success) == "ok"
        assert call_with_retry(lambda: "immediate") == "immediate"

    assert attempts == 2
    assert sleeps == [0.35]
    assert rng.calls == [(0, 0.25)]


def test_process_google_semaphore_clamps_configured_capacity(app_factory, monkeypatch):
    monkeypatch.setenv("HOTEL_FINDER_UPSTREAM_CONCURRENCY", "99")
    high = app_factory()
    monkeypatch.setenv("HOTEL_FINDER_UPSTREAM_CONCURRENCY", "0")
    low = app_factory()

    assert high.extensions["hotel_finder"]["upstream_limit"] == 6
    assert low.extensions["hotel_finder"]["upstream_limit"] == 1


def test_google_requests_never_exceed_process_semaphore(app_factory):
    lock = threading.Lock()
    release = threading.Event()
    entered_two = threading.Event()
    active = 0
    maximum = 0

    class BlockingClient:
        def get(self, _url, **_kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 2:
                    entered_two.set()
            assert release.wait(2)
            with lock:
                active -= 1
            return FakeResponse()

    application = app_factory(
        CLIENT_FACTORY=lambda: BlockingClient(),
        UPSTREAM_CONCURRENCY=2,
    )

    def search(index):
        with application.app_context():
            return search_hotels(f"Place {index}", "2026-08-10", "2026-08-11", 5)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(search, index) for index in range(5)]
        assert entered_two.wait(1)
        release.set()
        assert all(future.result(timeout=2) for future in futures)

    assert maximum == 2
