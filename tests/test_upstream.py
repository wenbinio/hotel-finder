from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date

import pytest
import requests

import app as app_module
import hotel_finder.cache as cache_module
from app import (
    UpstreamError,
    call_with_retry,
    create_app,
    fetch_provider_prices,
    fetch_xotelo_prices,
    search_hotels,
)
from hotel_finder.cache import CacheDeadlineExceeded, TTLCache

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


class RawBytesReader:
    def __init__(self, response):
        self.response = response
        self.offset = 0
        self.calls = []

    def read1(self, size, decode_content=True):
        self.calls.append((size, decode_content))
        body = self.response.body
        if self.offset >= len(body):
            return b""
        chunk = body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class JsonResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.body = json.dumps(payload).encode("utf-8")
        self.closed = False
        self.raw = RawBytesReader(self)

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=1):
        size = (len(self.body) or 1) if chunk_size is None else chunk_size
        for offset in range(0, len(self.body), size):
            yield self.body[offset : offset + size]

    def close(self):
        self.closed = True


class SlowStreamingJsonResponse(JsonResponse):
    def __init__(self, payload, clock, seconds_per_chunk=8):
        super().__init__(payload)
        self.clock = clock
        self.seconds_per_chunk = seconds_per_chunk
        self.raw = RawBytesReader(self)

        original_read = self.raw.read1
        max_read = max(1, len(self.body) // 2)

        def slow_read(size, decode_content=True):
            self.clock.advance(self.seconds_per_chunk)
            return original_read(
                min(size, max_read), decode_content=decode_content
            )

        self.raw.read1 = slow_read

    def iter_content(self, chunk_size=1):
        raise AssertionError("bounded raw reads must be used when available")
        yield b""  # pragma: no cover


class FailingStreamingJsonResponse(JsonResponse):
    def __init__(self, payload):
        super().__init__(payload)

        def failing_read(_size, decode_content=True):
            del decode_content
            raise requests.ConnectionError("Read timed out while streaming")

        self.raw.read1 = failing_read

    def iter_content(self, chunk_size=1):
        del chunk_size
        raise requests.ConnectionError("Read timed out while streaming")
        yield b""  # pragma: no cover


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


class BlockingHeaderClient:
    def __init__(self, response, hold_seconds=0.25):
        self.response = response
        self.hold_seconds = hold_seconds
        self.started = threading.Event()
        self.finished = threading.Event()

    def get(self, _url, **_kwargs):
        self.started.set()
        self.finished.wait(self.hold_seconds)
        self.finished.set()
        return self.response


class SaturatedXoteloClient:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()

    def get(self, _url, **_kwargs):
        with self.lock:
            self.calls += 1
            self.started.set()
        self.release.wait(1)
        return JsonResponse({"result": {"rates": []}})


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AdvanceOnCacheSampleClock:
    def __init__(self, clock, *, advance_on_call, seconds):
        self.clock = clock
        self.advance_on_call = advance_on_call
        self.seconds = seconds
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.calls += 1
            if self.calls == self.advance_on_call:
                self.clock.advance(self.seconds)
            return self.clock()


class TrackingClock:
    def __init__(self, clock):
        self.clock = clock
        self.calls = 0
        self.values = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.calls += 1
            value = self.clock()
            self.values.append(value)
            return value

    def was_sampled(self):
        with self.lock:
            return self.calls > 0


class PreemptingPublicationClock:
    def __init__(self, clock, service_clock, publication_ready):
        self.clock = clock
        self.service_clock = service_clock
        self.publication_ready = publication_ready
        self.anchor_sampled = False
        self.publication_advanced = False
        self.anchor_value = None
        self.publication_value = None
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            if not self.anchor_sampled:
                self.anchor_sampled = True
                if self.service_clock.was_sampled():
                    self._advance_to(10)
                self.anchor_value = self.clock()
                return self.anchor_value
            elif self.publication_ready.is_set() and not self.publication_advanced:
                self.publication_advanced = True
                self._advance_to(16)
                self.publication_value = self.clock()
                return self.publication_value
            return self.clock()

    def _advance_to(self, target):
        if self.clock.now < target:
            self.clock.advance(target - self.clock.now)


class BlockingSequenceClient(SequenceClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.started = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.blocked = False

    def get(self, url, **kwargs):
        with self.lock:
            should_block = not self.blocked
            self.blocked = True
        if should_block:
            self.started.set()
            assert self.release.wait(2)
        return super().get(url, **kwargs)


class AdvancingSequenceClient(SequenceClient):
    def __init__(self, responses, clock, publication_ready):
        super().__init__(responses)
        self.clock = clock
        self.publication_ready = publication_ready

    def get(self, url, **kwargs):
        if not self.publication_ready.is_set():
            if self.clock.now < 10:
                self.clock.advance(10 - self.clock.now)
            self.publication_ready.set()
        return super().get(url, **kwargs)


class DeterministicRng:
    def __init__(self, value=0.1):
        self.value = value
        self.calls = []

    def uniform(self, minimum, maximum):
        self.calls.append((minimum, maximum))
        return self.value


class HeldTrackingSemaphore:
    def __init__(self):
        self._semaphore = threading.BoundedSemaphore(1)
        assert self._semaphore.acquire()
        self.waiting = threading.Event()

    def acquire(self, blocking=True, timeout=None):
        self.waiting.set()
        return self._semaphore.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        self._semaphore.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.release()

    def try_acquire(self):
        return self._semaphore.acquire(blocking=False)

    def release_held_slot(self):
        self._semaphore.release()


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


def test_search_rejects_generic_maintenance_html_without_caching_it(app_factory):
    client = SequenceClient(
        [FakeResponse(text="<html><h1>Scheduled maintenance</h1></html>"), FakeResponse()]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        hotels = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == "unexpected_content"
    assert hotels[0]["name"] == "Test Grand Hotel"
    assert len(client.calls) == 2


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


def test_structural_search_card_filtered_by_stars_is_cached_as_empty(
    app_factory,
):
    html = """
    <html><body><div class="result uaTTDe sponsored">
      <h2 class="BgYkof">Three Star Hotel</h2>
      <span class="ne5qie Ih19Ad">3-star hotel</span><span>$120</span>
    </div></body></html>
    """
    client = SequenceClient([FakeResponse(text=html)])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        first = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        cached = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert first == []
    assert cached == []
    assert len(client.calls) == 1


def test_structural_search_card_filtered_by_price_is_cached_as_empty(
    app_factory,
):
    html = """
    <html><body><div class="uaTTDe">
      <h2 class="BgYkof">Over Budget Hotel</h2>
      <span class="ne5qie Ih19Ad">5-star hotel</span><span>$1,501</span>
    </div></body></html>
    """
    client = SequenceClient([FakeResponse(text=html)])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        first = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        cached = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert first == []
    assert cached == []
    assert len(client.calls) == 1


def test_lone_google_result_wrapper_is_not_proof_of_genuine_empty(app_factory):
    maintenance = '<html><div class="uaTTDe">Scheduled maintenance</div></html>'
    client = SequenceClient([FakeResponse(text=maintenance), FakeResponse()])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        recovered = search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)

    assert caught.value.code == "unexpected_content"
    assert recovered[0]["name"] == "Test Grand Hotel"
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


def test_provider_rejects_generic_maintenance_html_without_caching_it(app_factory):
    provider_html = r"<html>Agoda \u0024180</html>"
    client = SequenceClient(
        [
            FakeResponse(
                text="<html><h1>No availability during scheduled maintenance</h1></html>"
            ),
            FakeResponse(text=provider_html),
        ]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_provider_prices(
                "https://www.google.com/travel/hotels/entity/maintenance"
            )
        providers = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/maintenance"
        )

    assert caught.value.code == "unexpected_content"
    assert providers == {"agoda": 180.0}
    assert len(client.calls) == 2


def test_unrelated_google_entity_link_is_not_proof_of_genuine_empty(app_factory):
    maintenance = (
        '<html><a href="/travel/hotels/entity/unrelated">'
        "Scheduled maintenance</a></html>"
    )
    provider_html = r"<html>Agoda \u0024180</html>"
    client = SequenceClient(
        [FakeResponse(text=maintenance), FakeResponse(text=provider_html)]
    )
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_provider_prices(
                "https://www.google.com/travel/hotels/entity/maintenance-link"
            )
        recovered = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/maintenance-link"
        )

    assert caught.value.code == "unexpected_content"
    assert recovered == {"agoda": 180.0}
    assert len(client.calls) == 2


def test_provider_caches_explicit_no_prices_marker(app_factory):
    html = "<html><main data-google-hotel-entity>No prices available</main></html>"
    client = SequenceClient([FakeResponse(text=html)])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        first = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/no-rates"
        )
        second = fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/no-rates"
        )

    assert first == {}
    assert second == {}
    assert len(client.calls) == 1


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


def test_provider_request_replaces_stale_url_dates_with_authoritative_dates(
    app_factory,
):
    provider_html = r"<html>Agoda \u0024180</html>"
    client = SequenceClient([FakeResponse(text=provider_html)])
    application = app_factory(CLIENT_FACTORY=lambda: client)

    with application.app_context():
        fetch_provider_prices(
            "https://www.google.com/travel/hotels/entity/test?checkin=2026-09-01&checkout=2026-09-02",
            checkin="2026-08-10",
            checkout="2026-08-14",
        )

    url, options = client.calls[0]
    assert "2026-09" not in url
    assert options["params"]["checkin"] == "2026-08-10"
    assert options["params"]["checkout"] == "2026-08-14"


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


def test_call_with_retry_stops_before_jitter_and_retry_when_cancelled(app_factory):
    cancelled = threading.Event()
    sleeps = []
    rng = DeterministicRng(0.1)
    application = app_factory(SLEEP=sleeps.append, RNG=rng)
    attempts = 0

    def transient_then_cancel():
        nonlocal attempts
        attempts += 1
        cancelled.set()
        raise UpstreamError("rate_limited", source="google", retryable=True)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        call_with_retry(transient_then_cancel, cancelled=cancelled.is_set)

    assert caught.value.code == "rate_limited"
    assert attempts == 1
    assert sleeps == []
    assert rng.calls == []


def test_google_semaphore_wait_is_cancellable_without_releasing_unowned_slot(
    app_factory,
):
    gate = HeldTrackingSemaphore()
    client = SequenceClient([FakeResponse()])
    application = app_factory(CLIENT_FACTORY=lambda: client)
    application.extensions["hotel_finder"]["upstream_semaphore"] = gate
    cancelled = threading.Event()

    def request_google():
        with application.app_context():
            return app_module._request_upstream(
                "https://www.google.com/travel/hotels/bangkok",
                params={"hl": "en"},
                source="google",
                cancelled=cancelled.is_set,
            )

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(request_google)
    acquired_during_cancel = False
    try:
        assert gate.waiting.wait(1)
        cancelled.set()
        with pytest.raises(UpstreamError) as caught:
            future.result(timeout=1)
        assert caught.value.code == "upstream_failure"
        acquired_during_cancel = gate.try_acquire()
        assert acquired_during_cancel is False
        assert client.calls == []
    finally:
        if acquired_during_cancel:
            gate.release()
        else:
            gate.release_held_slot()
        pool.shutdown(wait=True)


def test_xotelo_cache_is_full_key_ttl_bounded_and_returns_detached_copies(app_factory):
    clock = FakeClock()
    payload = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0, "tax": 4.0}
            ]
        }
    }
    client = SequenceClient([JsonResponse(payload), JsonResponse(payload), JsonResponse(payload)])
    application = app_factory(CLOCK=clock, XOTELO_CLIENT=client)

    with application.app_context():
        first = fetch_xotelo_prices(
            "ta-key", "Hotel One", "2026-08-10", "2026-08-11"
        )
        first["Agoda"]["rate"] = 1.0
        cached = fetch_xotelo_prices(
            "ta-key", "Hotel Two", "2026-08-10", "2026-08-11"
        )
        different_dates = fetch_xotelo_prices(
            "ta-key", "Hotel One", "2026-08-10", "2026-08-12"
        )
        clock.advance(901)
        expired = fetch_xotelo_prices(
            "ta-key", "Hotel One", "2026-08-10", "2026-08-11"
        )

    assert cached["Agoda"]["rate"] == 88.0
    assert "Hotel+Two" in cached["Agoda"]["url"]
    assert different_dates["Agoda"]["rate"] == 88.0
    assert expired["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 3
    timeout = client.calls[0][1]["timeout"]
    assert timeout.total == 15
    assert timeout.connect_timeout == 5
    assert timeout.read_timeout == 1
    assert client.calls[0][1]["stream"] is True


def test_xotelo_slow_stream_fails_at_elapsed_deadline_and_is_not_cached(
    app_factory,
):
    clock = FakeClock()
    slow = SlowStreamingJsonResponse({"result": {"rates": []}}, clock)
    valid_payload = {
        "result": {
            "rates": [{"name": "Agoda", "code": "Agoda", "rate": 88.0}]
        }
    }
    client = SequenceClient([slow, JsonResponse(valid_payload)])
    application = app_factory(CLOCK=clock, XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "timeout"
    assert slow.closed is True
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_parse_crossing_deadline_is_not_returned_or_cached(
    app_factory, monkeypatch
):
    clock = FakeClock()
    expired = JsonResponse({"result": {"rates": []}})
    valid = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0}
            ]
        }
    }
    client = SequenceClient([expired, JsonResponse(valid)])
    original_loads = app_module.json.loads
    parse_calls = 0

    def advancing_loads(body):
        nonlocal parse_calls
        parse_calls += 1
        parsed = original_loads(body)
        if parse_calls == 1:
            clock.advance(16)
        return parsed

    monkeypatch.setattr(app_module.json, "loads", advancing_loads)
    application = app_factory(CLOCK=clock, XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "timeout"
    assert expired.closed is True
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_malformed_parse_crossing_deadline_reports_timeout(
    app_factory, monkeypatch
):
    clock = FakeClock()
    response = JsonResponse({"result": {"rates": []}})

    def failing_loads(_body):
        clock.advance(16)
        raise ValueError("malformed")

    monkeypatch.setattr(app_module.json, "loads", failing_loads)
    application = app_factory(CLOCK=clock)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        app_module._read_xotelo_rates(
            response,
            application.extensions["hotel_finder"],
            deadline=15,
        )

    assert caught.value.code == "timeout"


def test_xotelo_recursion_error_is_malformed_content(app_factory, monkeypatch):
    response = JsonResponse({"result": {"rates": []}})
    client = SequenceClient([response])

    def recursive_loads(_body):
        raise RecursionError("nested too deeply")

    monkeypatch.setattr(app_module.json, "loads", recursive_loads)
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert caught.value.retryable is False
    assert response.closed is True


def test_xotelo_publication_crossing_deadline_is_not_returned_or_cached(
    app_factory, monkeypatch
):
    waiter_joined = threading.Event()

    class WaiterAwareFuture(Future):
        def result(self, timeout=None):
            waiter_joined.set()
            return super().result(timeout=timeout)

    monkeypatch.setattr(cache_module, "Future", WaiterAwareFuture)
    service_clock = FakeClock()
    cache_clock = AdvanceOnCacheSampleClock(
        service_clock, advance_on_call=5, seconds=16
    )
    cache = TTLCache(max_entries=8, ttl_seconds=900, clock=cache_clock)
    expired = JsonResponse({"result": {"rates": []}})
    valid = JsonResponse(
        {
            "result": {
                "rates": [
                    {
                        "name": "Agoda",
                        "code": "Agoda",
                        "rate": 88.0,
                    }
                ]
            }
        }
    )
    client = BlockingSequenceClient([expired, valid])
    application = app_factory(
        CLOCK=service_clock, XOTELO_CACHE=cache, XOTELO_CLIENT=client
    )

    def capture_timeout():
        with application.app_context():
            try:
                fetch_xotelo_prices(
                    "ta-key", "Hotel", "2026-08-10", "2026-08-11"
                )
            except UpstreamError as error:
                return error
        raise AssertionError("late Xotelo result was returned")

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        owner = pool.submit(capture_timeout)
        assert client.started.wait(1)
        waiter = pool.submit(capture_timeout)
        assert waiter_joined.wait(1)
        client.release.set()
        errors = [owner.result(timeout=2), waiter.result(timeout=2)]
    finally:
        client.release.set()
        pool.shutdown(wait=True)

    assert [error.code for error in errors] == ["timeout", "timeout"]
    assert all(error.retryable for error in errors)
    assert isinstance(errors[0].__cause__, CacheDeadlineExceeded)
    assert errors[0].__cause__ is errors[1].__cause__
    assert cache.size() == 0

    with application.app_context():
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2
    assert expired.closed is True


def test_xotelo_publication_deadline_uses_injected_cache_clock_epoch(
    app_factory,
):
    service_clock = FakeClock()
    cache_clock = FakeClock()
    cache_clock.advance(57_000)
    cache = TTLCache(max_entries=8, ttl_seconds=900, clock=cache_clock)
    client = SequenceClient(
        [
            JsonResponse(
                {
                    "result": {
                        "rates": [
                            {
                                "name": "Agoda",
                                "code": "Agoda",
                                "rate": 88.0,
                            }
                        ]
                    }
                }
            )
        ]
    )
    application = app_factory(
        CLOCK=service_clock, XOTELO_CACHE=cache, XOTELO_CLIENT=client
    )

    with application.app_context():
        first = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )
        cached = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert first["Agoda"]["rate"] == 88.0
    assert cached == first
    assert len(client.calls) == 1


def test_xotelo_anchors_cache_deadline_before_transport_preemption(
    app_factory,
):
    base_clock = FakeClock()
    service_clock = TrackingClock(base_clock)
    publication_ready = threading.Event()
    cache_clock = PreemptingPublicationClock(
        base_clock, service_clock, publication_ready
    )
    cache = TTLCache(max_entries=8, ttl_seconds=900, clock=cache_clock)
    expired = JsonResponse({"result": {"rates": []}})
    valid = JsonResponse(
        {
            "result": {
                "rates": [
                    {
                        "name": "Agoda",
                        "code": "Agoda",
                        "rate": 88.0,
                    }
                ]
            }
        }
    )
    client = AdvancingSequenceClient(
        [expired, valid], base_clock, publication_ready
    )
    application = app_factory(
        CLOCK=service_clock, XOTELO_CACHE=cache, XOTELO_CLIENT=client
    )

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        last_loader_time = service_clock.values[-1]
        assert cache.size() == 0
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "timeout"
    assert caught.value.retryable is True
    assert cache_clock.anchor_value == 0.0
    assert cache_clock.publication_value == 16.0
    assert last_loader_time == 10.0
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2
    assert expired.closed is True


def test_xotelo_header_wait_obeys_hard_deadline_without_releasing_live_slot(
    app_factory, monkeypatch
):
    monkeypatch.setattr(app_module, "TOTAL_TIMEOUT_SECONDS", 0.05)
    response = JsonResponse({"result": {"rates": []}})
    client = BlockingHeaderClient(response)
    application = app_factory(XOTELO_CLIENT=client, UPSTREAM_CONCURRENCY=1)
    gate = application.extensions["hotel_finder"]["upstream_semaphore"]

    started_at = time.monotonic()
    with application.app_context(), pytest.raises(UpstreamError) as caught:
        fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )
    elapsed = time.monotonic() - started_at

    assert caught.value.code == "timeout"
    assert elapsed < 0.15
    assert client.started.is_set()
    assert gate.acquire(blocking=False) is False
    assert client.finished.wait(1)
    assert gate.acquire(timeout=1)
    gate.release()
    assert response.closed is True


def test_google_queue_deadline_survives_a_retained_xotelo_permit_and_recovers(
    app_factory, monkeypatch
):
    monkeypatch.setattr(app_module, "TOTAL_TIMEOUT_SECONDS", 0.05)
    xotelo_response = JsonResponse({"result": {"rates": []}})
    xotelo = BlockingHeaderClient(xotelo_response)
    google = SequenceClient([FakeResponse()])
    sleeps = []
    application = app_factory(
        XOTELO_CLIENT=xotelo,
        CLIENT_FACTORY=lambda: google,
        UPSTREAM_CONCURRENCY=1,
        UPSTREAM_QUEUE_TIMEOUT_SECONDS=0.03,
        SLEEP=sleeps.append,
    )
    gate = application.extensions["hotel_finder"]["upstream_semaphore"]

    with application.app_context():
        with pytest.raises(UpstreamError) as xotelo_error:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )

        started_at = time.monotonic()
        with pytest.raises(UpstreamError) as google_error:
            search_hotels("Bangkok", "2026-08-10", "2026-08-11", 5)
        elapsed = time.monotonic() - started_at

        assert xotelo_error.value.code == "timeout"
        assert google_error.value.code == "upstream_busy"
        assert google_error.value.retryable is True
        assert elapsed < 0.12
        assert xotelo.finished.is_set() is False
        assert google.calls == []
        assert sleeps == []

        assert xotelo.finished.wait(1)
        assert gate.acquire(timeout=1)
        gate.release()
        recovered = search_hotels(
            "Bangkok", "2026-08-10", "2026-08-11", 5
        )

    assert recovered[0]["name"] == "Test Grand Hotel"
    assert len(google.calls) == 1
    assert xotelo_response.closed is True


def test_xotelo_bulkhead_keeps_google_available_under_saturation(
    app_factory, monkeypatch
):
    monkeypatch.setattr(app_module, "TOTAL_TIMEOUT_SECONDS", 0.05)
    xotelo = SaturatedXoteloClient()
    google = SequenceClient([FakeResponse()])
    application = app_factory(
        XOTELO_CLIENT=xotelo,
        CLIENT_FACTORY=lambda: google,
        UPSTREAM_CONCURRENCY=4,
        UPSTREAM_QUEUE_TIMEOUT_SECONDS=0.03,
    )
    services = application.extensions["hotel_finder"]

    def fetch_one(index):
        with application.app_context():
            try:
                fetch_xotelo_prices(
                    f"ta-key-{index}",
                    "Hotel",
                    "2026-08-10",
                    "2026-08-11",
                )
            except UpstreamError as error:
                return error.code
        return "unexpected-success"

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(fetch_one, index) for index in range(4)]
    total_gate = services["upstream_semaphore"]
    xotelo_gate = services["xotelo_semaphore"]
    acquired_total = 0
    acquired_xotelo = False
    try:
        assert xotelo.started.wait(1)
        assert [future.result(timeout=1) for future in futures] == [
            "timeout",
            "timeout",
            "timeout",
            "timeout",
        ]
        with application.app_context():
            hotels = search_hotels(
                "Bangkok", "2026-08-10", "2026-08-11", 5
            )

        assert xotelo.calls == 1
        assert hotels[0]["name"] == "Test Grand Hotel"
        assert len(google.calls) == 1
    finally:
        xotelo.release.set()
        pool.shutdown(wait=True)
        for _index in range(4):
            if total_gate.acquire(timeout=1):
                acquired_total += 1
        acquired_xotelo = xotelo_gate.acquire(timeout=1)
        for _index in range(acquired_total):
            total_gate.release()
        if acquired_xotelo:
            xotelo_gate.release()

    assert acquired_total == 4
    assert acquired_xotelo is True


def test_xotelo_oversized_stream_is_rejected_without_caching(app_factory):
    oversized = JsonResponse({"result": {"rates": []}})
    oversized.body = b"x" * 1_100_000
    valid_payload = {
        "result": {
            "rates": [{"name": "Agoda", "code": "Agoda", "rate": 88.0}]
        }
    }
    client = SequenceClient([oversized, JsonResponse(valid_payload)])
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert oversized.closed is True
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_stream_read_timeout_is_reported_as_timeout(app_factory):
    client = SequenceClient(
        [FailingStreamingJsonResponse({"result": {"rates": []}})]
    )
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "timeout"


def test_xotelo_uses_one_attempt_and_does_not_cache_failure(app_factory):
    payload = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0, "tax": 0}
            ]
        }
    }
    client = SequenceClient(
        [JsonResponse({}, status_code=503), JsonResponse(payload)]
    )
    sleeps = []
    application = app_factory(XOTELO_CLIENT=client, SLEEP=sleeps.append)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        result = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "upstream_unavailable"
    assert result["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2
    assert sleeps == []


@pytest.mark.parametrize(
    "malformed_payload",
    [
        {},
        {"error": "upstream API error"},
        {"result": []},
        {"result": {}},
        {"result": {"rates": {}}},
    ],
)
def test_xotelo_rejects_malformed_shape_without_caching_it(
    app_factory, malformed_payload
):
    valid_payload = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0, "tax": 0}
            ]
        }
    }
    client = SequenceClient(
        [JsonResponse(malformed_payload), JsonResponse(valid_payload)]
    )
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_rejects_missing_provider_name_without_caching_it(app_factory):
    malformed = {
        "result": {"rates": [{"code": "Agoda", "rate": 88.0}]}
    }
    valid = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0}
            ]
        }
    }
    client = SequenceClient([JsonResponse(malformed), JsonResponse(valid)])
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "malformed_rate",
    [
        "not-an-object",
        {"name": "", "code": "Agoda", "rate": 88.0},
        {"name": "   ", "code": "Agoda", "rate": 88.0},
        {"name": "Bad\nProvider", "code": "Agoda", "rate": 88.0},
        {"name": "\tAgoda", "code": "Agoda", "rate": 88.0},
        {"name": "Agoda\n", "code": "Agoda", "rate": 88.0},
        {"name": " " * 256 + "Agoda", "code": "Agoda", "rate": 88.0},
        {"name": "A" * 129, "code": "Agoda", "rate": 88.0},
        {"name": "Agoda", "code": 7, "rate": 88.0},
        {"name": "Agoda", "code": "A" * 65, "rate": 88.0},
        {"name": "Agoda", "code": "Bad\tCode", "rate": 88.0},
        {"name": "Agoda", "code": "\tAgoda", "rate": 88.0},
        {"name": "Agoda", "code": "Agoda\n", "rate": 88.0},
        {"name": "Agoda", "code": " " * 256 + "Agoda", "rate": 88.0},
        {"name": "Agoda", "code": "Agoda", "rate": True},
        {"name": "Agoda", "code": "Agoda", "rate": 0},
        {"name": "Agoda", "code": "Agoda", "rate": -1},
        {"name": "Agoda", "code": "Agoda", "rate": 1_000_001},
        {"name": "Agoda", "code": "Agoda", "rate": 10**400},
        {"name": "Agoda", "code": "Agoda", "rate": 88.0, "tax": -1},
        {
            "name": "Agoda",
            "code": "Agoda",
            "rate": 88.0,
            "tax": 1_000_001,
        },
    ],
)
def test_xotelo_rejects_invalid_inner_rate_records(
    app_factory, malformed_rate
):
    client = SequenceClient(
        [JsonResponse({"result": {"rates": [malformed_rate]}})]
    )
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context(), pytest.raises(UpstreamError) as caught:
        fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"


def test_xotelo_rejects_large_raw_rate_amplification_without_caching(
    app_factory,
):
    amplified = JsonResponse(
        {
            "result": {
                "rates": [
                    {"name": "A", "code": "A", "rate": 1}
                    for _index in range(18_000)
                ]
            }
        }
    )
    assert len(amplified.body) < 1_000_000
    valid = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0}
            ]
        }
    }
    client = SequenceClient([amplified, JsonResponse(valid)])
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_deduplicates_normalized_providers_before_caching(app_factory):
    payload = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 90.0},
                {"name": " agoda ", "code": "Agoda", "rate": 88.0},
                {"name": "AGO.DA", "code": "Agoda", "rate": 80.0},
            ]
        }
    }
    client = SequenceClient([JsonResponse(payload)])
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        first = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )
        cached = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert len(first) == 1
    assert next(iter(first.values()))["rate"] == 80.0
    assert cached == first
    assert len(client.calls) == 1


def test_xotelo_rejects_too_many_normalized_providers_without_caching(
    app_factory,
):
    too_many = {
        "result": {
            "rates": [
                {
                    "name": f"Provider {index}",
                    "code": "Agoda",
                    "rate": float(index + 1),
                }
                for index in range(129)
            ]
        }
    }
    valid = {
        "result": {
            "rates": [
                {"name": "Agoda", "code": "Agoda", "rate": 88.0}
            ]
        }
    }
    client = SequenceClient([JsonResponse(too_many), JsonResponse(valid)])
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        with pytest.raises(UpstreamError) as caught:
            fetch_xotelo_prices(
                "ta-key", "Hotel", "2026-08-10", "2026-08-11"
            )
        recovered = fetch_xotelo_prices(
            "ta-key", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert caught.value.code == "unexpected_content"
    assert recovered["Agoda"]["rate"] == 88.0
    assert len(client.calls) == 2


def test_xotelo_cache_key_preserves_tripadvisor_key_identity(app_factory):
    first_payload = {
        "result": {
            "rates": [{"name": "Agoda", "code": "Agoda", "rate": 88.0}]
        }
    }
    second_payload = {
        "result": {
            "rates": [{"name": "Agoda", "code": "Agoda", "rate": 99.0}]
        }
    }
    client = SequenceClient(
        [JsonResponse(first_payload), JsonResponse(second_payload)]
    )
    application = app_factory(XOTELO_CLIENT=client)

    with application.app_context():
        numeric_key = fetch_xotelo_prices(
            7, "Hotel", "2026-08-10", "2026-08-11"
        )
        string_key = fetch_xotelo_prices(
            "7", "Hotel", "2026-08-10", "2026-08-11"
        )

    assert numeric_key["Agoda"]["rate"] == 88.0
    assert string_key["Agoda"]["rate"] == 99.0
    assert len(client.calls) == 2


def test_default_client_factory_is_thread_local_per_application(app_factory, monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **_kwargs):
            created.append(self)

    monkeypatch.setattr(app_module, "Client", FakeClient)
    first_app = app_factory()
    second_app = app_factory()

    first_factory = first_app.extensions["hotel_finder"]["client_factory"]
    second_factory = second_app.extensions["hotel_finder"]["client_factory"]
    first_client = first_factory()
    second_client = second_factory()

    assert first_factory() is first_client
    assert second_factory() is second_client
    assert first_client is not second_client
    assert len(created) == 2


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
