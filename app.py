"""Hardened Flask API for the private hotel finder service."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urljoin, urlsplit, urlunsplit

import requests
from fast_hotels.hotels_impl import Guests, HotelData, THSData
from fast_hotels.primp import Client
from flask import Flask, current_app, g, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from requests.adapters import TimeoutSauce
from werkzeug.exceptions import (
    BadRequest,
    MethodNotAllowed,
    NotFound,
    UnsupportedMediaType,
)

from hotel_finder import __version__
from hotel_finder.cache import CacheDeadlineExceeded, TTLCache
from hotel_finder.jobs import JobConflict, JobNotFound, SweepJob, SweepJobManager
from hotel_finder.parsing import (
    ParseContext,
    has_structural_hotel_card,
    parse_hotel_cards,
    parse_provider_prices,
)
from hotel_finder.validation import (
    CompareRequest,
    HotelInput,
    SearchAllRequest,
    SearchRequest,
    SweepRequest,
    ValidationProblem,
    parse_compare_request,
    parse_search_all_request,
    parse_search_request,
    parse_sweep_request,
    sample_stay_dates,
    validate_google_hotel_url,
)

IMPERSONATE_PROFILE = "chrome_146"
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 12
TOTAL_TIMEOUT_SECONDS = 15
DEFAULT_UPSTREAM_CONCURRENCY = 4
MAX_UPSTREAM_CONCURRENCY = 6
DEFAULT_UPSTREAM_QUEUE_TIMEOUT_SECONDS = 2.0
MIN_UPSTREAM_QUEUE_TIMEOUT_SECONDS = 0.01
SEARCH_CACHE_SECONDS = 600
EMPTY_SEARCH_CACHE_SECONDS = 60
PROVIDER_CACHE_SECONDS = 900
XOTELO_CACHE_SECONDS = 900
XOTELO_READ_TIMEOUT_SECONDS = 1
XOTELO_STREAM_CHUNK_BYTES = 16 * 1024
XOTELO_MAX_RESPONSE_BYTES = 1_000_000
XOTELO_MAX_RAW_RATES = 2_048
XOTELO_MAX_STORED_RATES = 128
XOTELO_MAX_PROVIDER_NAME_LENGTH = 128
XOTELO_MAX_PROVIDER_CODE_LENGTH = 64
XOTELO_MAX_RATE_VALUE = 1_000_000.0
UPSTREAM_SEMAPHORE_POLL_SECONDS = 0.05
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

DESTINATIONS = {
    "beachfront": [
        {"name": "Phuket", "airport": "HKT", "flight_usd": 150},
        {"name": "Khao Lak", "airport": "HKT", "flight_usd": 150},
        {"name": "Koh Samui", "airport": "USM", "flight_usd": 280},
        {"name": "Da Nang", "airport": "DAD", "flight_usd": 280},
        {"name": "Phu Quoc", "airport": "PQC", "flight_usd": 250},
        {"name": "Bali", "airport": "DPS", "flight_usd": 128},
        {"name": "Lombok", "airport": "LOP", "flight_usd": 185},
        {"name": "Langkawi", "airport": "LGK", "flight_usd": 137},
        {"name": "Sihanoukville", "airport": "KOS", "flight_usd": 200},
        {"name": "Bentota", "airport": "CMB", "flight_usd": 253},
        {"name": "Nha Trang", "airport": "CXR", "flight_usd": 282},
        {"name": "Hoi An", "airport": "DAD", "flight_usd": 280},
        {"name": "Bintan", "airport": "TNJ", "flight_usd": 60},
    ],
    "non_beachfront": [
        {"name": "Kuala Lumpur", "airport": "KUL", "flight_usd": 77},
        {"name": "Bangkok", "airport": "BKK", "flight_usd": 126},
        {"name": "Jakarta", "airport": "CGK", "flight_usd": 107},
        {"name": "Ho Chi Minh City", "airport": "SGN", "flight_usd": 113},
        {"name": "Hanoi", "airport": "HAN", "flight_usd": 226},
        {"name": "Colombo", "airport": "CMB", "flight_usd": 253},
        {"name": "Siem Reap", "airport": "SAI", "flight_usd": 188},
        {"name": "Phnom Penh", "airport": "PNH", "flight_usd": 200},
        {"name": "Yogyakarta", "airport": "JOG", "flight_usd": 150},
    ],
}

DESTINATION_BY_NAME = {
    destination["name"]: {**destination, "category": category}
    for category, destinations in DESTINATIONS.items()
    for destination in destinations
}
KNOWN_LOCATIONS = set(DESTINATION_BY_NAME)
FLIGHT_BUDGET_MAP = {
    name: float(destination["flight_usd"])
    for name, destination in DESTINATION_BY_NAME.items()
}

_TA_KEYS_PATH = Path(__file__).with_name("ta_keys.json")
try:
    _TA_KEYS = json.loads(_TA_KEYS_PATH.read_text("utf-8")) if _TA_KEYS_PATH.is_file() else {}
except (OSError, ValueError):
    _TA_KEYS = {}

_client_local = threading.local()


class _StrictJSONProvider(DefaultJSONProvider):
    def dumps(self, obj: object, **kwargs: object) -> str:
        kwargs["allow_nan"] = False
        return super().dumps(obj, **kwargs)


class _HotelFinderFlask(Flask):
    json_provider_class = _StrictJSONProvider


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in (
            "request_id",
            "job_id",
            "request_method",
            "request_path",
            "response_status",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))


class InvalidJSON(ValueError):
    """The request body was absent, malformed, or not a JSON object."""


class UpstreamError(RuntimeError):
    """An upstream call failed and must not be represented as empty inventory."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        source: str,
        retryable: bool,
    ) -> None:
        self.code = code
        self.source = source
        self.retryable = retryable
        if message is None:
            message = self._default_message(code, source)
        super().__init__(message)

    @staticmethod
    def _default_message(code: str, source: str) -> str:
        if code == "timeout":
            return f"The {source} request timed out."
        messages = {
            "rate_limited": f"The {source} service rate-limited the request.",
            "upstream_unavailable": f"The {source} service is unavailable.",
            "upstream_response": f"The {source} service rejected the request.",
            "upstream_busy": f"The {source} request queue is busy.",
            "unexpected_content": f"The {source} service returned unexpected content.",
            "redirect": f"The {source} service returned an unsupported redirect.",
            "unsafe_redirect": f"The {source} service redirected to an unsafe target.",
            "transport": f"The {source} request could not be completed.",
            "upstream_failure": f"The {source} operation failed.",
        }
        return messages.get(code, f"The {source} operation failed ({code}).")


def make_client() -> Client:
    """Return one hardened primp client per worker thread."""
    if not hasattr(_client_local, "client"):
        _client_local.client = Client(
            impersonate=IMPERSONATE_PROFILE,
            verify=True,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=READ_TIMEOUT_SECONDS,
            timeout=TOTAL_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
    return _client_local.client


def _thread_local_client_factory() -> Callable[[], Client]:
    local = threading.local()

    def factory() -> Client:
        if not hasattr(local, "client"):
            local.client = Client(
                impersonate=IMPERSONATE_PROFILE,
                verify=True,
                connect_timeout=CONNECT_TIMEOUT_SECONDS,
                read_timeout=READ_TIMEOUT_SECONDS,
                timeout=TOTAL_TIMEOUT_SECONDS,
                follow_redirects=False,
            )
        return local.client

    return factory


def _services() -> dict[str, Any]:
    return current_app.extensions["hotel_finder"]


def _runtime_services(runtime: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _services() if runtime is None else runtime


def _raise_if_upstream_aborted(
    services: Mapping[str, Any],
    *,
    source: str,
    cancelled: Callable[[], bool] | None = None,
    deadline: float | None = None,
) -> None:
    if cancelled is not None and cancelled():
        raise UpstreamError(
            "upstream_failure",
            "The upstream operation was cancelled.",
            source=source,
            retryable=False,
        )
    if deadline is not None and services["clock"]() >= deadline:
        raise UpstreamError("timeout", source=source, retryable=True)


@contextmanager
def _upstream_slot(
    services: Mapping[str, Any],
    *,
    source: str,
    cancelled: Callable[[], bool] | None = None,
    deadline: float | None = None,
):
    semaphore = _acquire_upstream_slot(
        services,
        source=source,
        cancelled=cancelled,
        deadline=deadline,
    )
    try:
        yield
    finally:
        semaphore.release()


def _acquire_upstream_slot(
    services: Mapping[str, Any],
    *,
    source: str,
    cancelled: Callable[[], bool] | None = None,
    deadline: float | None = None,
):
    return _acquire_semaphore(
        services["upstream_semaphore"],
        services,
        source=source,
        cancelled=cancelled,
        deadline=deadline,
    )


def _acquire_semaphore(
    semaphore: object,
    services: Mapping[str, Any],
    *,
    source: str,
    cancelled: Callable[[], bool] | None = None,
    deadline: float | None = None,
):
    acquired = False
    try:
        if cancelled is None and deadline is None:
            acquired = semaphore.acquire()
        else:
            while not acquired:
                _raise_if_upstream_aborted(
                    services,
                    source=source,
                    cancelled=cancelled,
                    deadline=deadline,
                )
                wait_seconds = UPSTREAM_SEMAPHORE_POLL_SECONDS
                if deadline is not None:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.001, deadline - services["clock"]()),
                    )
                acquired = semaphore.acquire(timeout=wait_seconds)
        _raise_if_upstream_aborted(
            services,
            source=source,
            cancelled=cancelled,
            deadline=deadline,
        )
        return semaphore
    except BaseException:
        if acquired:
            semaphore.release()
        raise


def _bounded_upstream_limit(value: object) -> int:
    try:
        configured = int(value)
    except (TypeError, ValueError):
        configured = DEFAULT_UPSTREAM_CONCURRENCY
    return max(1, min(MAX_UPSTREAM_CONCURRENCY, configured))


def _bounded_queue_timeout(value: object) -> float:
    try:
        configured = float(value)
    except (TypeError, ValueError):
        configured = DEFAULT_UPSTREAM_QUEUE_TIMEOUT_SECONDS
    if not math.isfinite(configured):
        configured = DEFAULT_UPSTREAM_QUEUE_TIMEOUT_SECONDS
    return max(
        MIN_UPSTREAM_QUEUE_TIMEOUT_SECONDS,
        min(TOTAL_TIMEOUT_SECONDS, configured),
    )


def _ensure_json_safe(value: object, *, source: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise UpstreamError(
            "unexpected_content", source=source, retryable=False
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise UpstreamError(
                "unexpected_content", source=source, retryable=False
            )
        for nested in value.values():
            _ensure_json_safe(nested, source=source)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _ensure_json_safe(nested, source=source)
        return
    raise UpstreamError("unexpected_content", source=source, retryable=False)


def _validated_hotels(value: object, *, source: str = "google") -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UpstreamError("unexpected_content", source=source, retryable=False)
    hotels: list[dict[str, Any]] = []
    for hotel in value:
        if not isinstance(hotel, Mapping):
            raise UpstreamError(
                "unexpected_content", source=source, retryable=False
            )
        copied = dict(hotel)
        _ensure_json_safe(copied, source=source)
        hotels.append(copied)
    return hotels


def _freeze_hotels(hotels: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    frozen: list[str] = []
    for hotel in hotels:
        copied = dict(hotel)
        _ensure_json_safe(copied, source="google")
        frozen.append(
            json.dumps(
                copied,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return tuple(frozen)


def _copy_hotels(frozen: Sequence[str]) -> list[dict[str, Any]]:
    return [json.loads(hotel) for hotel in frozen]


def _response_header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "get"):
        return headers.get(name) or headers.get(name.lower())
    return None


def _is_timeout_error(error: BaseException) -> bool:
    return (
        isinstance(error, (requests.Timeout, TimeoutError))
        or "timeout" in type(error).__name__.casefold()
        or "timed out" in str(error).casefold()
    )


def _response_html(response: object, source: str) -> str:
    text = getattr(response, "text", None)
    content_type = _response_header(response, "content-type")
    normalized_text = text.lower() if isinstance(text, str) else ""
    blocked = any(
        marker in normalized_text
        for marker in (
            "our systems have detected unusual traffic",
            "g-recaptcha",
            "/sorry/",
        )
    )
    if (
        not isinstance(text, str)
        or not text.strip()
        or (content_type and "html" not in content_type.lower())
        or "<" not in text
        or blocked
    ):
        raise UpstreamError(
            "unexpected_content", source=source, retryable=False
        )
    return text


def _recognized_search_page(html: str, hotels: Sequence[object]) -> bool:
    if hotels or has_structural_hotel_card(html):
        return True
    normalized = html.casefold()
    return any(
        marker in normalized
        for marker in (
            "no available properties",
            "no properties found",
            "no hotels found",
            "no results found",
        )
    )


def _recognized_provider_page(
    html: str, prices: Mapping[str, float]
) -> bool:
    if prices:
        return True
    normalized = html.casefold()
    return any(
        marker in normalized
        for marker in (
            "no prices available",
            "no booking options",
        )
    )


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    cancelled: Callable[[], bool] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> T:
    """Run an upstream operation with no more than one transient retry."""
    services = _runtime_services(runtime)
    is_cancelled = cancelled or (lambda: False)
    for attempt in range(2):
        if is_cancelled():
            raise UpstreamError(
                "upstream_failure",
                "The upstream operation was cancelled.",
                source="internal",
                retryable=False,
            )
        try:
            return operation()
        except UpstreamError as error:
            if (
                attempt == 1
                or not error.retryable
                or error.code == "upstream_busy"
                or is_cancelled()
            ):
                raise
            delay = 0.25 + services["rng"].uniform(0, 0.25)
            if is_cancelled():
                raise error
            services["sleep"](delay)
            if is_cancelled():
                raise error
    raise RuntimeError("unreachable retry state")


def _request_upstream(
    url: str,
    *,
    params: Mapping[str, object],
    source: str,
    return_redirect: bool = False,
    cancelled: Callable[[], bool] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> object:
    services = _runtime_services(runtime)

    def operation() -> object:
        queue_deadline = (
            services["clock"]() + services["upstream_queue_timeout"]
        )
        try:
            semaphore = _acquire_upstream_slot(
                services,
                source=source,
                cancelled=cancelled,
                deadline=queue_deadline,
            )
        except UpstreamError as error:
            if error.code == "timeout":
                raise UpstreamError(
                    "upstream_busy", source=source, retryable=True
                ) from error
            raise
        try:
            _raise_if_upstream_aborted(
                services, source=source, cancelled=cancelled
            )
            client = services["client_factory"]()
            _raise_if_upstream_aborted(
                services, source=source, cancelled=cancelled
            )
            response = client.get(url, params=dict(params))
        except UpstreamError:
            raise
        except Exception as error:
            raise UpstreamError(
                "timeout" if _is_timeout_error(error) else "transport",
                source=source,
                retryable=True,
            ) from error
        finally:
            semaphore.release()

        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int):
            raise UpstreamError(
                "unexpected_content", source=source, retryable=False
            )
        if status_code in _REDIRECT_STATUSES and return_redirect:
            return response
        if status_code == 429:
            raise UpstreamError("rate_limited", source=source, retryable=True)
        if 500 <= status_code <= 599:
            raise UpstreamError(
                "upstream_unavailable", source=source, retryable=True
            )
        if status_code in _REDIRECT_STATUSES:
            raise UpstreamError("redirect", source=source, retryable=False)
        if status_code != 200:
            raise UpstreamError(
                "upstream_response", source=source, retryable=False
            )
        return response

    return call_with_retry(
        operation, cancelled=cancelled, runtime=services
    )


def _safe_parsed_hotels(
    html: str, location: str, checkin: str, checkout: str, min_stars: int
) -> list[dict[str, Any]]:
    destination = DESTINATION_BY_NAME.get(location, {})
    context = ParseContext(
        location=location,
        checkin=checkin,
        checkout=checkout,
        min_stars=min_stars,
        category=str(destination.get("category", "non_beachfront")),
        flight_cost=float(destination.get("flight_usd", 0)),
    )
    hotels = parse_hotel_cards(html, context)
    for hotel in hotels:
        candidate = hotel.get("url")
        if candidate is None:
            continue
        try:
            hotel["url"] = validate_google_hotel_url(candidate)
        except ValidationProblem:
            hotel["url"] = None
    return hotels


def search_hotels(
    location: str,
    checkin: str,
    checkout: str,
    min_stars: int = 5,
    *,
    cancelled: Callable[[], bool] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search Google Hotels through the per-app cache and upstream gate."""
    normalized_location = location.strip()
    key = (normalized_location, checkin, checkout, int(min_stars))
    services = _runtime_services(runtime)
    cache = services["search_cache"]

    cached = cache.get(key)
    if cached is not None:
        return _copy_hotels(cached.value)

    def load() -> tuple[str, ...]:
        hotel_data = [
            HotelData(
                checkin_date=checkin,
                checkout_date=checkout,
                location=normalized_location,
            )
        ]
        ths = THSData.from_interface(
            hotel_data=hotel_data,
            guests=Guests(adults=1),
            room_type="standard",
        )
        params = {
            "ths": ths.as_b64().decode("utf-8"),
            "hl": "en",
            "curr": "USD",
            "q": f"{min_stars} star hotels {normalized_location}",
        }
        city = normalized_location.replace(" ", "+").lower()
        response = _request_upstream(
            f"https://www.google.com/travel/hotels/{city}",
            params=params,
            source="google",
            cancelled=cancelled,
            runtime=services,
        )
        html = _response_html(response, "google")
        hotels = _safe_parsed_hotels(
            html, normalized_location, checkin, checkout, int(min_stars)
        )
        if not _recognized_search_page(html, hotels):
            raise UpstreamError(
                "unexpected_content", source="google", retryable=False
            )
        return _freeze_hotels(hotels)

    loaded = cache.get_or_load(key, load, ttl_seconds=EMPTY_SEARCH_CACHE_SECONDS)
    if not loaded.hit and loaded.value:
        cache.set(key, loaded.value, ttl_seconds=SEARCH_CACHE_SECONDS)
    return _copy_hotels(loaded.value)


def _authoritative_google_url(
    canonical_url: str, checkin: str | None, checkout: str | None
) -> str:
    if checkin is None and checkout is None:
        return canonical_url
    parts = urlsplit(canonical_url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() not in {"checkin", "checkout"}
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), "")
    )


def fetch_provider_prices(
    entity_url: str,
    *,
    checkin: str | None = None,
    checkout: str | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Fetch a validated Google entity page and parse provider prices."""
    canonical_url = _authoritative_google_url(
        validate_google_hotel_url(entity_url), checkin, checkout
    )
    services = _runtime_services(runtime)
    cache = services["provider_cache"]
    cache_key = (canonical_url, checkin or "", checkout or "", "USD")
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached.value)

    request_params = {"hl": "en", "curr": "USD"}
    if checkin is not None:
        request_params["checkin"] = checkin
    if checkout is not None:
        request_params["checkout"] = checkout

    def load() -> tuple[tuple[str, float], ...]:
        response = _request_upstream(
            canonical_url,
            params=request_params,
            source="google_provider",
            return_redirect=True,
            runtime=services,
        )
        status_code = getattr(response, "status_code", None)
        if status_code in _REDIRECT_STATUSES:
            location = _response_header(response, "location")
            if not location:
                raise UpstreamError(
                    "redirect", source="google_provider", retryable=False
                )
            target = urljoin(canonical_url, location)
            try:
                target = _authoritative_google_url(
                    validate_google_hotel_url(target), checkin, checkout
                )
            except ValidationProblem as error:
                raise UpstreamError(
                    "unsafe_redirect", source="google_provider", retryable=False
                ) from error
            response = _request_upstream(
                target,
                params=request_params,
                source="google_provider",
                return_redirect=True,
                runtime=services,
            )
            if getattr(response, "status_code", None) in _REDIRECT_STATUSES:
                raise UpstreamError(
                    "redirect", source="google_provider", retryable=False
                )
        html = _response_html(response, "google_provider")
        prices = parse_provider_prices(html)
        if not _recognized_provider_page(html, prices):
            raise UpstreamError(
                "unexpected_content", source="google_provider", retryable=False
            )
        for price in prices.values():
            if not isinstance(price, (int, float)) or not math.isfinite(float(price)):
                raise UpstreamError(
                    "unexpected_content", source="google_provider", retryable=False
                )
        return tuple(sorted(prices.items()))

    loaded = cache.get_or_load(
        cache_key, load, ttl_seconds=PROVIDER_CACHE_SECONDS
    )
    return dict(loaded.value)


def run_bounded[T, R](
    items: Sequence[T],
    fn: Callable[[T], R],
    max_workers: int,
    *,
    cancelled: Callable[[], bool] | None = None,
    on_complete: Callable[[T, R | None, UpstreamError | None], None] | None = None,
) -> tuple[list[tuple[T, R]], list[tuple[T, UpstreamError]]]:
    """Run at most ``max_workers`` outstanding futures and retain input order."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    indexed = list(enumerate(items))
    if not indexed:
        return [], []
    is_cancelled = cancelled or (lambda: False)
    successes: dict[int, tuple[T, R]] = {}
    failures: dict[int, tuple[T, UpstreamError]] = {}
    next_index = 0

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hotel-bound") as pool:
        futures: dict[Future[R], tuple[int, T]] = {}

        def schedule() -> None:
            nonlocal next_index
            while (
                next_index < len(indexed)
                and len(futures) < max_workers
                and not is_cancelled()
            ):
                index, item = indexed[next_index]
                next_index += 1
                futures[pool.submit(fn, item)] = (index, item)

        schedule()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda value: futures[value][0]):
                index, item = futures.pop(future)
                result: R | None = None
                error: UpstreamError | None = None
                try:
                    result = future.result()
                except UpstreamError as upstream_error:
                    error = upstream_error
                    failures[index] = (item, upstream_error)
                except Exception as unexpected:
                    error = UpstreamError(
                        "upstream_failure",
                        str(unexpected),
                        source="internal",
                        retryable=False,
                    )
                    failures[index] = (item, error)
                else:
                    successes[index] = (item, result)
                if on_complete is not None:
                    on_complete(item, result, error)
            schedule()

    return (
        [successes[index] for index in sorted(successes)],
        [failures[index] for index in sorted(failures)],
    )


def resolve_tripadvisor_key(hotel_name: str, location: str = "") -> object | None:
    """Look up the checked-in TripAdvisor key while preserving the legacy workflow."""
    del location
    if hotel_name in _TA_KEYS:
        return _TA_KEYS[hotel_name]
    name_lower = hotel_name.lower()
    for key, value in _TA_KEYS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return value
    return None


def _ota_search_url(
    provider_code: str, hotel_name: str, checkin: str, checkout: str
) -> str:
    query = quote_plus(hotel_name)
    urls = {
        "BookingCom": f"https://www.booking.com/searchresults.html?ss={query}&checkin={checkin}&checkout={checkout}",
        "Agoda": f"https://www.agoda.com/search?q={query}&checkIn={checkin}&los=1",
        "CtripTA": f"https://www.trip.com/hotels/list?keyword={query}&checkIn={checkin}&checkOut={checkout}",
        "Expedia": f"https://www.expedia.com/Hotel-Search?destination={query}&startDate={checkin}&endDate={checkout}",
        "HotelsCom": f"https://www.hotels.com/search.do?q-destination={query}&q-check-in={checkin}&q-check-out={checkout}",
        "Traveloka": f"https://www.traveloka.com/en-sg/hotel/search?spec={checkin}.{checkout}.1.0.HOTEL_GEO.{query}",
        "VioTA": f"https://www.vio.com/Hotels/Search?q={query}&checkin={checkin}&checkout={checkout}",
        "Vio": f"https://www.vio.com/Hotels/Search?q={query}&checkin={checkin}&checkout={checkout}",
    }
    return urls.get(provider_code, "")


def _read_xotelo_rates(
    response: object,
    services: Mapping[str, Any],
    deadline: float,
) -> list[object]:
    content_type = _response_header(response, "content-type")
    if content_type and "json" not in content_type.casefold():
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    raw = getattr(response, "raw", None)
    read_once = getattr(raw, "read1", None)
    iter_content = getattr(response, "iter_content", None)
    if not callable(read_once) and not callable(iter_content):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )

    def chunks():
        if callable(read_once):
            while True:
                chunk = read_once(
                    XOTELO_STREAM_CHUNK_BYTES, decode_content=True
                )
                if not chunk:
                    return
                yield chunk
        else:
            # A one-byte fallback keeps deadline checks meaningful for response
            # doubles and non-urllib3 clients whose iterator may fill each chunk.
            yield from iter_content(chunk_size=1)

    body = bytearray()
    stream = iter(chunks())
    try:
        while True:
            _raise_if_upstream_aborted(
                services, source="xotelo", deadline=deadline
            )
            try:
                chunk = next(stream)
            except StopIteration:
                _raise_if_upstream_aborted(
                    services, source="xotelo", deadline=deadline
                )
                break
            _raise_if_upstream_aborted(
                services, source="xotelo", deadline=deadline
            )
            if not isinstance(chunk, (bytes, bytearray)):
                raise UpstreamError(
                    "unexpected_content", source="xotelo", retryable=False
                )
            if chunk:
                if len(body) + len(chunk) > XOTELO_MAX_RESPONSE_BYTES:
                    raise UpstreamError(
                        "unexpected_content", source="xotelo", retryable=False
                    )
                body.extend(chunk)
    except UpstreamError:
        raise
    except (requests.Timeout, TimeoutError) as error:
        raise UpstreamError(
            "timeout", source="xotelo", retryable=True
        ) from error
    except requests.RequestException as error:
        raise UpstreamError(
            "timeout" if _is_timeout_error(error) else "transport",
            source="xotelo",
            retryable=True,
        ) from error
    except Exception as error:
        raise UpstreamError(
            "timeout" if _is_timeout_error(error) else "transport",
            source="xotelo",
            retryable=True,
        ) from error
    parse_error: BaseException | None = None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        parse_error = error
        payload = None
    _raise_if_upstream_aborted(
        services, source="xotelo", deadline=deadline
    )
    if parse_error is not None:
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        ) from parse_error
    if not isinstance(payload, Mapping):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    if payload.get("error") not in (None, False, ""):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    rates = result.get("rates")
    if not isinstance(rates, list) or len(rates) > XOTELO_MAX_RAW_RATES:
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    return rates


def _xotelo_text(
    value: object, *, maximum_length: int, required: bool
) -> str:
    if not isinstance(value, str):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    if (
        len(value) > maximum_length
        or any(
            unicodedata.category(character).startswith("C")
            for character in value
        )
    ):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    cleaned = value.strip()
    if required and not cleaned:
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    return cleaned


def _xotelo_number(
    value: object, *, allow_zero: bool
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        ) from error
    minimum_valid = number >= 0 if allow_zero else number > 0
    if (
        not minimum_valid
        or not math.isfinite(number)
        or number > XOTELO_MAX_RATE_VALUE
    ):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    return number


def _freeze_xotelo_rates(
    rates: Sequence[object],
    services: Mapping[str, Any],
    deadline: float,
) -> tuple[tuple[str, str, float, float], ...]:
    by_provider: dict[str, tuple[str, str, float, float]] = {}
    for raw_rate in rates:
        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        if not isinstance(raw_rate, Mapping):
            raise UpstreamError(
                "unexpected_content", source="xotelo", retryable=False
            )
        name = _xotelo_text(
            raw_rate.get("name"),
            maximum_length=XOTELO_MAX_PROVIDER_NAME_LENGTH,
            required=True,
        )
        code = _xotelo_text(
            raw_rate.get("code", ""),
            maximum_length=XOTELO_MAX_PROVIDER_CODE_LENGTH,
            required=False,
        )
        rate = _xotelo_number(raw_rate.get("rate"), allow_zero=False)
        tax = _xotelo_number(raw_rate.get("tax", 0), allow_zero=True)
        provider_key = "".join(
            character for character in name.casefold() if character.isalnum()
        )
        if not provider_key:
            raise UpstreamError(
                "unexpected_content", source="xotelo", retryable=False
            )
        existing = by_provider.get(provider_key)
        if existing is None:
            if len(by_provider) >= XOTELO_MAX_STORED_RATES:
                raise UpstreamError(
                    "unexpected_content", source="xotelo", retryable=False
                )
            by_provider[provider_key] = (name, code, rate, tax)
        elif rate < existing[2]:
            by_provider[provider_key] = (name, code, rate, tax)
    _raise_if_upstream_aborted(
        services, source="xotelo", deadline=deadline
    )
    return tuple(by_provider.values())


def _request_xotelo_rates(
    services: Mapping[str, Any],
    *,
    hotel_key: object,
    checkin: str,
    checkout: str,
    currency: str,
    deadline: float,
) -> tuple[tuple[str, str, float, float], ...]:
    response = None
    try:
        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        remaining = max(0.001, deadline - services["clock"]())
        response = services["xotelo_client"].get(
            "https://data.xotelo.com/api/rates",
            params={
                "hotel_key": hotel_key,
                "chk_in": checkin,
                "chk_out": checkout,
                "currency": currency,
            },
            timeout=TimeoutSauce(
                connect=min(CONNECT_TIMEOUT_SECONDS, remaining),
                read=min(XOTELO_READ_TIMEOUT_SECONDS, remaining),
                total=remaining,
            ),
            allow_redirects=False,
            stream=True,
        )
        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        status_code = getattr(response, "status_code", None)
        if status_code == 429:
            raise UpstreamError(
                "rate_limited", source="xotelo", retryable=True
            )
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            raise UpstreamError(
                "upstream_unavailable", source="xotelo", retryable=True
            )
        if status_code in _REDIRECT_STATUSES:
            raise UpstreamError(
                "redirect", source="xotelo", retryable=False
            )
        if status_code != 200:
            raise UpstreamError(
                "upstream_response", source="xotelo", retryable=False
            )
        rates = _read_xotelo_rates(response, services, deadline)
        frozen_rates = _freeze_xotelo_rates(rates, services, deadline)
        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        return frozen_rates
    except UpstreamError:
        raise
    except Exception as error:
        raise UpstreamError(
            "timeout" if _is_timeout_error(error) else "transport",
            source="xotelo",
            retryable=True,
        ) from error
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


def fetch_xotelo_prices(
    hotel_key: object,
    hotel_name: str,
    checkin: str,
    checkout: str,
    currency: str = "USD",
    *,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, object]]:
    """Fetch cached legacy Xotelo rates with one total-deadline-bounded attempt."""
    services = _runtime_services(runtime)
    cache = services["xotelo_cache"]
    key = (hotel_key, checkin, checkout, currency, "xotelo")
    transport_deadline = services["clock"]() + TOTAL_TIMEOUT_SECONDS
    publication_deadline = cache.deadline_after(TOTAL_TIMEOUT_SECONDS)

    def load() -> tuple[tuple[str, str, float, float], ...]:
        deadline = transport_deadline
        xotelo_semaphore = _acquire_semaphore(
            services["xotelo_semaphore"],
            services,
            source="xotelo",
            deadline=deadline,
        )
        try:
            upstream_semaphore = _acquire_upstream_slot(
                services, source="xotelo", deadline=deadline
            )
        except BaseException:
            xotelo_semaphore.release()
            raise
        outcome: dict[str, object] = {}
        finished = threading.Event()

        def request_in_background() -> None:
            try:
                outcome["rates"] = _request_xotelo_rates(
                    services,
                    hotel_key=hotel_key,
                    checkin=checkin,
                    checkout=checkout,
                    currency=currency,
                    deadline=deadline,
                )
            except BaseException as error:
                outcome["error"] = error
            finally:
                try:
                    upstream_semaphore.release()
                finally:
                    xotelo_semaphore.release()
                    finished.set()

        worker = threading.Thread(
            target=request_in_background,
            name="xotelo-upstream",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            upstream_semaphore.release()
            xotelo_semaphore.release()
            raise

        while not finished.is_set():
            remaining = deadline - services["clock"]()
            if remaining <= 0:
                raise UpstreamError(
                    "timeout", source="xotelo", retryable=True
                )
            finished.wait(
                min(UPSTREAM_SEMAPHORE_POLL_SECONDS, remaining)
            )

        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        frozen_rates = outcome.get("rates")
        if not isinstance(frozen_rates, tuple):
            raise UpstreamError(
                "unexpected_content", source="xotelo", retryable=False
            )
        _raise_if_upstream_aborted(
            services, source="xotelo", deadline=deadline
        )
        return frozen_rates

    try:
        loaded = cache.get_or_load(
            key,
            load,
            ttl_seconds=XOTELO_CACHE_SECONDS,
            not_after=publication_deadline,
        )
    except CacheDeadlineExceeded as error:
        raise UpstreamError(
            "timeout", source="xotelo", retryable=True
        ) from error
    return {
        name: {
            "rate": rate,
            "tax": tax,
            "url": _ota_search_url(code, hotel_name, checkin, checkout),
        }
        for name, code, rate, tax in loaded.value
    }


def _serialize_hotel_input(hotel: HotelInput) -> dict[str, object]:
    return {
        "name": hotel.name,
        "location": hotel.location,
        "url": hotel.url,
        "checkin": hotel.checkin.isoformat(),
        "checkout": hotel.checkout.isoformat(),
        "price": hotel.price,
        "rating": hotel.rating,
        "star_class": hotel.star_class,
        "confirmation": hotel.confirmation,
        "amenities": list(hotel.amenities),
        "category": hotel.category,
        "flight_cost": hotel.flight_cost,
    }


def _serialize_sweep_request(sweep: SweepRequest) -> dict[str, object]:
    return {
        "locations": list(sweep.locations),
        "startDate": sweep.start_date.isoformat(),
        "endDate": sweep.end_date.isoformat(),
        "nights": sweep.nights,
        "sampleCount": sweep.sample_count,
        "minStars": sweep.min_stars,
        "logicalCallCount": sweep.logical_call_count,
    }


def _merge_provider_prices(
    google_prices: Mapping[str, float], xotelo_prices: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for name, price in google_prices.items():
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(float(price))
        ):
            raise UpstreamError(
                "unexpected_content", source="google_provider", retryable=False
            )
        merged[name] = {"rate": float(price), "url": ""}
    normalizations = {
        "booking": "booking.com",
        "agoda": "agoda",
        "trip": "trip.com",
        "expedia": "expedia",
        "hotels": "hotels.com",
        "traveloka": "traveloka",
        "vio": "vio.com",
    }
    for name, raw_info in xotelo_prices.items():
        info = raw_info if isinstance(raw_info, Mapping) else {"rate": raw_info}
        rate = info.get("rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            continue
        if not math.isfinite(float(rate)):
            raise UpstreamError(
                "unexpected_content", source="xotelo", retryable=False
            )
        rate = float(rate)
        url = info.get("url", "")
        key = name.lower().replace(".com", "").replace(" ", "").strip()
        normalized_name = next(
            (normalized for prefix, normalized in normalizations.items() if prefix in key),
            name,
        )
        existing = merged.get(normalized_name)
        if existing is None or rate < existing["rate"]:
            merged[normalized_name] = {"rate": rate, "url": url}
        elif not existing.get("url") and url:
            existing["url"] = url
    return merged


def _google_provider_prices(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise UpstreamError(
            "unexpected_content", source="google_provider", retryable=False
        )
    result: dict[str, float] = {}
    for name, price in value.items():
        if (
            not isinstance(name, str)
            or isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(float(price))
        ):
            raise UpstreamError(
                "unexpected_content", source="google_provider", retryable=False
            )
        result[name] = float(price)
    return result


def _call_google_provider_prices(
    provider_callable: Callable[..., object],
    url: str,
    checkin: str,
    checkout: str,
) -> dict[str, float]:
    try:
        parameters = inspect.signature(provider_callable).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    names = {parameter.name for parameter in parameters}
    kwargs: dict[str, str] = {}
    if accepts_kwargs or "checkin" in names:
        kwargs["checkin"] = checkin
    if accepts_kwargs or "checkout" in names:
        kwargs["checkout"] = checkout
    return _google_provider_prices(provider_callable(url, **kwargs))


def _xotelo_provider_prices(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise UpstreamError(
            "unexpected_content", source="xotelo", retryable=False
        )
    copied = dict(value)
    _ensure_json_safe(copied, source="xotelo")
    return copied


def _provider_failure(error: Exception, source: str) -> UpstreamError:
    if isinstance(error, UpstreamError):
        if error.source == source:
            return error
        return UpstreamError(
            error.code,
            str(error),
            source=source,
            retryable=error.retryable,
        )
    return UpstreamError(
        "upstream_failure", str(error), source=source, retryable=False
    )


def _json_body() -> Mapping[str, Any]:
    try:
        payload = request.get_json(silent=False)
    except (BadRequest, UnsupportedMediaType) as error:
        raise InvalidJSON from error
    if not isinstance(payload, Mapping):
        raise InvalidJSON
    return payload


def _error_response(
    code: str, message: str, fields: Mapping[str, object], status: int
):
    return jsonify({"error": {"code": code, "message": message, "fields": dict(fields)}}), status


def _search_callable() -> Callable[..., list[dict[str, Any]]]:
    return _services()["search_hotels"]


def _call_search(
    search_callable: Callable[..., object],
    location: str,
    checkin: str,
    checkout: str,
    min_stars: int,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, object] = {"min_stars": min_stars}
    if cancelled is not None:
        try:
            parameters = inspect.signature(search_callable).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        if any(
            parameter.name == "cancelled"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            kwargs["cancelled"] = cancelled
    value = search_callable(location, checkin, checkout, **kwargs)
    return _validated_hotels(value, source="google")


def _representative_failure[T](
    failures: Sequence[tuple[T, UpstreamError]],
) -> UpstreamError:
    return next(
        (error for _item, error in failures if error.code == "timeout"),
        failures[0][1],
    )


def _execute_sweep(job: SweepJob, sweep: SweepRequest) -> dict[str, object] | None:
    date_pairs = sample_stay_dates(
        sweep.start_date, sweep.end_date, sweep.nights, sweep.sample_count
    )
    total_calls = len(date_pairs) * len(sweep.locations)
    completed_calls = 0
    date_results: list[dict[str, Any]] = []
    best_date_hotels: list[dict[str, Any]] = []
    failures_seen: list[tuple[str, UpstreamError]] = []
    saw_usable_data = False
    search_callable = _search_callable()
    job.set_progress(
        completed=0,
        total=total_calls,
        currentDate=0,
        totalDates=len(date_pairs),
        destinationsCompleted=0,
        totalDestinations=len(sweep.locations),
        currentLocation="",
    )

    for date_index, (checkin_date, checkout_date) in enumerate(date_pairs, start=1):
        if job.cancel_requested:
            return None
        checkin = checkin_date.isoformat()
        checkout = checkout_date.isoformat()
        date_completed = 0

        def search_location(
            location: str,
            search_checkin: str = checkin,
            search_checkout: str = checkout,
        ) -> list[dict[str, Any]]:
            if job.cancel_requested:
                return []
            return _call_search(
                search_callable,
                location,
                search_checkin,
                search_checkout,
                sweep.min_stars,
                cancelled=lambda: job.cancel_requested,
            )

        def publish_completion(
            location: str,
            _hotels: list[dict[str, Any]] | None,
            _error: UpstreamError | None,
            current_date_index: int = date_index,
        ) -> None:
            nonlocal completed_calls, date_completed
            if job.cancel_requested:
                return
            completed_calls += 1
            date_completed += 1
            job.set_progress(
                completed=completed_calls,
                total=total_calls,
                currentDate=current_date_index,
                totalDates=len(date_pairs),
                destinationsCompleted=date_completed,
                totalDestinations=len(sweep.locations),
                currentLocation=location,
            )

        successes, failures = run_bounded(
            list(sweep.locations),
            search_location,
            max_workers=min(4, len(sweep.locations)),
            cancelled=lambda: job.cancel_requested,
            on_complete=publish_completion,
        )
        if job.cancel_requested:
            return None

        all_hotels = [hotel for _location, hotels in successes for hotel in hotels]
        saw_usable_data = saw_usable_data or bool(all_hotels)
        failures_seen.extend(failures)
        all_hotels.sort(key=lambda hotel: float(hotel.get("price", float("inf"))))
        cheapest = all_hotels[0] if all_hotels else None
        result = {
            "checkin": checkin,
            "checkout": checkout,
            "cheapest_price": cheapest.get("price") if cheapest else None,
            "hotel_count": len(all_hotels),
            "cheapest_hotel": cheapest.get("name") if cheapest else None,
            "location": cheapest.get("location") if cheapest else None,
            "all_hotels": all_hotels,
        }
        for location, error in failures:
            if job.cancel_requested:
                return None
            job.add_warning(
                {
                    "checkin": checkin,
                    "location": location,
                    "code": error.code,
                    "source": error.source,
                }
            )
        if job.cancel_requested:
            return None
        if all_hotels or not failures:
            job.add_partial(
                {
                    "checkin": checkin,
                    "checkout": checkout,
                    "cheapest_price": result["cheapest_price"],
                    "cheapest_hotel": result["cheapest_hotel"],
                    "location": result["location"],
                    "hotel_count": result["hotel_count"],
                }
            )
        date_results.append(result)

    if job.cancel_requested:
        return None
    if failures_seen and not saw_usable_data:
        raise _representative_failure(failures_seen)
    valid_results = [
        result for result in date_results if result["cheapest_price"] is not None
    ]
    cheapest_entry = min(
        valid_results,
        key=lambda result: float(result["cheapest_price"]),
        default=None,
    )
    if cheapest_entry is not None:
        best_date_hotels = list(cheapest_entry["all_hotels"])
    categorized = {"beachfront": [], "non_beachfront": []}
    for hotel in best_date_hotels:
        category = hotel.get("category", "non_beachfront")
        if category not in categorized:
            category = "non_beachfront"
        categorized[category].append(hotel)
    public_dates = [
        {key: value for key, value in result.items() if key != "all_hotels"}
        for result in date_results
    ]
    cheapest_date = None
    if cheapest_entry is not None:
        cheapest_date = {
            "checkin": cheapest_entry["checkin"],
            "checkout": cheapest_entry["checkout"],
            "cheapest_price": cheapest_entry["cheapest_price"],
            "cheapest_hotel": cheapest_entry["cheapest_hotel"],
            "location": cheapest_entry["location"],
        }
    return {
        "locations": list(sweep.locations),
        "location": sweep.locations[0] if len(sweep.locations) == 1 else None,
        "dateRange": {
            "start": sweep.start_date.isoformat(),
            "end": sweep.end_date.isoformat(),
        },
        "nights": sweep.nights,
        "dates": public_dates,
        "cheapestDate": cheapest_date,
        "bestDateResults": categorized,
        "totalBeachfront": len(categorized["beachfront"]),
        "totalNonBeachfront": len(categorized["non_beachfront"]),
    }


def create_app(test_config: Mapping[str, object] | None = None) -> Flask:
    """Create one process-local application runtime with injectable boundaries."""
    static_dir = Path(__file__).with_name("static")
    application = _HotelFinderFlask(__name__, static_folder=None)
    if test_config:
        application.config.update(test_config)
    application.logger.setLevel(logging.INFO)
    formatter = _StructuredFormatter()
    if not application.logger.handlers:
        application.logger.addHandler(logging.StreamHandler())
    for handler in application.logger.handlers:
        handler.setFormatter(formatter)

    def configured(name: str, default: object) -> object:
        value = application.config.get(name)
        return default if value is None else value

    clock = configured("CLOCK", time.monotonic)
    configured_limit = application.config.get(
        "UPSTREAM_CONCURRENCY",
        os.environ.get(
            "HOTEL_FINDER_UPSTREAM_CONCURRENCY", DEFAULT_UPSTREAM_CONCURRENCY
        ),
    )
    upstream_limit = _bounded_upstream_limit(configured_limit)
    configured_queue_timeout = application.config.get(
        "UPSTREAM_QUEUE_TIMEOUT_SECONDS",
        os.environ.get(
            "HOTEL_FINDER_UPSTREAM_QUEUE_TIMEOUT_SECONDS",
            DEFAULT_UPSTREAM_QUEUE_TIMEOUT_SECONDS,
        ),
    )
    upstream_queue_timeout = _bounded_queue_timeout(
        configured_queue_timeout
    )
    search_cache = application.config.get("SEARCH_CACHE")
    if search_cache is None:
        search_cache = TTLCache(
            max_entries=256, ttl_seconds=SEARCH_CACHE_SECONDS, clock=clock
        )
    provider_cache = application.config.get("PROVIDER_CACHE")
    if provider_cache is None:
        provider_cache = TTLCache(
            max_entries=256, ttl_seconds=PROVIDER_CACHE_SECONDS, clock=clock
        )
    xotelo_cache = application.config.get("XOTELO_CACHE")
    if xotelo_cache is None:
        xotelo_cache = TTLCache(
            max_entries=256, ttl_seconds=XOTELO_CACHE_SECONDS, clock=clock
        )
    job_manager = application.config.get("JOB_MANAGER")
    if job_manager is None:
        job_manager = SweepJobManager(clock=clock)
    services: dict[str, Any] = {
        "search_cache": search_cache,
        "provider_cache": provider_cache,
        "xotelo_cache": xotelo_cache,
        "upstream_semaphore": threading.BoundedSemaphore(upstream_limit),
        "xotelo_semaphore": threading.BoundedSemaphore(1),
        "upstream_limit": upstream_limit,
        "upstream_queue_timeout": upstream_queue_timeout,
        "job_manager": job_manager,
        "clock": clock,
        "sleep": configured("SLEEP", time.sleep),
        "rng": configured("RNG", random.Random()),
        "client_factory": configured(
            "CLIENT_FACTORY", _thread_local_client_factory()
        ),
        "xotelo_client": configured("XOTELO_CLIENT", requests),
        "today": configured("TODAY_PROVIDER", date.today),
        "resolve_tripadvisor": configured(
            "RESOLVE_TRIPADVISOR", resolve_tripadvisor_key
        ),
    }
    services["search_hotels"] = configured(
        "SEARCH_HOTELS",
        lambda *args, **kwargs: search_hotels(
            *args, runtime=services, **kwargs
        ),
    )
    services["provider_prices"] = configured(
        "PROVIDER_PRICES",
        lambda *args, **kwargs: fetch_provider_prices(
            *args, runtime=services, **kwargs
        ),
    )
    services["xotelo_prices"] = configured(
        "XOTELO_PRICES",
        lambda *args, **kwargs: fetch_xotelo_prices(
            *args, runtime=services, **kwargs
        ),
    )
    application.extensions["hotel_finder"] = services

    allowed_origins = application.config.get("ALLOWED_ORIGINS")
    if allowed_origins is None:
        allowed_origins = os.environ.get("HOTEL_FINDER_ALLOWED_ORIGINS")
    origins = (
        [origin.strip() for origin in str(allowed_origins).split(",") if origin.strip()]
        if allowed_origins
        else []
    )
    if origins:
        CORS(
            application,
            resources={r"/api/*": {"origins": origins}},
            send_wildcard=False,
        )

    @application.before_request
    def assign_request_id() -> None:
        candidate = request.headers.get("X-Request-ID", "")
        g.request_id = candidate if _REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex

    @application.after_request
    def add_request_id(response):
        request_id = getattr(g, "request_id", uuid.uuid4().hex)
        response.headers["X-Request-ID"] = request_id
        application.logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "request_method": request.method,
                "request_path": request.path,
                "response_status": response.status_code,
            },
        )
        return response

    @application.errorhandler(InvalidJSON)
    def handle_invalid_json(_error):
        return _error_response(
            "invalid_json",
            "Request body must be valid JSON.",
            {"body": "must be a JSON object"},
            400,
        )

    @application.errorhandler(ValidationProblem)
    def handle_validation(error: ValidationProblem):
        return _error_response(
            error.code, "Request validation failed.", error.fields, 400
        )

    @application.errorhandler(JobConflict)
    def handle_job_conflict(error: JobConflict):
        return _error_response(
            "sweep_conflict",
            "Another sweep is already active.",
            {"activeJobId": error.active_job_id},
            409,
        )

    @application.errorhandler(JobNotFound)
    def handle_job_not_found(error: JobNotFound):
        return _error_response(
            "sweep_not_found",
            "Sweep job was not found.",
            {"jobId": error.job_id},
            404,
        )

    @application.errorhandler(UpstreamError)
    def handle_upstream(error: UpstreamError):
        status = 504 if error.code == "timeout" else 502
        return _error_response(
            error.code,
            str(error),
            {"source": error.source, "retryable": error.retryable},
            status,
        )

    @application.errorhandler(NotFound)
    def handle_not_found(_error: NotFound):
        if request.path == "/api" or request.path.startswith("/api/"):
            return _error_response(
                "not_found",
                "API route was not found.",
                {"path": request.path},
                404,
            )
        return "Not found", 404

    @application.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(_error: MethodNotAllowed):
        if request.path == "/api" or request.path.startswith("/api/"):
            return _error_response(
                "method_not_allowed",
                "HTTP method is not allowed for this API route.",
                {"path": request.path, "method": request.method},
                405,
            )
        return "Method not allowed", 405

    @application.get("/api/destinations")
    def get_destinations():
        return jsonify(DESTINATIONS)

    @application.post("/api/search")
    def api_search():
        query: SearchRequest = parse_search_request(
            _json_body(), KNOWN_LOCATIONS, today=_services()["today"]()
        )
        checkin = query.checkin.isoformat()
        checkout = query.checkout.isoformat()
        hotels = _call_search(
            _search_callable(),
            query.location,
            checkin,
            checkout,
            query.min_stars,
        )
        return jsonify(
            {
                "location": query.location,
                "checkin": checkin,
                "checkout": checkout,
                "minStars": query.min_stars,
                "nights": query.nights,
                "hotels": hotels,
                "warnings": [],
            }
        )

    @application.post("/api/search-all")
    def api_search_all():
        query: SearchAllRequest = parse_search_all_request(
            _json_body(), KNOWN_LOCATIONS, today=_services()["today"]()
        )
        checkin = query.checkin.isoformat()
        checkout = query.checkout.isoformat()
        locations = [
            location
            for location in query.locations
            if FLIGHT_BUDGET_MAP[location] <= query.max_flight
        ]
        search_callable = _search_callable()

        def search_location(location: str) -> list[dict[str, Any]]:
            return _call_search(
                search_callable,
                location,
                checkin,
                checkout,
                query.min_stars,
            )

        successes, failures = run_bounded(
            locations,
            search_location,
            max_workers=min(_services()["upstream_limit"], max(1, len(locations))),
        )
        results = {"beachfront": [], "non_beachfront": []}
        for location, hotels in successes:
            for hotel in hotels:
                category = hotel.get("category", DESTINATION_BY_NAME[location]["category"])
                if category not in results:
                    category = DESTINATION_BY_NAME[location]["category"]
                results[category].append(hotel)
        if failures and not any(results.values()):
            raise _representative_failure(failures)
        for category in results:
            results[category].sort(
                key=lambda hotel: float(hotel.get("price", float("inf")))
            )
        failed_destinations = [
            {"location": location, "code": error.code}
            for location, error in failures
        ]
        warnings = [
            {
                "location": location,
                "code": error.code,
                "source": error.source,
            }
            for location, error in failures
        ]
        return jsonify(
            {
                "checkin": checkin,
                "checkout": checkout,
                "maxFlight": query.max_flight,
                "minStars": query.min_stars,
                "nights": query.nights,
                "results": results,
                "totalBeachfront": len(results["beachfront"]),
                "totalNonBeachfront": len(results["non_beachfront"]),
                "failedDestinations": failed_destinations,
                "warnings": warnings,
            }
        )

    @application.post("/api/cheapest-dates")
    def api_cheapest_dates():
        payload = _json_body()
        sweep = parse_sweep_request(
            payload, KNOWN_LOCATIONS, today=_services()["today"]()
        )
        replace = payload.get("replace", False)
        if not isinstance(replace, bool):
            raise ValidationProblem({"replace": "must be a boolean"})
        initiating_request_id = g.request_id

        def runner(job: SweepJob, request_payload: SweepRequest):
            with application.app_context():
                application.logger.info(
                    "sweep_started",
                    extra={
                        "request_id": initiating_request_id,
                        "job_id": job.id,
                    },
                )
                try:
                    return _execute_sweep(job, request_payload)
                finally:
                    application.logger.info(
                        "sweep_finished",
                        extra={
                            "request_id": initiating_request_id,
                            "job_id": job.id,
                        },
                    )

        manager: SweepJobManager = _services()["job_manager"]
        job = manager.start(sweep, runner, replace=replace)
        application.logger.info(
            "sweep_queued",
            extra={"request_id": g.request_id, "job_id": job.id},
        )
        return (
            jsonify(
                {
                    "jobId": job.id,
                    "status": "queued",
                    "statusUrl": f"/api/sweeps/{job.id}",
                }
            ),
            202,
        )

    @application.get("/api/sweeps/<job_id>")
    def api_get_sweep(job_id: str):
        manager: SweepJobManager = _services()["job_manager"]
        return jsonify(manager.get(job_id).to_dict())

    @application.delete("/api/sweeps/<job_id>")
    def api_cancel_sweep(job_id: str):
        manager: SweepJobManager = _services()["job_manager"]
        snapshot = manager.cancel(job_id)
        application.logger.info(
            "sweep_cancel_requested",
            extra={"request_id": g.request_id, "job_id": job_id},
        )
        return jsonify(snapshot.to_dict()), 202

    @application.get("/api/sweep-progress")
    def api_sweep_progress():
        manager: SweepJobManager = _services()["job_manager"]
        snapshot = manager.active_snapshot()
        if snapshot is None:
            return jsonify(
                {
                    "active": False,
                    "phase": "done",
                    "current_date": 0,
                    "total_dates": 0,
                    "current_dest": "",
                    "dests_done": 0,
                    "total_dests": 0,
                    "dates_done": [],
                }
            )
        progress = snapshot.to_dict()["progress"]
        return jsonify(
            {
                "active": snapshot.status in {"queued", "running"},
                "jobId": snapshot.id,
                "status": snapshot.status,
                "phase": "searching",
                "current_date": progress.get("currentDate", 0),
                "total_dates": progress.get("totalDates", 0),
                "current_dest": progress.get("currentLocation", ""),
                "dests_done": progress.get("destinationsCompleted", 0),
                "total_dests": progress.get("totalDestinations", 0),
                "dates_done": snapshot.to_dict()["partial"],
            }
        )

    @application.post("/api/compare-prices")
    def api_compare_prices():
        query: CompareRequest = parse_compare_request(
            _json_body(), KNOWN_LOCATIONS, today=_services()["today"]()
        )
        services = _services()
        provider_prices = services["provider_prices"]
        resolve_tripadvisor = services["resolve_tripadvisor"]
        xotelo_prices_callable = services["xotelo_prices"]

        def enrich(
            hotel: HotelInput,
        ) -> tuple[
            dict[str, object], list[dict[str, str]], dict[str, str] | None
        ]:
            provider_warnings: list[dict[str, str]] = []
            provider_errors: list[UpstreamError] = []
            attempted_sources = 0
            google_prices: Mapping[str, float] = {}
            if hotel.url is not None:
                attempted_sources += 1
                try:
                    google_prices = _call_google_provider_prices(
                        provider_prices,
                        hotel.url,
                        hotel.checkin.isoformat(),
                        hotel.checkout.isoformat(),
                    )
                except Exception as error:
                    failure = _provider_failure(error, "google_provider")
                    provider_errors.append(failure)
                    provider_warnings.append(
                        {
                            "name": hotel.name,
                            "code": failure.code,
                            "source": failure.source,
                        }
                    )
            tripadvisor_key = None
            try:
                tripadvisor_key = resolve_tripadvisor(
                    hotel.name, hotel.location
                )
            except Exception as error:
                attempted_sources += 1
                failure = _provider_failure(error, "xotelo")
                provider_errors.append(failure)
                provider_warnings.append(
                    {
                        "name": hotel.name,
                        "code": failure.code,
                        "source": failure.source,
                    }
                )
            xotelo_prices: Mapping[str, object] = {}
            if tripadvisor_key is not None:
                attempted_sources += 1
                try:
                    xotelo_prices = _xotelo_provider_prices(
                        xotelo_prices_callable(
                            tripadvisor_key,
                            hotel.name,
                            hotel.checkin.isoformat(),
                            hotel.checkout.isoformat(),
                        )
                    )
                except Exception as error:
                    failure = _provider_failure(error, "xotelo")
                    provider_errors.append(failure)
                    provider_warnings.append(
                        {
                            "name": hotel.name,
                            "code": failure.code,
                            "source": failure.source,
                        }
                    )
            try:
                merged_prices = _merge_provider_prices(
                    google_prices, xotelo_prices
                )
            except Exception as error:
                source = (
                    error.source
                    if isinstance(error, UpstreamError)
                    else "internal"
                )
                failure = _provider_failure(error, source)
                provider_errors.append(failure)
                provider_warnings.append(
                    {
                        "name": hotel.name,
                        "code": failure.code,
                        "source": failure.source,
                    }
                )
                merged_prices = {}
            failed_hotel = None
            if attempted_sources and len(provider_errors) >= attempted_sources:
                failure = next(
                    (
                        error
                        for error in provider_errors
                        if error.code == "timeout"
                    ),
                    provider_errors[0],
                )
                failed_hotel = {
                    "name": hotel.name,
                    "code": failure.code,
                    "source": failure.source,
                }
            return (
                {
                    **_serialize_hotel_input(hotel),
                    "providers": merged_prices,
                    "xotelo_key": tripadvisor_key,
                },
                provider_warnings,
                failed_hotel,
            )

        successes, failures = run_bounded(
            list(query.hotels), enrich, max_workers=min(4, len(query.hotels))
        )
        enriched = [value[0] for _hotel, value in successes]
        warnings = [
            warning
            for _hotel, (_value, hotel_warnings, _failed) in successes
            for warning in hotel_warnings
        ]
        failed_hotels = [
            failed
            for _hotel, (_value, _warnings, failed) in successes
            if failed is not None
        ]
        for hotel, error in failures:
            enriched.append(
                {
                    **_serialize_hotel_input(hotel),
                    "providers": {},
                    "xotelo_key": None,
                }
            )
            failure = {
                "name": hotel.name,
                "code": error.code,
                "source": error.source,
            }
            failed_hotels.append(failure)
            warnings.append(failure)
        enriched.sort(key=lambda hotel: float(hotel["price"]))
        return jsonify(
            {
                "hotels": enriched,
                "checkin": query.checkin.isoformat(),
                "checkout": query.checkout.isoformat(),
                "failedHotels": failed_hotels,
                "warnings": warnings,
            }
        )

    @application.get("/api/health")
    def api_health():
        services = _services()
        active = services["job_manager"].active_snapshot()
        return jsonify(
            {
                "status": "ok",
                "version": __version__,
                "primp": {
                    "profile": IMPERSONATE_PROFILE,
                    "verify": True,
                    "connectTimeoutSeconds": CONNECT_TIMEOUT_SECONDS,
                    "readTimeoutSeconds": READ_TIMEOUT_SECONDS,
                    "totalTimeoutSeconds": TOTAL_TIMEOUT_SECONDS,
                    "followRedirects": False,
                },
                "caches": {
                    "search": services["search_cache"].size(),
                    "providers": services["provider_cache"].size(),
                    "xotelo": services["xotelo_cache"].size(),
                },
                "activeSweep": active.to_dict() if active is not None else None,
                "runtime": {
                    "workers": 1,
                    "upstreamConcurrency": services["upstream_limit"],
                },
            }
        )

    @application.get("/")
    def serve_index():
        if static_dir.joinpath("index.html").is_file():
            return send_from_directory(str(static_dir), "index.html")
        return "API running. Frontend at http://localhost:5173", 200

    @application.get("/<path:path>")
    def serve_static(path: str):
        if path == "api" or path.startswith("api/"):
            return _error_response(
                "not_found", "API route was not found.", {"path": f"/{path}"}, 404
            )
        static_root = static_dir.resolve()
        requested = static_root.joinpath(path).resolve()
        try:
            requested.relative_to(static_root)
        except ValueError:
            return "Not found", 404
        if requested.is_file():
            return send_from_directory(str(static_root), path)
        if static_root.joinpath("index.html").is_file():
            return send_from_directory(str(static_root), "index.html")
        return "Not found", 404

    return application


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
