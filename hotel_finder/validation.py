"""Typed, side-effect-free validation for hotel finder API requests."""

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

MIN_STARS = 3
MAX_STARS = 5
MAX_NIGHTS = 30
MAX_FLIGHT_USD = 500
MAX_SWEEP_DAYS = 365
MAX_SAMPLE_COUNT = 10
MAX_LOGICAL_CALLS = 200
MAX_COMPARE_HOTELS = 15
MAX_HOTEL_NAME_LENGTH = 200
MAX_LOCATION_LENGTH = 100
MAX_AMENITIES = 32
MAX_AMENITY_LENGTH = 100
MAX_URL_LENGTH = 2048
MAX_URL_QUERY_LENGTH = 1024
GOOGLE_HOTEL_HOST = "www.google.com"
GOOGLE_HOTEL_PATH_PREFIX = "/travel/hotels/entity/"
HOTEL_CATEGORIES = {"beachfront", "non_beachfront"}
STAR_CONFIRMATIONS = {"html", "brand"}
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


class ValidationProblem(ValueError):
    """A request cannot be safely normalized into the service's input model."""

    code = "validation_error"

    def __init__(self, fields: Mapping[str, str]):
        self.fields = dict(fields)
        message = "; ".join(f"{field}: {detail}" for field, detail in self.fields.items())
        super().__init__(message)


@dataclass(frozen=True)
class SearchRequest:
    location: str
    checkin: date
    checkout: date
    min_stars: int

    @property
    def nights(self) -> int:
        return (self.checkout - self.checkin).days


@dataclass(frozen=True)
class SearchAllRequest:
    locations: tuple[str, ...]
    checkin: date
    checkout: date
    min_stars: int
    max_flight: float

    @property
    def nights(self) -> int:
        return (self.checkout - self.checkin).days


@dataclass(frozen=True)
class SweepRequest:
    locations: tuple[str, ...]
    start_date: date
    end_date: date
    nights: int
    sample_count: int
    min_stars: int

    @property
    def logical_call_count(self) -> int:
        return len(self.locations) * len(
            sample_stay_dates(self.start_date, self.end_date, self.nights, self.sample_count)
        )


@dataclass(frozen=True)
class HotelInput:
    name: str
    location: str
    url: str | None
    checkin: date
    checkout: date
    price: float
    rating: float | None
    star_class: int
    confirmation: str
    amenities: tuple[str, ...]
    category: str
    flight_cost: float


@dataclass(frozen=True)
class CompareRequest:
    hotels: tuple[HotelInput, ...]
    checkin: date
    checkout: date


def _problem(field: str, detail: str) -> ValidationProblem:
    return ValidationProblem({field: detail})


def _object(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise _problem("body", "must be a JSON object")
    return payload


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _normalized_string(
    value: Any, field: str, *, maximum: int | None = None, strip: bool = True
) -> str:
    if not isinstance(value, str):
        raise _problem(field, "must be a non-empty string")
    if _contains_control(value):
        raise _problem(field, "must not contain control characters")
    normalized = value.strip() if strip else value
    if not normalized:
        raise _problem(field, "must be a non-empty string")
    if maximum is not None and len(normalized) > maximum:
        raise _problem(field, f"must contain at most {maximum} characters")
    return normalized


def _string(
    payload: Mapping[str, Any],
    field: str,
    *,
    maximum: int | None = None,
    strip: bool = True,
) -> str:
    return _normalized_string(payload.get(field), field, maximum=maximum, strip=strip)


def _date(payload: Mapping[str, Any], field: str) -> date:
    value = _string(payload, field, maximum=10, strip=False)
    if _ISO_DATE.fullmatch(value) is None:
        raise _problem(field, "must use YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise _problem(field, "must be a real calendar date") from error


def _integer(payload: Mapping[str, Any], field: str, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _problem(field, "must be an integer")
    if not minimum <= value <= maximum:
        raise _problem(field, f"must be between {minimum} and {maximum}")
    return value


def _number(payload: Mapping[str, Any], field: str, default: float, minimum: float, maximum: float) -> float:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _problem(field, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise _problem(field, "must be a finite number") from error
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise _problem(field, f"must be between {minimum:g} and {maximum:g}")
    return normalized


def _required_integer(
    payload: Mapping[str, Any], field: str, minimum: int, maximum: int
) -> int:
    if field not in payload:
        raise _problem(field, "is required")
    return _integer(payload, field, minimum, minimum, maximum)


def _required_number(
    payload: Mapping[str, Any], field: str, minimum: float, maximum: float
) -> float:
    if field not in payload:
        raise _problem(field, "is required")
    return _number(payload, field, minimum, minimum, maximum)


def _known_location(value: str, field: str, known_locations: set[str]) -> str:
    if value not in known_locations:
        raise _problem(field, "must name a configured destination")
    return value


def _locations_for_field(
    payload: Mapping[str, Any], known_locations: set[str], field: str
) -> tuple[str, ...]:
    value = payload.get(field)
    if value is None:
        return tuple(sorted(known_locations))
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _problem(field, "must be an array of configured destinations")
    if not value:
        raise _problem(field, "must contain at least one destination")
    locations: list[str] = []
    for index, location in enumerate(value):
        item_field = f"{field}[{index}]"
        normalized = _normalized_string(location, item_field, maximum=MAX_LOCATION_LENGTH)
        _known_location(normalized, item_field, known_locations)
        locations.append(normalized)
    if len(set(locations)) != len(locations):
        raise _problem(field, "must not contain duplicates")
    return tuple(locations)


def _locations(payload: Mapping[str, Any], known_locations: set[str]) -> tuple[str, ...]:
    return _locations_for_field(payload, known_locations, "locations")


def _search_all_locations(
    payload: Mapping[str, Any], known_locations: set[str]
) -> tuple[str, ...]:
    has_locations = "locations" in payload
    has_destinations = "destinations" in payload
    if has_locations and has_destinations:
        locations = _locations_for_field(payload, known_locations, "locations")
        destinations = _locations_for_field(payload, known_locations, "destinations")
        if locations != destinations:
            raise _problem("destinations", "conflicts with locations")
        return locations
    if has_destinations:
        return _locations_for_field(payload, known_locations, "destinations")
    return _locations(payload, known_locations)


def _stay_dates(
    payload: Mapping[str, Any],
    checkin_field: str,
    checkout_field: str,
    today: date,
) -> tuple[date, date]:
    checkin = _date(payload, checkin_field)
    checkout = _date(payload, checkout_field)
    if checkin < today:
        raise _problem(checkin_field, "must not be in the past")
    if checkout <= checkin:
        raise _problem(checkout_field, "must be after check-in")
    if (checkout - checkin).days > MAX_NIGHTS:
        raise _problem(checkout_field, f"stay must not exceed {MAX_NIGHTS} nights")
    return checkin, checkout


def parse_search_request(
    payload: Any, known_locations: set[str], *, today: date | None = None
) -> SearchRequest:
    """Validate one named destination and a bounded future stay."""
    body = _object(payload)
    current_day = today or date.today()
    location = _known_location(
        _string(body, "location", maximum=MAX_LOCATION_LENGTH), "location", known_locations
    )
    checkin, checkout = _stay_dates(body, "checkin", "checkout", current_day)
    min_stars = _integer(body, "minStars", 5, MIN_STARS, MAX_STARS)
    return SearchRequest(location, checkin, checkout, min_stars)


def parse_search_all_request(
    payload: Any, known_locations: set[str], *, today: date | None = None
) -> SearchAllRequest:
    """Validate an all-destination search while keeping its inputs bounded."""
    body = _object(payload)
    current_day = today or date.today()
    locations = _search_all_locations(body, known_locations)
    checkin, checkout = _stay_dates(body, "checkin", "checkout", current_day)
    min_stars = _integer(body, "minStars", 5, MIN_STARS, MAX_STARS)
    max_flight = _number(body, "maxFlight", 300, 0, MAX_FLIGHT_USD)
    return SearchAllRequest(locations, checkin, checkout, min_stars, max_flight)


def sample_stay_dates(
    start_date: date, end_date: date, nights: int, sample_count: int
) -> list[tuple[date, date]]:
    """Return evenly distributed, distinct check-ins in an inclusive date window."""
    days_between = (end_date - start_date).days
    if days_between < 0:
        raise ValueError("end_date must not precede start_date")
    effective_count = min(sample_count, days_between + 1)
    if effective_count < 1:
        return []
    if effective_count == 1:
        checkins = [start_date]
    else:
        checkins = [
            start_date + timedelta(days=(days_between * index) // (effective_count - 1))
            for index in range(effective_count)
        ]
    return [(checkin, checkin + timedelta(days=nights)) for checkin in checkins]


def parse_sweep_request(
    payload: Any, known_locations: set[str], *, today: date | None = None
) -> SweepRequest:
    """Validate a cancellable date sweep before it can schedule upstream work."""
    body = _object(payload)
    current_day = today or date.today()
    locations = _locations(body, known_locations)
    start_date = _date(body, "startDate")
    end_date = _date(body, "endDate")
    if start_date < current_day:
        raise _problem("startDate", "must not be in the past")
    if end_date <= start_date:
        raise _problem("endDate", "must be after startDate")
    if (end_date - start_date).days > MAX_SWEEP_DAYS:
        raise _problem("endDate", f"range must not exceed {MAX_SWEEP_DAYS} days")
    nights = _integer(body, "nights", 1, 1, MAX_NIGHTS)
    sample_count = _integer(body, "sampleCount", 6, 1, MAX_SAMPLE_COUNT)
    min_stars = _integer(body, "minStars", 5, MIN_STARS, MAX_STARS)
    request = SweepRequest(locations, start_date, end_date, nights, sample_count, min_stars)
    if request.logical_call_count > MAX_LOGICAL_CALLS:
        raise _problem("sampleCount", f"sweep exceeds {MAX_LOGICAL_CALLS} logical hotel searches")
    return request


def validate_google_hotel_url(url: Any) -> str:
    """Accept only a direct HTTPS Google Hotels entity URL before any fetch occurs."""
    if not isinstance(url, str) or not url:
        raise _problem("url", "must be a Google Hotels entity URL")
    if len(url) > MAX_URL_LENGTH:
        raise _problem("url", f"must contain at most {MAX_URL_LENGTH} characters")
    if _contains_control(url) or any(character.isspace() for character in url):
        raise _problem("url", "must not contain whitespace or control characters")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise _problem("url", "must be a valid URL") from error
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != GOOGLE_HOTEL_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith(GOOGLE_HOTEL_PATH_PREFIX)
    ):
        raise _problem("url", "must be an HTTPS www.google.com hotel entity URL")
    if len(parsed.query) > MAX_URL_QUERY_LENGTH:
        raise _problem("url", f"query must contain at most {MAX_URL_QUERY_LENGTH} characters")
    if _contains_control(unquote(parsed.path)) or _contains_control(unquote(parsed.query)):
        raise _problem("url", "must not contain encoded control characters")
    canonical = urlunsplit(("https", GOOGLE_HOTEL_HOST, parsed.path, parsed.query, ""))
    if len(canonical) > MAX_URL_LENGTH:
        raise _problem("url", f"must contain at most {MAX_URL_LENGTH} characters")
    return canonical


def _amenities(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("amenities")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _problem("amenities", "must be an array of strings")
    if len(value) > MAX_AMENITIES:
        raise _problem("amenities", f"must contain at most {MAX_AMENITIES} items")
    return tuple(
        _normalized_string(amenity, f"amenities[{index}]", maximum=MAX_AMENITY_LENGTH)
        for index, amenity in enumerate(value)
    )


def _hotel_input(
    payload: Any, index: int, known_locations: set[str], checkin: date, checkout: date, today: date
) -> HotelInput:
    if not isinstance(payload, Mapping):
        raise _problem(f"hotels[{index}]", "must be an object")
    prefix = f"hotels[{index}]"
    try:
        name = _string(payload, "name", maximum=MAX_HOTEL_NAME_LENGTH)
        location = _known_location(
            _string(payload, "location", maximum=MAX_LOCATION_LENGTH),
            "location",
            known_locations,
        )
        url = payload.get("url")
        if url is not None:
            url = validate_google_hotel_url(url)
        hotel_checkin, hotel_checkout = checkin, checkout
        if "checkin" in payload or "checkout" in payload:
            hotel_checkin, hotel_checkout = _stay_dates(payload, "checkin", "checkout", today)
        price = _required_number(payload, "price", 0.01, 1500)
        rating = (
            None
            if payload.get("rating") is None
            else _required_number(payload, "rating", 0, 5)
        )
        star_class = _required_integer(payload, "star_class", 1, 5)
        confirmation = _string(payload, "confirmation", maximum=16, strip=False)
        if confirmation not in STAR_CONFIRMATIONS:
            raise _problem("confirmation", "must be html or brand")
        amenities = _amenities(payload)
        category = _string(payload, "category", maximum=32, strip=False)
        if category not in HOTEL_CATEGORIES:
            raise _problem("category", "must be beachfront or non_beachfront")
        flight_cost = _required_number(payload, "flight_cost", 0, MAX_FLIGHT_USD)
    except ValidationProblem as error:
        field, detail = next(iter(error.fields.items()))
        raise _problem(f"{prefix}.{field}", detail) from error
    return HotelInput(
        name,
        location,
        url,
        hotel_checkin,
        hotel_checkout,
        price,
        rating,
        star_class,
        confirmation,
        amenities,
        category,
        flight_cost,
    )


def parse_compare_request(
    payload: Any, known_locations: set[str], *, today: date | None = None
) -> CompareRequest:
    """Validate at most fifteen parsed hotels before provider comparison."""
    body = _object(payload)
    current_day = today or date.today()
    checkin, checkout = _stay_dates(body, "checkin", "checkout", current_day)
    hotels_value = body.get("hotels")
    if isinstance(hotels_value, str) or not isinstance(hotels_value, Sequence):
        raise _problem("hotels", "must be an array of hotel objects")
    if not hotels_value:
        raise _problem("hotels", "must contain at least one hotel")
    if len(hotels_value) > MAX_COMPARE_HOTELS:
        raise _problem("hotels", f"must contain at most {MAX_COMPARE_HOTELS} hotels")
    hotels = tuple(
        _hotel_input(hotel, index, known_locations, checkin, checkout, current_day)
        for index, hotel in enumerate(hotels_value)
    )
    return CompareRequest(hotels, checkin, checkout)
