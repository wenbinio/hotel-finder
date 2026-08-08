from datetime import date

import pytest

from hotel_finder.validation import (
    CompareRequest,
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

TODAY = date(2026, 8, 8)
KNOWN_LOCATIONS = {"Bangkok", "Phuket"}


def hotel_payload(**overrides):
    payload = {
        "name": "Test Grand Hotel",
        "location": "Bangkok",
        "url": "https://www.google.com/travel/hotels/entity/abc",
        "checkin": "2026-08-09",
        "checkout": "2026-08-10",
        "price": 220,
        "rating": 4.7,
        "star_class": 5,
        "confirmation": "html",
        "amenities": ["Pool", "Spa"],
        "category": "non_beachfront",
        "flight_cost": 126,
    }
    payload.update(overrides)
    return payload


def test_search_parses_typed_future_request():
    request = parse_search_request(
        {"location": "Bangkok", "checkin": "2026-08-09", "checkout": "2026-08-14", "minStars": 4},
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert request == SearchRequest(
        location="Bangkok",
        checkin=date(2026, 8, 9),
        checkout=date(2026, 8, 14),
        min_stars=4,
    )
    assert request.nights == 5


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"location": "Unknown", "checkin": "2026-08-09", "checkout": "2026-08-10"}, "location"),
        ({"location": "Bangkok", "checkin": "2026-08-07", "checkout": "2026-08-10"}, "checkin"),
        ({"location": "Bangkok", "checkin": "2026-08-09", "checkout": "2026-08-09"}, "checkout"),
        ({"location": "Bangkok", "checkin": "2026-08-09", "checkout": "2026-08-10", "minStars": "5"}, "minStars"),
    ],
)
def test_search_rejects_invalid_or_ambiguous_fields(payload, field):
    with pytest.raises(ValidationProblem) as error:
        parse_search_request(payload, KNOWN_LOCATIONS, today=TODAY)

    assert field in error.value.fields


def test_search_all_defaults_to_known_locations_and_respects_flight_budget():
    request = parse_search_all_request(
        {"checkin": "2026-08-09", "checkout": "2026-08-10", "maxFlight": 300},
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert request == SearchAllRequest(
        locations=("Bangkok", "Phuket"),
        checkin=date(2026, 8, 9),
        checkout=date(2026, 8, 10),
        min_stars=5,
        max_flight=300,
    )


def test_search_all_accepts_established_destinations_field():
    request = parse_search_all_request(
        {
            "destinations": ["Phuket", "Bangkok"],
            "checkin": "2026-08-09",
            "checkout": "2026-08-10",
        },
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert request.locations == ("Phuket", "Bangkok")


def test_search_all_rejects_unknown_destination_from_established_field():
    with pytest.raises(ValidationProblem) as error:
        parse_search_all_request(
            {
                "destinations": ["Atlantis"],
                "checkin": "2026-08-09",
                "checkout": "2026-08-10",
            },
            KNOWN_LOCATIONS,
            today=TODAY,
        )

    assert "destinations[0]" in error.value.fields


def test_search_all_rejects_conflicting_destination_aliases():
    with pytest.raises(ValidationProblem) as error:
        parse_search_all_request(
            {
                "destinations": ["Bangkok"],
                "locations": ["Phuket"],
                "checkin": "2026-08-09",
                "checkout": "2026-08-10",
            },
            KNOWN_LOCATIONS,
            today=TODAY,
        )

    assert "destinations" in error.value.fields


def test_sweep_rejects_past_and_excessive_work():
    with pytest.raises(ValidationProblem, match="startDate"):
        parse_sweep_request(
            {"locations": ["Bangkok"], "startDate": "2026-08-07", "endDate": "2026-08-20"},
            {"Bangkok"},
            today=TODAY,
        )
    known = {f"City {index}" for index in range(21)}
    with pytest.raises(ValidationProblem, match="200"):
        parse_sweep_request(
            {
                "locations": sorted(known),
                "startDate": "2026-08-09",
                "endDate": "2026-11-07",
                "sampleCount": 10,
            },
            known,
            today=TODAY,
        )


def test_sweep_uses_no_more_than_requested_evenly_spaced_future_dates():
    request = parse_sweep_request(
        {
            "locations": ["Bangkok"],
            "startDate": "2026-08-09",
            "endDate": "2026-08-19",
            "nights": 2,
            "sampleCount": 3,
        },
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert request == SweepRequest(
        locations=("Bangkok",),
        start_date=date(2026, 8, 9),
        end_date=date(2026, 8, 19),
        nights=2,
        sample_count=3,
        min_stars=5,
    )
    assert sample_stay_dates(
        request.start_date, request.end_date, request.nights, request.sample_count
    ) == [
        (date(2026, 8, 9), date(2026, 8, 11)),
        (date(2026, 8, 14), date(2026, 8, 16)),
        (date(2026, 8, 19), date(2026, 8, 21)),
    ]
    assert request.logical_call_count == 3


@pytest.mark.parametrize(
    "url",
    [
        "http://www.google.com/travel/hotels/entity/abc",
        "https://evil.example/travel/hotels/entity/abc",
        "https://127.0.0.1/latest/meta-data",
        "https://www.google.com:444/travel/hotels/entity/abc",
        "https://www.google.com/travel/hotels/search/abc",
    ],
)
def test_provider_url_allowlist_rejects_non_google_targets(url):
    with pytest.raises(ValidationProblem):
        validate_google_hotel_url(url)


def test_provider_url_allowlist_accepts_google_entity_url():
    assert validate_google_hotel_url("https://www.google.com/travel/hotels/entity/abc?foo=bar") == (
        "https://www.google.com/travel/hotels/entity/abc?foo=bar"
    )


def test_provider_url_allowlist_returns_canonical_bounded_url():
    assert validate_google_hotel_url(
        "HTTPS://WWW.GOOGLE.COM:443/travel/hotels/entity/abc?foo=bar#section"
    ) == "https://www.google.com/travel/hotels/entity/abc?foo=bar"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com/travel/hotels/entity/abc\r\nX-Test: injected",
        f"https://www.google.com/travel/hotels/entity/abc?query={'x' * 1025}",
        f"https://www.google.com/travel/hotels/entity/{'x' * 2048}",
    ],
)
def test_provider_url_allowlist_rejects_controls_and_oversize_query(url):
    with pytest.raises(ValidationProblem):
        validate_google_hotel_url(url)


def test_compare_round_trips_safe_display_metadata():
    request = parse_compare_request(
        {
            "checkin": "2026-08-09",
            "checkout": "2026-08-10",
            "hotels": [
                hotel_payload(
                    url="HTTPS://WWW.GOOGLE.COM:443/travel/hotels/entity/abc#discarded"
                )
            ],
        },
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert isinstance(request, CompareRequest)
    assert request.hotels[0].name == "Test Grand Hotel"
    assert request.hotels[0].url == "https://www.google.com/travel/hotels/entity/abc"
    assert request.hotels[0].checkin == date(2026, 8, 9)
    assert request.hotels[0].price == 220.0
    assert request.hotels[0].rating == 4.7
    assert request.hotels[0].star_class == 5
    assert request.hotels[0].confirmation == "html"
    assert request.hotels[0].amenities == ("Pool", "Spa")
    assert request.hotels[0].category == "non_beachfront"
    assert request.hotels[0].flight_cost == 126.0


def test_compare_top_level_dates_are_authoritative_over_stale_hotel_metadata():
    request = parse_compare_request(
        {
            "checkin": "2026-08-09",
            "checkout": "2026-08-14",
            "hotels": [
                hotel_payload(checkin="2026-09-01", checkout="2026-09-02")
            ],
        },
        KNOWN_LOCATIONS,
        today=TODAY,
    )

    assert request.hotels[0].checkin == date(2026, 8, 9)
    assert request.hotels[0].checkout == date(2026, 8, 14)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"price": float("nan")}, "price"),
        ({"price": True}, "price"),
        ({"price": 10**400}, "price"),
        ({"rating": float("nan")}, "rating"),
        ({"rating": True}, "rating"),
        ({"star_class": True}, "star_class"),
        ({"flight_cost": float("nan")}, "flight_cost"),
        ({"flight_cost": False}, "flight_cost"),
    ],
)
def test_compare_rejects_nan_and_boolean_metadata(overrides, field):
    with pytest.raises(ValidationProblem) as error:
        parse_compare_request(
            {
                "checkin": "2026-08-09",
                "checkout": "2026-08-10",
                "hotels": [hotel_payload(**overrides)],
            },
            KNOWN_LOCATIONS,
            today=TODAY,
        )

    assert f"hotels[0].{field}" in error.value.fields


@pytest.mark.parametrize(
    ("payload_overrides", "known_locations", "expected_field"),
    [
        ({"name": "N" * 201}, KNOWN_LOCATIONS, "hotels[0].name"),
        ({"name": "Safe\nInjected"}, KNOWN_LOCATIONS, "hotels[0].name"),
        ({"location": "B" * 101}, {"B" * 101}, "hotels[0].location"),
        ({"location": "Bangkok\x00"}, {"Bangkok\x00"}, "hotels[0].location"),
    ],
)
def test_compare_rejects_oversize_or_controlled_names_and_locations(
    payload_overrides, known_locations, expected_field
):
    with pytest.raises(ValidationProblem) as error:
        parse_compare_request(
            {
                "checkin": "2026-08-09",
                "checkout": "2026-08-10",
                "hotels": [hotel_payload(**payload_overrides)],
            },
            known_locations,
            today=TODAY,
        )

    assert expected_field in error.value.fields


def test_compare_rejects_control_character_in_date():
    with pytest.raises(ValidationProblem) as error:
        parse_compare_request(
            {
                "checkin": "2026-08-09\n",
                "checkout": "2026-08-10",
                "hotels": [hotel_payload()],
            },
            KNOWN_LOCATIONS,
            today=TODAY,
        )

    assert "checkin" in error.value.fields
