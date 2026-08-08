from hotel_finder.parsing import ParseContext, parse_hotel_cards, parse_provider_prices


def test_hotel_card_extracts_star_price_and_metadata():
    html = """<div class="uaTTDe"><h2 class="BgYkof">Test Grand Hotel</h2>
    <span class="KFi5wf lA0BZ">4.7</span><span class="ne5qie Ih19Ad">5-star hotel</span>
    <span class="LtjZ2d">Pool</span><span>$220</span><a href="/travel/hotels/entity/abc">open</a></div>"""

    hotels = parse_hotel_cards(
        html,
        ParseContext("Bangkok", "2026-08-09", "2026-08-10", 5, "non_beachfront", 126),
    )

    assert hotels == [
        {
            "name": "Test Grand Hotel",
            "price": 220.0,
            "rating": 4.7,
            "star_class": 5,
            "confirmation": "html",
            "amenities": ["Pool"],
            "url": "https://www.google.com/travel/hotels/entity/abc",
            "location": "Bangkok",
            "checkin": "2026-08-09",
            "checkout": "2026-08-10",
            "category": "non_beachfront",
            "flight_cost": 126,
        }
    ]


def test_hotel_parser_falls_back_to_known_amenities_and_discards_high_prices():
    html = """
    <div class="uaTTDe"><h2 class="Cx32Ud">The Westin Bangkok</h2>
    <span>Free Wi-Fi Spa Airport shuttle</span><span>$1,499</span></div>
    <div class="uaTTDe"><h2 class="BgYkof">The Westin Overpriced</h2>
    <span>$1,501</span></div>
    """

    hotels = parse_hotel_cards(
        html,
        ParseContext("Bangkok", "2026-08-09", "2026-08-10", 4, "non_beachfront", 126),
    )

    assert hotels == [
        {
            "name": "The Westin Bangkok",
            "price": 1499.0,
            "rating": None,
            "star_class": 4,
            "confirmation": "brand",
            "amenities": ["Free Wi-Fi", "Spa", "Airport shuttle"],
            "url": None,
            "location": "Bangkok",
            "checkin": "2026-08-09",
            "checkout": "2026-08-10",
            "category": "non_beachfront",
            "flight_cost": 126,
        }
    ]


def test_hotel_parser_rejects_disqualified_brand_names():
    html = '<div class="uaTTDe"><h2 class="BgYkof">Westin Villa</h2><span>$220</span></div>'

    assert parse_hotel_cards(
        html,
        ParseContext("Bangkok", "2026-08-09", "2026-08-10", 4, "non_beachfront", 126),
    ) == []


def test_provider_parser_reads_escaped_dollar_prices():
    html = r"Agoda data \u0024220 Booking.com data \x24230"

    assert parse_provider_prices(html) == {"agoda": 220.0, "booking.com": 230.0}


def test_provider_parser_keeps_lowest_valid_price_for_each_provider():
    html = r"Agoda \x24220 unrelated text Agoda \u0024180 Agoda $7"

    assert parse_provider_prices(html) == {"agoda": 180.0}
