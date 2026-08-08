from hotel_finder.parsing import (
    ParseContext,
    has_structural_hotel_card,
    parse_hotel_cards,
    parse_provider_prices,
)


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


def test_hotel_card_sanitizes_nonfinite_review_rating():
    html = """<div class="uaTTDe"><h2 class="BgYkof">Test Grand Hotel</h2>
    <span class="KFi5wf lA0BZ">NaN</span><span class="ne5qie Ih19Ad">5-star hotel</span>
    <span>$220</span></div>"""

    hotels = parse_hotel_cards(
        html,
        ParseContext("Bangkok", "2026-08-09", "2026-08-10", 5, "non_beachfront", 126),
    )

    assert hotels[0]["rating"] is None


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


def test_provider_parser_does_not_claim_price_after_next_provider_label():
    html = r"Agoda unavailable Booking.com \x24230"

    assert parse_provider_prices(html) == {"booking.com": 230.0}


def test_provider_parser_binds_visible_offer_price_over_later_script_decoy():
    html = r"""
    <html><body>
      <a href="/travel/lodging/clk?pc=real-offer">
        <div><span><span>Agoda</span></span></div>
        <div><span>$185</span><span>Visit site</span></div>
      </a>
      <script>window.bootstrap = {"provider":"Agoda","price":"\u002417"};</script>
    </body></html>
    """

    assert parse_provider_prices(html) == {"agoda": 185.0}


def test_provider_parser_never_borrows_price_from_another_offer_row():
    html = """
    <a href="/travel/lodging/clk?pc=unavailable"><span>Agoda</span></a>
    <a href="/travel/lodging/clk?pc=unlabelled"><span>$90</span></a>
    """

    assert parse_provider_prices(html) == {}


def test_provider_parser_fails_closed_on_script_only_provider_data():
    html = r"""
    <html><body><main>Hotel details</main>
    <script>window.bootstrap = {"provider":"Agoda","price":"\x2417"};</script>
    </body></html>
    """

    assert parse_provider_prices(html) == {}


def test_provider_parser_ignores_inert_text_inside_offer_row():
    html = r"""
    <a href="/travel/lodging/clk?pc=scripted-offer">
      <script>Agoda</script><span>$17</span>
    </a>
    """

    assert parse_provider_prices(html) == {}


def test_provider_parser_reads_direct_visible_text_in_offer_row():
    html = '<a href="/travel/lodging/clk?pc=direct-offer">Agoda $185</a>'

    assert parse_provider_prices(html) == {"agoda": 185.0}


def test_provider_parser_ignores_comments_and_attributes_in_simple_markup():
    html = '<html><!-- Agoda $17 --><body data-offer="Agoda $18">Hotel</body></html>'

    assert parse_provider_prices(html) == {}


def test_provider_parser_ignores_hidden_labels_and_prices_in_offer_rows():
    html = """
    <a href="/travel/lodging/clk?pc=hidden-label">
      <span hidden>Agoda</span><span>$17</span>
    </a>
    <a href="/travel/lodging/clk?pc=visible-offer">
      <span>Agoda</span><span aria-hidden="true">$18</span><span>$185</span>
    </a>
    """

    assert parse_provider_prices(html) == {"agoda": 185.0}


def test_hotel_parser_skips_malformed_price_without_aborting_later_cards():
    html = """
    <div class="uaTTDe"><h2 class="BgYkof">Broken Price Hotel</h2>
    <span class="ne5qie Ih19Ad">5-star hotel</span><span>$,,,</span></div>
    <div class="uaTTDe"><h2 class="BgYkof">Valid Price Hotel</h2>
    <span class="ne5qie Ih19Ad">5-star hotel</span><span>$240</span></div>
    """

    hotels = parse_hotel_cards(
        html,
        ParseContext("Bangkok", "2026-08-09", "2026-08-10", 5, "non_beachfront", 126),
    )

    assert [hotel["name"] for hotel in hotels] == ["Valid Price Hotel"]
    assert hotels[0]["price"] == 240.0


def test_structural_hotel_card_recognition_is_independent_of_business_filters():
    filtered_cards = """
    <div class="uaTTDe"><h2 class="BgYkof">Three Star Hotel</h2>
    <span class="ne5qie Ih19Ad">3-star hotel</span><span>$120</span></div>
    <div class="uaTTDe"><h2 class="BgYkof">Over Budget Hotel</h2>
    <span class="ne5qie Ih19Ad">5-star hotel</span><span>$1,501</span></div>
    """

    assert has_structural_hotel_card(filtered_cards) is True
    assert has_structural_hotel_card(
        '<div class="uaTTDe">Scheduled maintenance</div>'
    ) is False
