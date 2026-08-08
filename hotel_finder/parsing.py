"""Pure parsers for Google Hotels response markup."""

import math
import re
from dataclasses import dataclass
from typing import Any

from selectolax.lexbor import LexborHTMLParser

LUXURY_BRANDS = {
    5: [
        "jw marriott",
        "ritz-carlton",
        "ritz carlton",
        "st. regis",
        "st regis",
        "w hotel",
        "luxury collection",
        "edition hotel",
        "waldorf astoria",
        "conrad",
        "sofitel",
        "fairmont",
        "raffles",
        "park hyatt",
        "grand hyatt",
        "andaz",
        "alila",
        "intercontinental",
        "regent",
        "kimpton",
        "six senses",
        "mandarin oriental",
        "aman ",
        "banyan tree",
        "shangri-la",
        "shangri la",
        "four seasons",
        "capella",
        "rosewood",
        "taj hotel",
        "taj resort",
        "oberoi",
        "anantara",
        "dusit thani",
        "kempinski",
    ],
    4: [
        "le meridien",
        "le méridien",
        "westin",
        "autograph collection",
        "hilton ",
        "pullman ",
        "mgallery",
        "movenpick",
        "mövenpick",
        "novotel",
        "avani",
        "centara grand",
        "vinpearl",
        "cinnamon grand",
        "cinnamon life",
        "grand mercure",
        "sheraton ",
        "marriott ",
        "hyatt regency",
        "crowne plaza",
        "renaissance ",
        "doubletree",
        "wyndham grand",
        "wyndham ",
        "melia ",
        "mélia ",
        "four points",
        "oakwood premier",
        "courtyard by marriott",
        "sokha",
    ],
}
DISQUALIFIERS = (
    "villa",
    "hostel",
    "guesthouse",
    "guest house",
    "homestay",
    "apartment",
    "dormitory",
    "capsule",
    "backpacker",
)
KNOWN_AMENITIES = (
    "Free Wi-Fi",
    "Pool",
    "Spa",
    "Restaurant",
    "Fitness",
    "Bar",
    "Breakfast",
    "Beach",
    "Airport shuttle",
    "Kid-friendly",
    "Gym",
)
PROVIDER_NAMES = (
    ("Trip.com", "trip.com"),
    ("Agoda", "agoda"),
    ("Expedia", "expedia"),
    ("Booking.com", "booking.com"),
    ("Hotels.com", "hotels.com"),
    ("Traveloka", "traveloka"),
    ("Official Site", "official"),
)
GOOGLE_LODGING_CLICK_PATH = "/travel/lodging/clk"
SIMPLE_PROVIDER_MARKUP_MAX_CHARS = 4_096
INERT_PROVIDER_TEXT_TAGS = frozenset({"script", "style", "template", "noscript"})
PRICE_PATTERN = re.compile(
    r"\$((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+))(?![0-9,])"
)


@dataclass(frozen=True)
class ParseContext:
    location: str
    checkin: str
    checkout: str
    min_stars: int
    category: str
    flight_cost: float


def brand_star_class(name: str) -> int | None:
    """Return a known brand's star class, excluding non-hotel property types."""
    normalized_name = name.lower()
    if any(disqualifier in normalized_name for disqualifier in DISQUALIFIERS):
        return None
    for star_class in (5, 4):
        for brand in LUXURY_BRANDS[star_class]:
            if normalized_name.startswith(brand) or f" {brand}" in f" {normalized_name}":
                return star_class
    return None


def _parse_rating(card: Any) -> float | None:
    rating_node = card.css_first("span.KFi5wf.lA0BZ")
    if rating_node is None:
        return None
    try:
        rating = float(rating_node.text(strip=True))
    except ValueError:
        return None
    return rating if math.isfinite(rating) and 0 <= rating <= 5 else None


def _html_star_class(card: Any) -> int | None:
    for node in card.css("span.ne5qie.Ih19Ad"):
        match = re.match(r"([1-5])-star", node.text(strip=True))
        if match is not None:
            return int(match.group(1))
    return None


def _amenities(card: Any) -> list[str]:
    for selector in ("span.LtjZ2d", "span.QYEgn"):
        amenities = [
            text
            for node in card.css(selector)
            if len(text := node.text(strip=True)) > 2
        ]
        if amenities:
            return amenities
    card_text = card.text()
    return [amenity for amenity in KNOWN_AMENITIES if amenity.lower() in card_text.lower()]


def _normalized_url(card: Any) -> str | None:
    link = card.css_first("a[href]")
    href = link.attributes.get("href", "") if link is not None else ""
    if href.startswith("/travel/"):
        return f"https://www.google.com{href}"
    return href or None


def _parse_price(card: Any) -> float | None:
    price_match = PRICE_PATTERN.search(card.text())
    if price_match is None:
        return None
    try:
        return float(price_match.group(1).replace(",", ""))
    except ValueError:
        return None


def has_structural_hotel_card(html: str) -> bool:
    """Return whether Google markup contains a complete card before filters."""
    parser = LexborHTMLParser(html)
    for card in parser.css("div.uaTTDe"):
        name_node = card.css_first("h2.BgYkof") or card.css_first("h2.Cx32Ud")
        if name_node is None:
            continue
        name = name_node.text(strip=True)
        if not name or (_html_star_class(card) is None and brand_star_class(name) is None):
            continue
        if _parse_price(card) is not None:
            return True
    return False


def parse_hotel_cards(html: str, context: ParseContext) -> list[dict[str, Any]]:
    """Extract bounded, display-ready hotel records from Google Hotels HTML."""
    parser = LexborHTMLParser(html)
    hotels: list[dict[str, Any]] = []
    for card in parser.css("div.uaTTDe"):
        name_node = card.css_first("h2.BgYkof") or card.css_first("h2.Cx32Ud")
        if name_node is None:
            continue
        name = name_node.text(strip=True)
        html_star = _html_star_class(card)
        star_class = html_star or brand_star_class(name)
        if star_class is None or star_class < context.min_stars:
            continue
        price = _parse_price(card)
        if price is None:
            continue
        if price > 1500:
            continue
        hotels.append(
            {
                "name": name,
                "price": price,
                "rating": _parse_rating(card),
                "star_class": star_class,
                "confirmation": "html" if html_star is not None else "brand",
                "amenities": _amenities(card),
                "url": _normalized_url(card),
                "location": context.location,
                "checkin": context.checkin,
                "checkout": context.checkout,
                "category": context.category,
                "flight_cost": context.flight_cost,
            }
        )
    return sorted(hotels, key=lambda hotel: float(hotel["price"]))


def _visible_text_fragments(row: Any) -> list[str]:
    fragments: list[str] = []
    for node in row.traverse(include_text=True, skip_empty=True):
        if not node.is_text_node:
            continue
        ancestor = node.parent
        inert = False
        while ancestor is not None:
            attributes = ancestor.attributes
            normalized_style = re.sub(
                r"\s+", "", attributes.get("style", "").casefold()
            )
            if (
                ancestor.tag in INERT_PROVIDER_TEXT_TAGS
                or "hidden" in attributes
                or attributes.get("aria-hidden", "").casefold() == "true"
                or "display:none" in normalized_style
                or "visibility:hidden" in normalized_style
            ):
                inert = True
                break
            ancestor = ancestor.parent
        if inert:
            continue
        text = " ".join(node.text().split())
        if text:
            fragments.append(text)
    return fragments


def _provider_key_in_offer(fragments: list[str]) -> str | None:
    labels: set[str] = set()
    for text in fragments:
        for display_name, provider_key in PROVIDER_NAMES:
            if re.search(
                rf"(?<![\w.]){re.escape(display_name)}(?![\w.])",
                text,
                re.IGNORECASE,
            ):
                labels.add(provider_key)
    return next(iter(labels)) if len(labels) == 1 else None


def _valid_prices(text: str) -> list[float]:
    prices: list[float] = []
    for value in PRICE_PATTERN.findall(text):
        try:
            price = float(value.replace(",", ""))
        except ValueError:
            continue
        if 10 < price < 2000:
            prices.append(price)
    return prices


def _is_google_lodging_offer(row: Any) -> bool:
    href = row.attributes.get("href", "")
    return href == GOOGLE_LODGING_CLICK_PATH or href.startswith(
        f"{GOOGLE_LODGING_CLICK_PATH}?"
    )


def _parse_structured_provider_prices(rows: list[Any]) -> dict[str, float]:
    providers: dict[str, float] = {}
    for row in rows:
        fragments = _visible_text_fragments(row)
        provider_key = _provider_key_in_offer(fragments)
        if provider_key is None:
            continue
        prices = _valid_prices(" ".join(fragments))
        if prices:
            price = min(prices)
            providers[provider_key] = min(price, providers.get(provider_key, price))
    return providers


def _parse_simple_provider_markup(text: str) -> dict[str, float]:
    providers: dict[str, float] = {}
    labels: list[tuple[int, int, str]] = []
    for display_name, provider_key in PROVIDER_NAMES:
        for match in re.finditer(re.escape(display_name), text, re.IGNORECASE):
            labels.append((match.start(), match.end(), provider_key))
    labels.sort()
    for index, (_, label_end, provider_key) in enumerate(labels):
        next_label_start = labels[index + 1][0] if index + 1 < len(labels) else len(text)
        chunk = text[label_end : min(label_end + 300, next_label_start)]
        for value in re.findall(r"(?:\\x24|\\u0024|\$)(\d+)", chunk):
            price = float(value)
            if 10 < price < 2000:
                providers[provider_key] = min(price, providers.get(provider_key, price))
    return providers


def parse_provider_prices(html: str) -> dict[str, float]:
    """Extract row-associated provider prices from Google Hotels entity markup."""
    parser = LexborHTMLParser(html)
    offer_rows = [row for row in parser.css("a[href]") if _is_google_lodging_offer(row)]
    if offer_rows:
        return _parse_structured_provider_prices(offer_rows)
    if (
        len(html) > SIMPLE_PROVIDER_MARKUP_MAX_CHARS
        or parser.css_first("script, style, template, noscript") is not None
    ):
        return {}
    return _parse_simple_provider_markup(" ".join(_visible_text_fragments(parser.root)))
