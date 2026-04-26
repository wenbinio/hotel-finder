"""
Flask API backend for hotel/flight price search.
Wraps fast-hotels and fast-flights to query Google Hotels/Flights programmatically.
Serves React frontend from static/ when available (hostable mode).
"""
import os
import re
import json
import time
import threading
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Global progress state ─────────────────────────────────────────────────────
sweep_progress = {
    "active": False,
    "phase": "",        # "dates" or "destinations"
    "current_date": 0,
    "total_dates": 0,
    "current_dest": "",
    "dests_done": 0,
    "total_dests": 0,
    "dates_done": [],   # completed date results so far
}
progress_lock = threading.Lock()
from selectolax.lexbor import LexborHTMLParser

from fast_hotels.primp import Client
from fast_hotels.hotels_impl import HotelData, Guests, THSData

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app = Flask(__name__, static_folder=static_dir, static_url_path="")
else:
    app = Flask(__name__)
CORS(app)


# ── TTL cache ─────────────────────────────────────────────────────────────────
# Hotel results don't change second-to-second, but the same (location, dates,
# currency) tuple gets queried dozens of times within a single sweep and
# repeatedly across users. A short-lived in-memory cache cuts request volume,
# trims wall time, and reduces our chance of getting rate-limited by Google.
SEARCH_CACHE_TTL = int(os.environ.get("SEARCH_CACHE_TTL", "600"))   # 10 minutes
PROVIDER_CACHE_TTL = int(os.environ.get("PROVIDER_CACHE_TTL", "600"))
XOTELO_CACHE_TTL = int(os.environ.get("XOTELO_CACHE_TTL", "1800"))  # 30 minutes


class _TTLCache:
    """Tiny thread-safe TTL cache. Not LRU — entries simply expire."""
    def __init__(self, ttl: int, max_size: int = 1024):
        self.ttl = ttl
        self.max_size = max_size
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._data.pop(key, None)
                return None
            return value

    def set(self, key, value):
        with self._lock:
            if len(self._data) >= self.max_size:
                # Evict the oldest expiring entry. Cheap O(n) sweep — fine for
                # max_size=1024.
                oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(oldest, None)
            self._data[key] = (time.time() + self.ttl, value)


_search_cache = _TTLCache(SEARCH_CACHE_TTL)
_provider_cache = _TTLCache(PROVIDER_CACHE_TTL)
_xotelo_cache = _TTLCache(XOTELO_CACHE_TTL)


# Pure helpers (currency parsing, brand classification) live in parsing.py
# so they can be unit-tested without Flask/scraping/network deps.
from parsing import (
    CURRENCY_SYMBOLS,
    DEFAULT_MAX_PRICE_USD,
    LUXURY_BRANDS,
    DISQUALIFIERS,
    brand_star_class,
    currency_price_re as _currency_price_re,
    default_max_price,
    extract_price as _extract_price,
    extract_star_class,
    parse_localized_number as _parse_localized_number,
)


# ── Destinations ──────────────────────────────────────────────────────────────
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

FLIGHT_BUDGET_MAP = {}
for cat in DESTINATIONS.values():
    for d in cat:
        FLIGHT_BUDGET_MAP[d["name"]] = d["flight_usd"]


# ── Hotel search ──────────────────────────────────────────────────────────────
def _category_for(location: str, custom_categories: "dict | None" = None) -> str:
    """Return the destination category for a location.

    ``custom_categories`` is a {location_name: category_name} map supplied by
    the caller, used when the request includes user-defined destinations.
    Falls back to the built-in DESTINATIONS table, then to ``"other"``.
    """
    if custom_categories and location in custom_categories:
        return custom_categories[location]
    for cat, dlist in DESTINATIONS.items():
        if any(d["name"] == location for d in dlist):
            return cat
    return "other"


def search_hotels(
    location: str,
    checkin: str,
    checkout: str,
    min_stars: int = 5,
    language: str = "en",
    currency: str = "USD",
    region: "str | None" = None,
    max_price: "float | None" = None,
    category_map: "dict | None" = None,
    flight_cost: "float | None" = None,
) -> list[dict]:
    """Search Google Hotels for hotels at or above the given star class.

    ``language``/``currency``/``region`` are forwarded to Google as ``hl``,
    ``curr`` and ``gl`` so the same backend can serve any market.
    ``max_price`` is in the requested currency; defaults vary by currency.

    Results are cached for SEARCH_CACHE_TTL seconds (default 10 min) keyed on
    everything that affects what Google returns. ``category_map`` and
    ``flight_cost`` only affect post-processing of the cached card list, so
    they're applied after the cache lookup.
    """
    cur = (currency or "USD").upper()
    cache_key = (location, checkin, checkout, min_stars, language or "en", cur, region or "")
    cached = _search_cache.get(cache_key)
    if cached is not None:
        # Re-stamp category and flight_cost in case the caller's mapping has
        # changed since the cache was populated.
        return [
            {**h,
             "category": _category_for(h["location"], category_map),
             "flight_cost": (flight_cost if flight_cost is not None
                             else FLIGHT_BUDGET_MAP.get(h["location"], 0))}
            for h in cached
        ]

    hotel_data = [HotelData(checkin_date=checkin, checkout_date=checkout, location=location)]
    guests = Guests(adults=1)
    ths = THSData.from_interface(hotel_data=hotel_data, guests=guests, room_type="standard")
    data = ths.as_b64()
    params = {
        "ths": data.decode("utf-8"),
        "hl": language or "en",
        "curr": (currency or "USD").upper(),
        "q": f"{min_stars} star hotels {location}",
    }
    if region:
        params["gl"] = region
    client = Client(impersonate="chrome_126", verify=False)
    city = location.strip().replace(" ", "+").lower()
    url = f"https://www.google.com/travel/hotels/{city}"
    res = client.get(url, params=params)
    if res.status_code != 200:
        return []

    # In USD, prices over 1500 are likely junk; in JPY/IDR/VND that ceiling
    # is ridiculously low. Scale automatically for high-denomination currencies.
    if max_price is None:
        max_price = default_max_price(currency)

    parser = LexborHTMLParser(res.text)
    hotels = []
    for card in parser.css("div.uaTTDe"):
        name_elem = card.css_first("h2.BgYkof") or card.css_first("h2.Cx32Ud")
        if not name_elem:
            continue
        name = name_elem.text(strip=True)

        review_rating = None
        re_elem = card.css_first("span.KFi5wf.lA0BZ")
        if re_elem:
            num = _parse_localized_number(re_elem.text(strip=True))
            if num is not None:
                review_rating = num

        # Star class label is localized ("5-star hotel" / "Hôtel 5 étoiles" /
        # "5성급 호텔" / ...) so extract_star_class just grabs the first digit.
        html_star_class = None
        for span in card.css("span.ne5qie.Ih19Ad"):
            html_star_class = extract_star_class(span.text(strip=True))
            if html_star_class:
                break

        # Also check brand recognition
        brand_stars = brand_star_class(name)

        # Use the highest-confidence star class
        star_class = html_star_class or brand_stars

        # Filter: must meet minimum star threshold
        if star_class is None or star_class < min_stars:
            continue

        amenities = []
        for sel in ["span.LtjZ2d", "span.QYEgn"]:
            elems = card.css(sel)
            if elems:
                amenities = [e.text(strip=True) for e in elems if e.text(strip=True) and len(e.text(strip=True)) > 2]
                break
        # Note: amenity fallback by keyword is intentionally omitted — the
        # keywords would be locale-specific and Google already returns
        # localized amenity chips in the selectors above.

        price = _extract_price(card.text(), currency)
        if price is None or price > max_price:
            continue

        link_url = None
        link = card.css_first("a[href]")
        if link:
            href = link.attributes.get("href", "")
            link_url = ("https://www.google.com" + href) if href.startswith("/travel/") else href

        hotels.append({
            "name": name,
            "price": price,
            "currency": (currency or "USD").upper(),
            "rating": review_rating,
            "star_class": star_class,
            "confirmation": "html" if html_star_class else "brand",
            "amenities": amenities,
            "url": link_url,
            "location": location,
            "checkin": checkin,
            "checkout": checkout,
            "category": _category_for(location, category_map),
            "flight_cost": flight_cost if flight_cost is not None else FLIGHT_BUDGET_MAP.get(location, 0),
        })

    hotels.sort(key=lambda h: h["price"])
    _search_cache.set(cache_key, hotels)
    return hotels


def fetch_provider_prices(entity_url, language: str = "en", currency: str = "USD"):
    """Fetch a Google Hotels entity page and extract prices from all booking providers."""
    cur = (currency or "USD").upper()
    cache_key = (entity_url, language or "en", cur)
    cached = _provider_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        client = Client(impersonate="chrome_126", verify=False)
        res = client.get(entity_url, params={
            "hl": language or "en",
            "curr": cur,
        })
        if res.status_code != 200:
            return {}

        html = res.text
        providers_found = {}
        provider_names = [
            ("Trip.com", "trip.com"),
            ("Agoda", "agoda"),
            ("Expedia", "expedia"),
            ("Booking.com", "booking.com"),
            ("Hotels.com", "hotels.com"),
            ("Traveloka", "traveloka"),
            ("Official Site", "official"),
        ]
        for display_name, key in provider_names:
            pattern = re.escape(display_name)
            for m in re.finditer(pattern, html, re.IGNORECASE):
                start = max(0, m.start() - 300)
                end = min(len(html), m.end() + 300)
                chunk = html[start:end]
                price = _extract_price(chunk, currency)
                if price is not None:
                    if 10 < price < 200000:
                        if key not in providers_found or price < providers_found[key]:
                            providers_found[key] = price
                    break
        _provider_cache.set(cache_key, providers_found)
        return providers_found
    except Exception:
        return {}


# ── API Routes ────────────────────────────────────────────────────────────────

def _locale_params(data: dict) -> dict:
    """Extract globalization params (language/currency/region) from a request body.

    Defaults to en/USD/none for backwards compatibility.
    """
    return {
        "language": (data.get("language") or "en"),
        "currency": (data.get("currency") or "USD").upper(),
        "region": data.get("region") or None,
    }


def _resolve_destinations(data: dict) -> "tuple[dict, dict, dict]":
    """Resolve which destinations to search.

    Returns (destinations_by_category, flight_budget_map, category_map).
    If the caller passes ``customDestinations`` (a list of {name, airport,
    flight_usd, category}), those replace the built-in SE Asia table — so the
    backend can be used anywhere in the world.
    """
    custom = data.get("customDestinations")
    if not custom:
        return DESTINATIONS, FLIGHT_BUDGET_MAP, {}
    by_cat: dict[str, list[dict]] = {}
    flight_map: dict[str, float] = {}
    cat_map: dict[str, str] = {}
    for d in custom:
        name = d.get("name")
        if not name:
            continue
        cat = d.get("category", "other")
        flight_usd = d.get("flight_usd", 0)
        by_cat.setdefault(cat, []).append({
            "name": name,
            "airport": d.get("airport", ""),
            "flight_usd": flight_usd,
        })
        flight_map[name] = flight_usd
        cat_map[name] = cat
    return by_cat, flight_map, cat_map


@app.route("/api/destinations", methods=["GET"])
def get_destinations():
    return jsonify(DESTINATIONS)


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search a single destination for a single date."""
    data = request.json
    location = data.get("location", "Bangkok")
    checkin = data.get("checkin", "2026-06-15")
    checkout = data.get("checkout", "2026-06-16")
    min_stars = data.get("minStars", 5)
    locale = _locale_params(data)
    max_price = data.get("maxPrice")
    try:
        hotels = search_hotels(
            location, checkin, checkout, min_stars=min_stars,
            max_price=max_price, **locale,
        )
        return jsonify({
            "location": location,
            "checkin": checkin,
            "hotels": hotels,
            "currency": locale["currency"],
            "language": locale["language"],
        })
    except Exception as e:
        return jsonify({"error": str(e), "location": location}), 500


@app.route("/api/search-all", methods=["POST"])
def api_search_all():
    """Search all destinations for given dates. Returns results as they come in."""
    data = request.json
    checkin = data.get("checkin", "2026-06-15")
    checkout = data.get("checkout", "2026-06-16")
    max_flight = data.get("maxFlight", 300)
    min_stars = data.get("minStars", 5)
    destinations = data.get("destinations", None)
    locale = _locale_params(data)
    max_price = data.get("maxPrice")
    dests_by_cat, flight_map, cat_map = _resolve_destinations(data)

    # Collect all destinations within flight budget
    all_dests = []
    dest_to_cat: dict[str, str] = {}
    for cat, dlist in dests_by_cat.items():
        for d in dlist:
            if d.get("flight_usd", 0) <= max_flight:
                if destinations is None or d["name"] in destinations:
                    all_dests.append(d)
                    dest_to_cat[d["name"]] = cat

    results: dict[str, list] = {cat: [] for cat in dests_by_cat.keys()}

    def search_dest(d):
        try:
            hotels = search_hotels(
                d["name"], checkin, checkout, min_stars=min_stars,
                max_price=max_price, category_map=cat_map or None,
                flight_cost=d.get("flight_usd", 0), **locale,
            )
            time.sleep(0.3)
            return d, hotels
        except Exception:
            return d, []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(search_dest, d): d for d in all_dests}
        for future in as_completed(futures):
            d, hotels = future.result()
            cat = dest_to_cat.get(d["name"], "other")
            results.setdefault(cat, []).extend(hotels)

    for cat_list in results.values():
        cat_list.sort(key=lambda h: h["price"])

    totals = {f"total_{cat}": len(items) for cat, items in results.items()}
    # Back-compat shortcuts for the existing default categories
    return jsonify({
        "checkin": checkin,
        "checkout": checkout,
        "maxFlight": max_flight,
        "currency": locale["currency"],
        "language": locale["language"],
        "results": results,
        "totals": totals,
        "totalBeachfront": len(results.get("beachfront", [])),
        "totalNonBeachfront": len(results.get("non_beachfront", [])),
    })


@app.route("/api/sweep-progress", methods=["GET"])
def api_sweep_progress():
    """Poll this to get current sweep progress."""
    with progress_lock:
        return jsonify(dict(sweep_progress))


@app.route("/api/cheapest-dates", methods=["POST"])
def api_cheapest_dates():
    """Find cheapest dates within a range. Supports single location, list, category, or all."""
    global sweep_progress
    data = request.json
    locations = data.get("locations", None)
    mode = data.get("mode", "single")
    start_date = data.get("startDate", "2026-05-01")
    end_date = data.get("endDate", "2026-10-31")
    nights = data.get("nights", 1)
    sample_count = data.get("sampleCount", 8)
    min_stars = data.get("minStars", 5)
    locale = _locale_params(data)
    max_price = data.get("maxPrice")
    dests_by_cat, flight_map, cat_map = _resolve_destinations(data)

    if locations and len(locations) >= 1:
        dest_names = locations
    else:
        dest_names = [d["name"] for cat in dests_by_cat.values() for d in cat]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    if total_days < 1:
        return jsonify({"error": "Invalid date range"}), 400

    step = max(1, total_days // sample_count)
    dates_to_check = []
    current = start
    while current < end:
        ci = current.strftime("%Y-%m-%d")
        co = (current + timedelta(days=nights)).strftime("%Y-%m-%d")
        dates_to_check.append((ci, co))
        current += timedelta(days=step)

    # Init progress
    with progress_lock:
        sweep_progress = {
            "active": True,
            "phase": "searching",
            "current_date": 0,
            "total_dates": len(dates_to_check),
            "current_dest": "",
            "dests_done": 0,
            "total_dests": len(dest_names),
            "dates_done": [],
        }

    date_results = []

    def check_date(date_idx, ci, co):
        # Parallelize destinations within a single date sample. With ~4 workers
        # and 20+ destinations this is ~4x faster than the prior sequential
        # loop. Progress reporting is updated as each destination completes.
        all_hotels = []
        completed = 0
        completed_lock = threading.Lock()

        def search_one(loc):
            try:
                return search_hotels(
                    loc, ci, co, min_stars=min_stars,
                    max_price=max_price, category_map=cat_map or None,
                    flight_cost=flight_map.get(loc),
                    **locale,
                )
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_loc = {executor.submit(search_one, loc): loc for loc in dest_names}
            for future in as_completed(future_to_loc):
                loc = future_to_loc[future]
                hotels = future.result()
                all_hotels.extend(hotels)
                with completed_lock:
                    completed += 1
                with progress_lock:
                    sweep_progress["current_date"] = date_idx + 1
                    sweep_progress["current_dest"] = loc
                    sweep_progress["dests_done"] = completed

        all_hotels.sort(key=lambda h: h["price"])
        cheapest = all_hotels[0] if all_hotels else None
        result = {
            "checkin": ci,
            "checkout": co,
            "cheapest_price": cheapest["price"] if cheapest else None,
            "hotel_count": len(all_hotels),
            "cheapest_hotel": cheapest["name"] if cheapest else None,
            "location": cheapest["location"] if cheapest else None,
            "all_hotels": all_hotels,
        }

        with progress_lock:
            sweep_progress["dates_done"].append({
                "checkin": ci, "cheapest_price": result["cheapest_price"],
                "cheapest_hotel": result.get("cheapest_hotel"),
                "location": result.get("location"),
            })

        return result

    # Run date samples — process sequentially so progress is meaningful,
    # but parallelize destinations WITHIN each date sample
    for idx, (ci, co) in enumerate(dates_to_check):
        result = check_date(idx, ci, co)
        date_results.append(result)

    date_results.sort(key=lambda r: r["checkin"])
    valid = [r for r in date_results if r["cheapest_price"] is not None]
    cheapest_entry = min(valid, key=lambda r: r["cheapest_price"], default=None)

    best_date_hotels: dict[str, list] = {cat: [] for cat in dests_by_cat.keys()}
    if cheapest_entry:
        for h in cheapest_entry["all_hotels"]:
            cat = h.get("category", "other")
            best_date_hotels.setdefault(cat, []).append(h)

    for r in date_results:
        r.pop("all_hotels", None)

    # Clear progress
    with progress_lock:
        sweep_progress["active"] = False
        sweep_progress["phase"] = "done"

    return jsonify({
        "locations": dest_names,
        "location": dest_names[0] if len(dest_names) == 1 else None,
        "dateRange": {"start": start_date, "end": end_date},
        "nights": nights,
        "currency": locale["currency"],
        "language": locale["language"],
        "dates": date_results,
        "cheapestDate": {
            "checkin": cheapest_entry["checkin"],
            "checkout": cheapest_entry["checkout"],
            "cheapest_price": cheapest_entry["cheapest_price"],
            "cheapest_hotel": cheapest_entry["cheapest_hotel"],
            "location": cheapest_entry["location"],
        } if cheapest_entry else None,
        "bestDateResults": best_date_hotels,
        "totals": {f"total_{cat}": len(items) for cat, items in best_date_hotels.items()},
        "totalBeachfront": len(best_date_hotels.get("beachfront", [])),
        "totalNonBeachfront": len(best_date_hotels.get("non_beachfront", [])),
    })


# Load pre-resolved TripAdvisor hotel keys
_TA_KEYS_PATH = os.path.join(os.path.dirname(__file__), "ta_keys.json")
_TA_KEYS = {}
if os.path.exists(_TA_KEYS_PATH):
    with open(_TA_KEYS_PATH, "r", encoding="utf-8") as _f:
        _TA_KEYS = json.load(_f)


def resolve_tripadvisor_key(hotel_name, location=""):
    """Look up TripAdvisor hotel key from pre-resolved mapping. Fuzzy match on name."""
    if hotel_name in _TA_KEYS:
        return _TA_KEYS[hotel_name]
    # Fuzzy: check if any key is a substring of the hotel name or vice versa
    name_lower = hotel_name.lower()
    for key, val in _TA_KEYS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return val
    return None


def _ota_search_url(provider_code, hotel_name, checkin, checkout,
                    language: str = "en", region: "str | None" = None,
                    currency: str = "USD"):
    """Construct a search URL for the given OTA pre-filled with hotel + dates.

    ``language``/``region`` produce locale tags such as ``en-sg`` or ``de-de``
    used by Traveloka and other OTAs that require them. ``currency`` is passed
    where supported.
    """
    from urllib.parse import quote_plus
    q = quote_plus(hotel_name)
    lang = (language or "en").lower()
    reg = (region or "us").lower()
    locale = f"{lang}-{reg}"
    cur = (currency or "USD").upper()
    urls = {
        "BookingCom":  f"https://www.booking.com/searchresults.html?ss={q}&checkin={checkin}&checkout={checkout}&selected_currency={cur}&lang={lang}",
        "Agoda":       f"https://www.agoda.com/search?q={q}&checkIn={checkin}&los=1&currency={cur}&locale={locale}",
        "CtripTA":     f"https://www.trip.com/hotels/list?keyword={q}&checkIn={checkin}&checkOut={checkout}&curr={cur}&locale={locale}",
        "Expedia":     f"https://www.expedia.com/Hotel-Search?destination={q}&startDate={checkin}&endDate={checkout}&currency={cur}&langid={lang}",
        "HotelsCom":   f"https://www.hotels.com/search.do?q-destination={q}&q-check-in={checkin}&q-check-out={checkout}&currency={cur}&locale={locale}",
        "Traveloka":   f"https://www.traveloka.com/{locale}/hotel/search?spec={checkin}.{checkout}.1.0.HOTEL_GEO.{q}",
        "VioTA":       f"https://www.vio.com/Hotels/Search?q={q}&checkin={checkin}&checkout={checkout}&currency={cur}",
        "Vio":         f"https://www.vio.com/Hotels/Search?q={q}&checkin={checkin}&checkout={checkout}&currency={cur}",
    }
    return urls.get(provider_code, "")


def fetch_xotelo_prices(hotel_key, hotel_name, checkin, checkout, currency="USD",
                        language: str = "en", region: "str | None" = None):
    """Fetch per-OTA prices + booking links from the free Xotelo API."""
    cur = (currency or "USD").upper()
    # Cache key is hotel-name-agnostic on purpose — Xotelo's response is keyed
    # off ``hotel_key`` only, so a second call for the same hotel under a
    # different display name can reuse it. OTA URLs (which DO depend on
    # hotel_name) are recomputed below from the stored provider code.
    cache_key = (hotel_key, checkin, checkout, cur, language or "en", region or "")
    cached = _xotelo_cache.get(cache_key)
    if cached is not None:
        return {
            name: {
                "rate": info["rate"], "tax": info.get("tax", 0),
                "url": _ota_search_url(
                    info.get("_code", ""), hotel_name, checkin, checkout,
                    language=language, region=region, currency=currency),
            }
            for name, info in cached.items()
        }
    try:
        import requests as req
        res = req.get("https://data.xotelo.com/api/rates", params={
            "hotel_key": hotel_key,
            "chk_in": checkin,
            "chk_out": checkout,
            "currency": cur,
        }, timeout=12)
        if res.status_code != 200:
            return {}
        data = res.json()
        rates = data.get("result", {}).get("rates", [])
        result = {}
        for r in rates:
            if r.get("rate"):
                code = r.get("code", "")
                url = _ota_search_url(
                    code, hotel_name, checkin, checkout,
                    language=language, region=region, currency=currency,
                )
                result[r["name"]] = {
                    "rate": r["rate"],
                    "tax": r.get("tax", 0),
                    "url": url,
                    "_code": code,  # kept for cache re-stamping
                }
        # Cache the result (incl. provider code) so a second call can recompute
        # hotel-specific URLs without re-hitting Xotelo.
        _xotelo_cache.set(cache_key, result)
        # Strip the internal _code field from what we return to callers.
        return {name: {k: v for k, v in info.items() if k != "_code"}
                for name, info in result.items()}
    except Exception:
        return {}


@app.route("/api/compare-prices", methods=["POST"])
def api_compare_prices():
    """Fetch multi-provider prices from Google entity pages + Xotelo API."""
    data = request.json
    hotels = data.get("hotels", [])
    checkin = data.get("checkin", "2026-06-15")
    checkout = data.get("checkout", "2026-06-16")
    locale = _locale_params(data)
    enriched = []

    # Map both Xotelo's display names ("Booking.com") and Google's display
    # names ("booking.com") to the same canonical provider key, so prices
    # from the two sources merge cleanly.
    norm_map = {
        "booking": "booking.com", "agoda": "agoda",
        "trip": "trip.com", "expedia": "expedia",
        "hotels": "hotels.com", "traveloka": "traveloka",
        "vio": "vio.com",
    }

    def _canonicalize(name: str) -> str:
        key = name.lower().replace(".com", "").replace(" ", "").strip()
        for prefix, norm_name in norm_map.items():
            if prefix in key:
                return norm_name
        return name

    def enrich(h):
        # Xotelo is the more reliable source (structured per-OTA rates) and
        # is also faster — try it first. Only fall back to scraping the Google
        # entity page when Xotelo has no key for this hotel or returned empty.
        merged: dict[str, dict] = {}
        source = None
        ta_key = resolve_tripadvisor_key(h.get("name", ""), h.get("location", ""))

        if ta_key:
            ci = h.get("checkin", checkin)
            co = h.get("checkout", checkout)
            xotelo = fetch_xotelo_prices(
                ta_key, h.get("name", ""), ci, co,
                currency=locale["currency"],
                language=locale["language"],
                region=locale["region"],
            )
            for name, info in xotelo.items():
                rate = info["rate"] if isinstance(info, dict) else info
                url = info.get("url", "") if isinstance(info, dict) else ""
                canon = _canonicalize(name)
                existing = merged.get(canon)
                if existing is None or rate < existing["rate"]:
                    merged[canon] = {"rate": rate, "url": url, "source": "xotelo"}
                elif not existing.get("url"):
                    existing["url"] = url
            if merged:
                source = "xotelo"

        # Fallback: scrape Google's entity page only if Xotelo gave us nothing
        # useful. The window-regex on Google can mistake a "from $X" banner for
        # the actual rate, so it's intentionally relegated to a fallback.
        if not merged and h.get("url"):
            providers = fetch_provider_prices(
                h["url"],
                language=locale["language"],
                currency=locale["currency"],
            )
            for name, price in providers.items():
                canon = _canonicalize(name)
                merged[canon] = {"rate": price, "url": "", "source": "google"}
            if merged:
                source = "google"

        return {**h, "providers": merged, "xotelo_key": ta_key, "providers_source": source}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(enrich, h) for h in hotels[:15]]
        for f in as_completed(futures):
            enriched.append(f.result())

    enriched.sort(key=lambda h: h.get("price", 9999))
    return jsonify({
        "hotels": enriched,
        "currency": locale["currency"],
        "language": locale["language"],
    })


@app.route("/")
def serve_index():
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return "API running. Frontend at http://localhost:5173", 200


@app.route("/<path:path>")
def serve_static(path):
    if app.static_folder:
        full = os.path.join(app.static_folder, path)
        if os.path.exists(full):
            return send_from_directory(app.static_folder, path)
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return "Not found", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
