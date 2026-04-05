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

# ── Known luxury brands (must START with or BE the brand, not just contain it) ──
# Format: (brand_prefix, min_star_class)
# These are checked: hotel name must START WITH the brand or brand must be a
# standalone word boundary match (not a substring of "villa" or "guesthouse")
LUXURY_BRANDS = {
    5: [
        "jw marriott", "ritz-carlton", "ritz carlton", "st. regis", "st regis",
        "w hotel", "luxury collection", "edition hotel",
        "waldorf astoria", "conrad",
        "sofitel", "fairmont", "raffles",
        "park hyatt", "grand hyatt", "andaz", "alila",
        "intercontinental", "regent", "kimpton", "six senses",
        "mandarin oriental", "aman ", "banyan tree", "shangri-la",
        "shangri la", "four seasons", "capella", "rosewood",
        "taj hotel", "taj resort", "oberoi",
        "anantara", "dusit thani", "kempinski",
    ],
    4: [
        "le meridien", "le méridien", "westin", "autograph collection",
        "hilton ", "pullman ", "mgallery", "movenpick", "mövenpick",
        "novotel", "avani", "centara grand", "vinpearl",
        "cinnamon grand", "cinnamon life",
        "grand mercure", "sheraton ", "marriott ", "hyatt regency",
        "crowne plaza", "renaissance ", "doubletree", "wyndham grand",
        "wyndham ", "melia ", "mélia ", "four points", "oakwood premier",
        "courtyard by marriott", "sokha",
    ],
}

# Words that disqualify a brand match (it's a villa/hostel/guesthouse, not the chain)
DISQUALIFIERS = ["villa", "hostel", "guesthouse", "guest house", "homestay",
                 "apartment", "dormitory", "capsule", "backpacker"]


def brand_star_class(name: str) -> int | None:
    """Return the star class if the hotel name matches a known luxury brand, else None."""
    nl = name.lower()
    # Disqualify non-hotel properties
    if any(dq in nl for dq in DISQUALIFIERS):
        return None
    for star_class in [5, 4]:
        for brand in LUXURY_BRANDS[star_class]:
            # Brand must appear at the START of the name, or after a common prefix
            # like "The ", or be a clean word-boundary match
            if nl.startswith(brand) or f" {brand}" in f" {nl}":
                return star_class
    return None


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
def search_hotels(location: str, checkin: str, checkout: str, min_stars: int = 5) -> list[dict]:
    """Search Google Hotels for hotels at or above the given star class."""
    hotel_data = [HotelData(checkin_date=checkin, checkout_date=checkout, location=location)]
    guests = Guests(adults=1)
    ths = THSData.from_interface(hotel_data=hotel_data, guests=guests, room_type="standard")
    data = ths.as_b64()
    params = {
        "ths": data.decode("utf-8"),
        "hl": "en",
        "curr": "USD",
        "q": f"{min_stars} star hotels {location}",
    }
    client = Client(impersonate="chrome_126", verify=False)
    city = location.strip().replace(" ", "+").lower()
    url = f"https://www.google.com/travel/hotels/{city}"
    res = client.get(url, params=params)
    if res.status_code != 200:
        return []

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
            try:
                review_rating = float(re_elem.text(strip=True))
            except ValueError:
                pass

        # Extract star class from Google's HTML tag
        html_star_class = None
        for span in card.css("span.ne5qie.Ih19Ad"):
            m = re.match(r"(\d)-star", span.text(strip=True))
            if m:
                html_star_class = int(m.group(1))
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
        if not amenities:
            ct = card.text()
            known_a = ["Free Wi-Fi", "Pool", "Spa", "Restaurant", "Fitness",
                        "Bar", "Breakfast", "Beach", "Airport shuttle", "Kid-friendly", "Gym"]
            amenities = [a for a in known_a if a.lower() in ct.lower()]

        price = None
        ct = card.text()
        pm = re.findall(r"\$([0-9,]+)", ct)
        if pm:
            try:
                price = float(pm[0].replace(",", ""))
            except ValueError:
                pass
        if price is None or price > 1500:
            continue

        link_url = None
        link = card.css_first("a[href]")
        if link:
            href = link.attributes.get("href", "")
            link_url = ("https://www.google.com" + href) if href.startswith("/travel/") else href

        hotels.append({
            "name": name,
            "price": price,
            "rating": review_rating,
            "star_class": star_class,
            "confirmation": "html" if html_star_class else "brand",
            "amenities": amenities,
            "url": link_url,
            "location": location,
            "checkin": checkin,
            "checkout": checkout,
            "category": "beachfront" if location in [d["name"] for d in DESTINATIONS["beachfront"]] else "non_beachfront",
            "flight_cost": FLIGHT_BUDGET_MAP.get(location, 0),
        })

    hotels.sort(key=lambda h: h["price"])
    return hotels


def fetch_provider_prices(entity_url):
    """Fetch a Google Hotels entity page and extract prices from all booking providers."""
    try:
        client = Client(impersonate="chrome_126", verify=False)
        res = client.get(entity_url, params={"hl": "en", "curr": "USD"})
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
                # Google encodes $ as \x24 or \u0024 in its JS data
                prices = re.findall(r"(?:\\x24|\\u0024|\$)(\d+)", chunk)
                if prices:
                    price = float(prices[0])
                    if 10 < price < 2000:
                        if key not in providers_found or price < providers_found[key]:
                            providers_found[key] = price
                    break
        return providers_found
    except Exception:
        return {}


# ── API Routes ────────────────────────────────────────────────────────────────

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
    try:
        hotels = search_hotels(location, checkin, checkout, min_stars=min_stars)
        return jsonify({"location": location, "checkin": checkin, "hotels": hotels})
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

    # Collect all destinations within flight budget
    all_dests = []
    for cat, dlist in DESTINATIONS.items():
        for d in dlist:
            if d["flight_usd"] <= max_flight:
                if destinations is None or d["name"] in destinations:
                    all_dests.append(d)

    results = {"beachfront": [], "non_beachfront": []}

    def search_dest(d):
        try:
            hotels = search_hotels(d["name"], checkin, checkout, min_stars=min_stars)
            time.sleep(0.3)
            return d, hotels
        except Exception as e:
            return d, []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(search_dest, d): d for d in all_dests}
        for future in as_completed(futures):
            d, hotels = future.result()
            cat = "beachfront" if d in DESTINATIONS["beachfront"] else "non_beachfront"
            results[cat].extend(hotels)

    results["beachfront"].sort(key=lambda h: h["price"])
    results["non_beachfront"].sort(key=lambda h: h["price"])

    return jsonify({
        "checkin": checkin,
        "checkout": checkout,
        "maxFlight": max_flight,
        "results": results,
        "totalBeachfront": len(results["beachfront"]),
        "totalNonBeachfront": len(results["non_beachfront"]),
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

    if locations and len(locations) >= 1:
        dest_names = locations
    else:
        dest_names = [d["name"] for cat in DESTINATIONS.values() for d in cat]

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
        all_hotels = []
        for i, loc in enumerate(dest_names):
            with progress_lock:
                sweep_progress["current_date"] = date_idx + 1
                sweep_progress["current_dest"] = loc
                sweep_progress["dests_done"] = i
            try:
                hotels = search_hotels(loc, ci, co, min_stars=min_stars)
                all_hotels.extend(hotels)
            except Exception:
                pass
            time.sleep(0.15)

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

    best_date_hotels = {"beachfront": [], "non_beachfront": []}
    if cheapest_entry:
        for h in cheapest_entry["all_hotels"]:
            cat = h.get("category", "non_beachfront")
            best_date_hotels[cat].append(h)

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
        "dates": date_results,
        "cheapestDate": {
            "checkin": cheapest_entry["checkin"],
            "checkout": cheapest_entry["checkout"],
            "cheapest_price": cheapest_entry["cheapest_price"],
            "cheapest_hotel": cheapest_entry["cheapest_hotel"],
            "location": cheapest_entry["location"],
        } if cheapest_entry else None,
        "bestDateResults": best_date_hotels,
        "totalBeachfront": len(best_date_hotels["beachfront"]),
        "totalNonBeachfront": len(best_date_hotels["non_beachfront"]),
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


def _ota_search_url(provider_code, hotel_name, checkin, checkout):
    """Construct a search URL for the given OTA pre-filled with hotel + dates."""
    from urllib.parse import quote_plus
    q = quote_plus(hotel_name)
    urls = {
        "BookingCom":  f"https://www.booking.com/searchresults.html?ss={q}&checkin={checkin}&checkout={checkout}",
        "Agoda":       f"https://www.agoda.com/search?q={q}&checkIn={checkin}&los=1",
        "CtripTA":     f"https://www.trip.com/hotels/list?keyword={q}&checkIn={checkin}&checkOut={checkout}",
        "Expedia":     f"https://www.expedia.com/Hotel-Search?destination={q}&startDate={checkin}&endDate={checkout}",
        "HotelsCom":   f"https://www.hotels.com/search.do?q-destination={q}&q-check-in={checkin}&q-check-out={checkout}",
        "Traveloka":   f"https://www.traveloka.com/en-sg/hotel/search?spec={checkin}.{checkout}.1.0.HOTEL_GEO.{q}",
        "VioTA":       f"https://www.vio.com/Hotels/Search?q={q}&checkin={checkin}&checkout={checkout}",
        "Vio":         f"https://www.vio.com/Hotels/Search?q={q}&checkin={checkin}&checkout={checkout}",
    }
    return urls.get(provider_code, "")


def fetch_xotelo_prices(hotel_key, hotel_name, checkin, checkout, currency="USD"):
    """Fetch per-OTA prices + booking links from the free Xotelo API."""
    try:
        import requests as req
        res = req.get("https://data.xotelo.com/api/rates", params={
            "hotel_key": hotel_key,
            "chk_in": checkin,
            "chk_out": checkout,
            "currency": currency,
        }, timeout=12)
        if res.status_code != 200:
            return {}
        data = res.json()
        rates = data.get("result", {}).get("rates", [])
        result = {}
        for r in rates:
            if r.get("rate"):
                url = _ota_search_url(r.get("code", ""), hotel_name, checkin, checkout)
                result[r["name"]] = {"rate": r["rate"], "tax": r.get("tax", 0), "url": url}
        return result
    except Exception:
        return {}


@app.route("/api/compare-prices", methods=["POST"])
def api_compare_prices():
    """Fetch multi-provider prices from Google entity pages + Xotelo API."""
    data = request.json
    hotels = data.get("hotels", [])
    checkin = data.get("checkin", "2026-06-15")
    checkout = data.get("checkout", "2026-06-16")
    enriched = []

    def enrich(h):
        providers = {}
        xotelo = {}

        # Source 1: Google Hotels entity page
        if h.get("url"):
            providers = fetch_provider_prices(h["url"])
            time.sleep(0.2)

        # Source 2: Xotelo API (needs TripAdvisor key lookup)
        ta_key = resolve_tripadvisor_key(h.get("name", ""), h.get("location", ""))
        if ta_key:
            ci = h.get("checkin", checkin)
            co = h.get("checkout", checkout)
            xotelo = fetch_xotelo_prices(ta_key, ci, co)
            time.sleep(0.2)

        # Merge: Xotelo returns {name: {rate, tax, url}}, Google returns {name: price}
        # Normalize everything to {rate, url} dicts
        merged = {}
        for name, price in providers.items():
            merged[name] = {"rate": price, "url": ""}

        norm_map = {
            "booking": "booking.com", "agoda": "agoda",
            "trip": "trip.com", "expedia": "expedia",
            "hotels": "hotels.com", "traveloka": "traveloka",
            "vio": "vio.com",
        }
        for name, info in xotelo.items():
            rate = info["rate"] if isinstance(info, dict) else info
            url = info.get("url", "") if isinstance(info, dict) else ""
            key = name.lower().replace(".com", "").replace(" ", "").strip()
            matched = False
            for prefix, norm_name in norm_map.items():
                if prefix in key:
                    existing = merged.get(norm_name)
                    if existing is None or rate < existing["rate"]:
                        merged[norm_name] = {"rate": rate, "url": url}
                    elif not existing.get("url"):
                        existing["url"] = url  # keep lower price, add URL
                    matched = True
                    break
            if not matched:
                merged[name] = {"rate": rate, "url": url}

        return {**h, "providers": merged, "xotelo_key": ta_key}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(enrich, h) for h in hotels[:15]]
        for f in as_completed(futures):
            enriched.append(f.result())

    enriched.sort(key=lambda h: h.get("price", 9999))
    return jsonify({"hotels": enriched})


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
