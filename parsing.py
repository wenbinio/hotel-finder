"""Pure helpers for currency-aware price parsing and brand-based star classification.

Kept free of Flask, scraping, and network deps so it can be unit-tested in
isolation. Importing this module must not have side effects.
"""
import re

# ── Globalization: currency symbols ───────────────────────────────────────────
# ISO 4217 code → most common display symbol returned by Google. Used to build
# a per-currency price-extraction regex.
CURRENCY_SYMBOLS = {
    "USD": "$",  "EUR": "€",  "GBP": "£",  "JPY": "¥",  "CNY": "¥",
    "INR": "₹",  "THB": "฿",  "AUD": "A$", "CAD": "C$", "SGD": "S$",
    "MYR": "RM", "IDR": "Rp", "VND": "₫",  "KRW": "₩",  "HKD": "HK$",
    "TWD": "NT$","PHP": "₱",  "CHF": "CHF","NZD": "NZ$","BRL": "R$",
    "MXN": "MX$","ZAR": "R",  "TRY": "₺",  "RUB": "₽",  "AED": "AED",
    "SAR": "SAR","ILS": "₪",  "PLN": "zł", "SEK": "kr", "NOK": "kr",
    "DKK": "kr", "CZK": "Kč",
}

# Default upper bound on hotel prices in USD. Scaled below for high-denomination
# currencies (JPY, IDR, VND, …) so non-USD callers aren't filtered to zero.
DEFAULT_MAX_PRICE_USD = 1500

# Multiplier applied to DEFAULT_MAX_PRICE_USD when the requested currency is
# high-denomination. Approximate; callers can override via maxPrice.
HIGH_DENOMINATION_SCALE = {
    "JPY": 150, "KRW": 1300, "IDR": 15000, "VND": 25000,
    "INR": 85, "THB": 35, "PHP": 55, "TWD": 30, "HUF": 350, "RUB": 90,
}

# Whitespace chars that may appear inside localized numbers (regular space,
# NBSP, narrow NBSP, thin space). Used in the price-extraction regex.
_NUMBER_SPACES = "    "


def default_max_price(currency: str) -> float:
    """Return a sensible upper bound on hotel prices for the given currency."""
    cur = (currency or "USD").upper()
    return DEFAULT_MAX_PRICE_USD * HIGH_DENOMINATION_SCALE.get(cur, 1)


def currency_price_re(currency: str) -> "re.Pattern[str]":
    """Build a regex matching prices in the given currency.

    Matches the currency symbol (or ISO code) on either side of the number
    (``€100`` or ``100 €``). Also matches JS-escaped single-char symbols
    (``\\xNN``, ``\\uNNNN``) used by Google's embedded JSON. The captured
    group always ends in a digit, so trailing punctuation never leaks.
    """
    cur = (currency or "USD").upper()
    symbol = CURRENCY_SYMBOLS.get(cur, cur)
    alts = {re.escape(symbol), re.escape(cur)}
    # Many $-suffix currencies (AUD/CAD/SGD/HKD/NZD/MXN/BRL/NT$) are rendered
    # by Google as a bare ``$`` once curr= sets the context. Accept the bare
    # symbol too, otherwise we'd extract zero prices for those markets.
    if "$" in symbol and symbol != "$":
        alts.add(re.escape("$"))
    if len(symbol) == 1:
        cp = ord(symbol)
        alts.add(rf"\\x{cp:02x}")
        alts.add(rf"\\u{cp:04x}")
    sym = "(?:" + "|".join(sorted(alts, key=len, reverse=True)) + ")"
    spaces = re.escape(_NUMBER_SPACES)
    # Number: starts and ends with a digit; may contain digit-grouping
    # separators (``,`` ``.``) or locale spaces in the middle.
    num = rf"([0-9](?:[0-9.,{spaces}]*[0-9])?)"
    # Match symbol on EITHER side — Google uses prefix in most locales
    # (en-US: ``$100``) and postfix in others (de-DE: ``100 €``).
    return re.compile(rf"(?:{sym}\s?{num})|(?:{num}\s?{sym})")


def extract_price(text: str, currency: str) -> "float | None":
    """Return the first numeric price for ``currency`` found in ``text``."""
    m = currency_price_re(currency).search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return parse_localized_number(raw) if raw else None


def parse_localized_number(raw: str) -> "float | None":
    """Parse a number that may use either ``1,234.56`` or ``1.234,56`` formats.

    Heuristics:
    * Both ``.`` and ``,`` present → the rightmost is the decimal separator.
    * Only one separator type, multiple occurrences → thousands separator.
    * Only one separator, single occurrence → decimal UNLESS exactly 3 digits
      follow it (currencies almost universally use 0–2 fractional digits, so
      ``1,234`` and ``1.234`` are treated as thousands).
    """
    if raw is None:
        return None
    s = raw.strip()
    for ws in _NUMBER_SPACES:
        s = s.replace(ws, "")
    if not s:
        return None
    n_dots = s.count(".")
    n_commas = s.count(",")
    if n_dots == 0 and n_commas == 0:
        try:
            return float(s)
        except ValueError:
            return None
    if n_dots > 0 and n_commas > 0:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        sep = "." if n_dots > 0 else ","
        count = n_dots if n_dots > 0 else n_commas
        after = s.rsplit(sep, 1)[1]
        if count > 1 or len(after) == 3:
            s = s.replace(sep, "")
        elif sep == ",":
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ── Brand-based star classification ───────────────────────────────────────────
# Brand names tend to be English globally, so this stays language-agnostic.
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

# Words that disqualify a brand match (it's a villa/hostel/guesthouse, not the
# chain). English-only on purpose: Google's `q=N star hotels` query is itself
# English so the listings tend to use English property-type words even in
# non-English locales.
DISQUALIFIERS = ["villa", "hostel", "guesthouse", "guest house", "homestay",
                 "apartment", "dormitory", "capsule", "backpacker"]


def brand_star_class(name: str) -> "int | None":
    """Return the star class if ``name`` matches a known luxury brand, else None."""
    nl = name.lower()
    if any(dq in nl for dq in DISQUALIFIERS):
        return None
    for star_class in (5, 4):
        for brand in LUXURY_BRANDS[star_class]:
            if nl.startswith(brand) or f" {brand}" in f" {nl}":
                return star_class
    return None


def extract_star_class(label: str) -> "int | None":
    """Extract the star class from a localized Google label like ``5-star hotel``,
    ``Hôtel 5 étoiles``, or ``5성급 호텔``. Returns the first 1–5 digit found."""
    if not label:
        return None
    m = re.search(r"([1-5])", label)
    return int(m.group(1)) if m else None
