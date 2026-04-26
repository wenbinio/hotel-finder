"""Unit tests for the pure parsing helpers in parsing.py.

Run with::

    python -m unittest discover

These cover the three places where the globalization rewrite was most
error-prone: localized number parsing, the per-currency price regex (with
prefix/postfix symbol placement, JS-escaped symbols, $-suffix-currency
fallback, and trailing punctuation), and brand-based star classification.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsing import (
    brand_star_class,
    currency_price_re,
    default_max_price,
    extract_price,
    extract_star_class,
    parse_localized_number,
)


class ParseLocalizedNumberTests(unittest.TestCase):
    def test_plain_integer(self):
        self.assertEqual(parse_localized_number("1234"), 1234.0)

    def test_us_thousands(self):
        self.assertEqual(parse_localized_number("1,234"), 1234.0)
        self.assertEqual(parse_localized_number("18,500"), 18500.0)

    def test_eu_thousands(self):
        # Single dot with 3 digits after → thousands separator (1234), since
        # currencies almost universally use 0-2 fractional digits.
        self.assertEqual(parse_localized_number("1.234"), 1234.0)

    def test_us_decimal(self):
        self.assertEqual(parse_localized_number("12.99"), 12.99)
        self.assertEqual(parse_localized_number("99.5"), 99.5)

    def test_eu_decimal(self):
        self.assertEqual(parse_localized_number("12,99"), 12.99)

    def test_us_full(self):
        self.assertEqual(parse_localized_number("1,234.56"), 1234.56)

    def test_eu_full(self):
        self.assertEqual(parse_localized_number("1.234,56"), 1234.56)

    def test_multiple_thousands(self):
        self.assertEqual(parse_localized_number("1,234,567"), 1234567.0)
        self.assertEqual(parse_localized_number("1.234.567"), 1234567.0)

    def test_nbsp_thousands_separator(self):
        # French/Czech/Russian use NBSP or narrow NBSP for grouping.
        self.assertEqual(parse_localized_number("1 234,56"), 1234.56)
        self.assertEqual(parse_localized_number("1 234"), 1234.0)

    def test_empty_and_garbage(self):
        self.assertIsNone(parse_localized_number(""))
        self.assertIsNone(parse_localized_number("   "))
        self.assertIsNone(parse_localized_number("abc"))

    def test_none_input(self):
        self.assertIsNone(parse_localized_number(None))


class ExtractPriceTests(unittest.TestCase):
    """Cover prefix/postfix placement, $-fallback, and JS escapes."""

    def test_usd_prefix(self):
        self.assertEqual(extract_price("rate $1,234 nightly", "USD"), 1234.0)

    def test_usd_single_digit(self):
        self.assertEqual(extract_price("just $5 stay", "USD"), 5.0)

    def test_usd_js_escaped(self):
        self.assertEqual(extract_price("rate \\\\x24450 in JS", "USD"), 450.0)

    def test_eur_prefix(self):
        self.assertEqual(extract_price("Tarif: €1.234,56", "EUR"), 1234.56)

    def test_eur_postfix(self):
        # Postfix with space — common in de-DE, fr-FR
        self.assertEqual(extract_price("Prix: 1.234,56 € HT", "EUR"), 1234.56)

    def test_eur_postfix_no_space(self):
        self.assertEqual(extract_price("Total 99,50€", "EUR"), 99.5)

    def test_jpy_prefix(self):
        self.assertEqual(extract_price("¥18,500", "JPY"), 18500.0)

    def test_jpy_postfix(self):
        self.assertEqual(extract_price("18,500 ¥", "JPY"), 18500.0)

    def test_thb(self):
        self.assertEqual(extract_price("฿4,500.50", "THB"), 4500.50)

    def test_sgd_explicit_symbol(self):
        self.assertEqual(extract_price("S$320", "SGD"), 320.0)

    def test_sgd_bare_dollar_fallback(self):
        # Critical: Google often drops the country prefix once curr=SGD sets
        # the context, so SGD must also match a bare $.
        self.assertEqual(extract_price("price was $220 there", "SGD"), 220.0)

    def test_aud_bare_dollar_fallback(self):
        self.assertEqual(extract_price("A bare $199 fare", "AUD"), 199.0)

    def test_inr(self):
        self.assertEqual(extract_price("₹15,500", "INR"), 15500.0)

    def test_gbp_decimal(self):
        self.assertEqual(extract_price("£42.99", "GBP"), 42.99)

    def test_chf_iso_code_either_side(self):
        # CHF is its own ISO code AND its own display symbol.
        self.assertEqual(extract_price("CHF 250", "CHF"), 250.0)
        self.assertEqual(extract_price("250 CHF", "CHF"), 250.0)

    def test_trailing_punctuation_ignored(self):
        # The captured group ends in a digit, so trailing dot/comma can't
        # leak into the parser.
        self.assertEqual(extract_price("Cost is $100.", "USD"), 100.0)
        self.assertEqual(extract_price("Total: 1.234,56€.", "EUR"), 1234.56)

    def test_no_match_returns_none(self):
        self.assertIsNone(extract_price("no prices here at all", "USD"))
        # Wrong currency: looking for EUR but text has USD
        self.assertIsNone(extract_price("$100 only", "EUR"))

    def test_picks_first_match(self):
        # Ensures we get the first occurrence, not the cheapest or last.
        self.assertEqual(extract_price("$100 then $50", "USD"), 100.0)


class CurrencyPriceReTests(unittest.TestCase):
    def test_unknown_currency_falls_back_to_iso_code(self):
        # E.g. an exotic ISO that's not in CURRENCY_SYMBOLS table.
        rx = currency_price_re("XYZ")
        m = rx.search("XYZ 100")
        self.assertIsNotNone(m)

    def test_currency_argument_is_case_insensitive_to_callers(self):
        self.assertEqual(extract_price("$50", "usd"), 50.0)
        self.assertEqual(extract_price("$50", "Usd"), 50.0)

    def test_none_currency_defaults_to_usd(self):
        self.assertEqual(extract_price("$50", None), 50.0)


class ExtractStarClassTests(unittest.TestCase):
    def test_english(self):
        self.assertEqual(extract_star_class("5-star hotel"), 5)
        self.assertEqual(extract_star_class("4-star"), 4)

    def test_french(self):
        self.assertEqual(extract_star_class("Hôtel 5 étoiles"), 5)

    def test_spanish(self):
        self.assertEqual(extract_star_class("Hotel de 4 estrellas"), 4)

    def test_korean(self):
        self.assertEqual(extract_star_class("5성급 호텔"), 5)

    def test_japanese(self):
        self.assertEqual(extract_star_class("ホテル 5 つ星"), 5)

    def test_only_1_to_5_match(self):
        # Hotels are 1-5 stars; '7 stars' should not match (not a valid class).
        self.assertIsNone(extract_star_class("7 stars"))
        self.assertIsNone(extract_star_class("0 stars"))

    def test_empty(self):
        self.assertIsNone(extract_star_class(""))
        self.assertIsNone(extract_star_class(None))


class BrandStarClassTests(unittest.TestCase):
    def test_5_star_brand(self):
        self.assertEqual(brand_star_class("Four Seasons Hotel Bangkok"), 5)
        self.assertEqual(brand_star_class("The Ritz-Carlton, Tokyo"), 5)

    def test_4_star_brand(self):
        self.assertEqual(brand_star_class("Hilton Garden Inn"), 4)
        self.assertEqual(brand_star_class("Novotel Bangkok"), 4)

    def test_disqualifier_villa(self):
        # Same brand string but in a property type that isn't the chain.
        self.assertIsNone(brand_star_class("Marriott Villa Resort"))

    def test_disqualifier_hostel(self):
        self.assertIsNone(brand_star_class("Conrad Hostel Bali"))

    def test_unknown_brand(self):
        self.assertIsNone(brand_star_class("Some Random Inn"))

    def test_substring_does_not_match(self):
        # 'aman ' (trailing space) shouldn't match 'Amanora' or 'Salaman'.
        self.assertIsNone(brand_star_class("Amanora Park Town"))


class DefaultMaxPriceTests(unittest.TestCase):
    def test_usd_baseline(self):
        self.assertEqual(default_max_price("USD"), 1500)

    def test_jpy_scaled(self):
        self.assertEqual(default_max_price("JPY"), 1500 * 150)

    def test_idr_scaled(self):
        self.assertEqual(default_max_price("IDR"), 1500 * 15000)

    def test_unknown_currency_keeps_baseline(self):
        self.assertEqual(default_max_price("EUR"), 1500)

    def test_none_defaults_to_usd(self):
        self.assertEqual(default_max_price(None), 1500)


if __name__ == "__main__":
    unittest.main()
