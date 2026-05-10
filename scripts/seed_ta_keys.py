"""Seed public.tripadvisor_keys from ta_keys.json.

Usage:
    SUPABASE_DB_URL=postgresql://... python scripts/seed_ta_keys.py

SUPABASE_DB_URL is the "Connection string" shown in Supabase Studio under
Project Settings → Database. The service role isn't needed because we connect
straight to Postgres.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
TA_KEYS_PATH = ROOT / "ta_keys.json"


def main() -> int:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("SUPABASE_DB_URL is required", file=sys.stderr)
        return 1

    keys = json.loads(TA_KEYS_PATH.read_text())
    rows = [(name, key) for name, key in keys.items()]

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            insert into public.tripadvisor_keys (hotel_name, ta_key)
            values (%s, %s)
            on conflict (hotel_name) do update
                set ta_key = excluded.ta_key,
                    updated_at = now()
            """,
            rows,
        )
        conn.commit()

    print(f"Seeded {len(rows)} TripAdvisor keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
