// POST /functions/v1/compare-prices — port of POST /api/compare-prices.
//
// Differences from the Flask version:
//   - Reads TripAdvisor keys from public.tripadvisor_keys instead of
//     ta_keys.json. Seed via scripts/seed_ta_keys.py.
//   - Skips the Google entity-page scrape fallback. Deno's fetch can't mimic
//     Chrome's TLS fingerprint the way curl_cffi does, so the scrape is
//     unreliable here. When Xotelo has no key, the frontend falls back to the
//     Flask /api/compare-prices route, which still does the Google path.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";
import { jsonResponse, preflight } from "../_shared/cors.ts";
import { fetchXoteloPrices } from "../_shared/xotelo.ts";

type Hotel = {
  name?: string;
  location?: string;
  url?: string;
  checkin?: string;
  checkout?: string;
  price?: number;
  [key: string]: unknown;
};

const supabase = createClient(
  Deno.env.get("SUPABASE_URL") ?? "",
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? Deno.env.get("SUPABASE_ANON_KEY") ?? "",
);

let taKeysCache: Map<string, string> | null = null;

async function loadTaKeys(): Promise<Map<string, string>> {
  if (taKeysCache) return taKeysCache;
  const { data, error } = await supabase
    .from("tripadvisor_keys")
    .select("hotel_name, ta_key");
  if (error || !data) return new Map();
  const m = new Map<string, string>();
  for (const row of data) m.set(row.hotel_name as string, row.ta_key as string);
  taKeysCache = m;
  return m;
}

function resolveTaKey(keys: Map<string, string>, hotelName: string): string | null {
  const exact = keys.get(hotelName);
  if (exact) return exact;
  const lower = hotelName.toLowerCase();
  for (const [k, v] of keys) {
    const kl = k.toLowerCase();
    if (kl.includes(lower) || lower.includes(kl)) return v;
  }
  return null;
}

const NORM_MAP: Record<string, string> = {
  booking: "booking.com",
  agoda: "agoda",
  trip: "trip.com",
  expedia: "expedia",
  hotels: "hotels.com",
  traveloka: "traveloka",
  vio: "vio.com",
};

function canonicalize(name: string): string {
  const key = name.toLowerCase().replace(".com", "").replace(/\s+/g, "").trim();
  for (const prefix in NORM_MAP) {
    if (key.includes(prefix)) return NORM_MAP[prefix];
  }
  return name;
}

function localeParams(data: Record<string, unknown>) {
  return {
    language: (data.language as string) || "en",
    currency: ((data.currency as string) || "USD").toUpperCase(),
    region: (data.region as string) || null,
  };
}

async function enrich(
  h: Hotel,
  keys: Map<string, string>,
  defaultCheckin: string,
  defaultCheckout: string,
  locale: ReturnType<typeof localeParams>,
) {
  const taKey = resolveTaKey(keys, h.name ?? "");
  const merged: Record<string, { rate: number; url: string; source: string }> = {};
  let source: string | null = null;

  if (taKey) {
    const xotelo = await fetchXoteloPrices({
      hotelKey: taKey,
      hotelName: h.name ?? "",
      checkin: h.checkin ?? defaultCheckin,
      checkout: h.checkout ?? defaultCheckout,
      currency: locale.currency,
      language: locale.language,
      region: locale.region,
    });
    for (const [name, info] of Object.entries(xotelo)) {
      const canon = canonicalize(name);
      const existing = merged[canon];
      if (!existing || info.rate < existing.rate) {
        merged[canon] = { rate: info.rate, url: info.url, source: "xotelo" };
      } else if (!existing.url) {
        existing.url = info.url;
      }
    }
    if (Object.keys(merged).length) source = "xotelo";
  }

  return {
    ...h,
    providers: merged,
    xotelo_key: taKey,
    providers_source: source,
  };
}

Deno.serve(async (req) => {
  const pre = preflight(req);
  if (pre) return pre;
  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  let data: Record<string, unknown>;
  try {
    data = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  const hotels = (data.hotels as Hotel[]) ?? [];
  const checkin = (data.checkin as string) ?? "2026-06-15";
  const checkout = (data.checkout as string) ?? "2026-06-16";
  const locale = localeParams(data);

  const keys = await loadTaKeys();
  const enriched = await Promise.all(
    hotels.slice(0, 15).map((h) => enrich(h, keys, checkin, checkout, locale)),
  );
  enriched.sort(
    (a, b) => ((a.price as number) ?? 9999) - ((b.price as number) ?? 9999),
  );

  return jsonResponse({
    hotels: enriched,
    currency: locale.currency,
    language: locale.language,
  });
});
