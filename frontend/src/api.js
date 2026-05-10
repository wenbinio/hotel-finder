// Thin wrapper around the Flask API. Centralized so language/currency/region
// always travel with every request without each component re-passing them.
//
// `comparePrices` and `getDestinations` try the Supabase Edge Function first
// (when VITE_SUPABASE_URL is configured) and fall back to Flask on any error.
// All other endpoints still go direct to Flask — only those two have been
// ported. See supabase/functions/.

const BASE = ""; // same-origin in prod; Vite proxies /api during dev

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "";
const EDGE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

async function postJson(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text || res.statusText}`);
  }
  return res.json();
}

async function callEdge(name, init) {
  const res = await fetch(`${SUPABASE_URL}/functions/v1/${name}`, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
    },
  });
  if (!res.ok) throw new Error(`edge ${name} ${res.status}`);
  return res.json();
}

export function searchOne({ location, checkin, checkout, minStars, language, currency, region, maxPrice }) {
  return postJson("/api/search", {
    location, checkin, checkout, minStars,
    language, currency, region, maxPrice,
  });
}

export async function comparePrices({ hotels, checkin, checkout, language, currency, region }) {
  const body = { hotels, checkin, checkout, language, currency, region };
  if (EDGE_ENABLED) {
    try {
      return await callEdge("compare-prices", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      console.warn("Edge compare-prices failed, falling back to Flask:", err);
    }
  }
  return postJson("/api/compare-prices", body);
}

export async function getDestinations() {
  if (EDGE_ENABLED) {
    try {
      return await callEdge("destinations", { method: "GET" });
    } catch (err) {
      console.warn("Edge destinations failed, falling back to Flask:", err);
    }
  }
  const res = await fetch(`${BASE}/api/destinations`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}
