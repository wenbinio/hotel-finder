// Thin wrapper around the Flask API. Centralized so language/currency/region
// always travel with every request without each component re-passing them.

const BASE = ""; // same-origin in prod; Vite proxies /api during dev

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

export function searchOne({ location, checkin, checkout, minStars, language, currency, region, maxPrice }) {
  return postJson("/api/search", {
    location, checkin, checkout, minStars,
    language, currency, region, maxPrice,
  });
}

export function comparePrices({ hotels, checkin, checkout, language, currency, region }) {
  return postJson("/api/compare-prices", {
    hotels, checkin, checkout, language, currency, region,
  });
}

export async function getDestinations() {
  const res = await fetch(`${BASE}/api/destinations`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}
