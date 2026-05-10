// Port of fetch_xotelo_prices and _ota_search_url from app.py.
// Cache lives per warm Edge Function instance only — Supabase recycles
// instances frequently, so this is best-effort. The Xotelo API itself is fast
// enough that we don't need a Postgres-backed cache for v1.

type OtaInfo = { rate: number; tax?: number; url: string };

const cache = new Map<string, Record<string, OtaInfo & { _code?: string }>>();

function otaSearchUrl(
  code: string,
  hotelName: string,
  checkin: string,
  checkout: string,
  language: string,
  region: string | null,
  currency: string,
): string {
  const q = encodeURIComponent(hotelName);
  const lang = (language || "en").toLowerCase();
  const reg = (region || "us").toLowerCase();
  const locale = `${lang}-${reg}`;
  const cur = (currency || "USD").toUpperCase();
  const map: Record<string, string> = {
    BookingCom: `https://www.booking.com/searchresults.html?ss=${q}&checkin=${checkin}&checkout=${checkout}&selected_currency=${cur}&lang=${lang}`,
    Agoda: `https://www.agoda.com/search?q=${q}&checkIn=${checkin}&los=1&currency=${cur}&locale=${locale}`,
    CtripTA: `https://www.trip.com/hotels/list?keyword=${q}&checkIn=${checkin}&checkOut=${checkout}&curr=${cur}&locale=${locale}`,
    Expedia: `https://www.expedia.com/Hotel-Search?destination=${q}&startDate=${checkin}&endDate=${checkout}&currency=${cur}&langid=${lang}`,
    HotelsCom: `https://www.hotels.com/search.do?q-destination=${q}&q-check-in=${checkin}&q-check-out=${checkout}&currency=${cur}&locale=${locale}`,
    Traveloka: `https://www.traveloka.com/${locale}/hotel/search?spec=${checkin}.${checkout}.1.0.HOTEL_GEO.${q}`,
    VioTA: `https://www.vio.com/Hotels/Search?q=${q}&checkin=${checkin}&checkout=${checkout}&currency=${cur}`,
    Vio: `https://www.vio.com/Hotels/Search?q=${q}&checkin=${checkin}&checkout=${checkout}&currency=${cur}`,
  };
  return map[code] ?? "";
}

export async function fetchXoteloPrices(opts: {
  hotelKey: string;
  hotelName: string;
  checkin: string;
  checkout: string;
  currency?: string;
  language?: string;
  region?: string | null;
}): Promise<Record<string, OtaInfo>> {
  const cur = (opts.currency || "USD").toUpperCase();
  const lang = opts.language || "en";
  const reg = opts.region || "";
  const cacheKey = [opts.hotelKey, opts.checkin, opts.checkout, cur, lang, reg].join("|");

  const stamp = (entry: Record<string, OtaInfo & { _code?: string }>) => {
    const out: Record<string, OtaInfo> = {};
    for (const [name, info] of Object.entries(entry)) {
      out[name] = {
        rate: info.rate,
        tax: info.tax ?? 0,
        url: otaSearchUrl(info._code ?? "", opts.hotelName, opts.checkin, opts.checkout, lang, opts.region ?? null, cur),
      };
    }
    return out;
  };

  const cached = cache.get(cacheKey);
  if (cached) return stamp(cached);

  try {
    const url = new URL("https://data.xotelo.com/api/rates");
    url.searchParams.set("hotel_key", opts.hotelKey);
    url.searchParams.set("chk_in", opts.checkin);
    url.searchParams.set("chk_out", opts.checkout);
    url.searchParams.set("currency", cur);
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 12_000);
    const res = await fetch(url, { signal: ctl.signal });
    clearTimeout(t);
    if (!res.ok) return {};
    const data = await res.json();
    const rates = data?.result?.rates ?? [];
    const result: Record<string, OtaInfo & { _code?: string }> = {};
    for (const r of rates) {
      if (r?.rate) {
        const code = r.code ?? "";
        result[r.name] = {
          rate: r.rate,
          tax: r.tax ?? 0,
          url: otaSearchUrl(code, opts.hotelName, opts.checkin, opts.checkout, lang, opts.region ?? null, cur),
          _code: code,
        };
      }
    }
    cache.set(cacheKey, result);
    return stamp(result);
  } catch {
    return {};
  }
}
