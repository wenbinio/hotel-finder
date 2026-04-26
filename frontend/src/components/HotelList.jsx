import { useState } from "react";
import { useTranslation } from "react-i18next";
import { comparePrices } from "../api";

function fmtMoney(amount, currency, language) {
  if (amount == null) return "—";
  try {
    return new Intl.NumberFormat(language || "en", {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: amount >= 1000 ? 0 : 2,
    }).format(amount);
  } catch {
    return `${currency} ${amount}`;
  }
}

function HotelCard({ hotel, currency, language, checkin, checkout, region }) {
  const { t } = useTranslation();
  const [providers, setProviders] = useState(hotel.providers);
  const [loading, setLoading] = useState(false);

  async function loadProviders() {
    setLoading(true);
    try {
      const r = await comparePrices({
        hotels: [hotel], checkin, checkout, language, currency, region,
      });
      const enriched = r.hotels && r.hotels[0];
      if (enriched) setProviders(enriched.providers);
    } finally {
      setLoading(false);
    }
  }

  return (
    <article className="hotel-card">
      <header>
        <div className="name">{hotel.name}</div>
        <div className="stars" aria-label={t("stars", { count: hotel.star_class })}>
          {"★".repeat(hotel.star_class || 0)}
        </div>
      </header>
      <div className="price-row">
        <span className="price">
          {fmtMoney(hotel.price, currency, language)}
          <span className="per-night"> {t("perNight")}</span>
        </span>
        {hotel.rating != null && (
          <span className="rating">★ {hotel.rating.toFixed(1)}</span>
        )}
      </div>
      {!!hotel.amenities?.length && (
        <ul className="amenities">
          {hotel.amenities.slice(0, 6).map(a => <li key={a}>{a}</li>)}
        </ul>
      )}
      <div className="actions">
        {hotel.url && (
          <a href={hotel.url} target="_blank" rel="noreferrer">{t("viewOnGoogle")}</a>
        )}
        {!providers && !loading && (
          <button onClick={loadProviders}>{t("compare")}</button>
        )}
        {loading && <span className="muted">{t("compareLoading")}</span>}
      </div>
      {providers && Object.keys(providers).length > 0 && (
        <ul className="providers">
          {Object.entries(providers)
            .sort((a, b) => (a[1].rate ?? 0) - (b[1].rate ?? 0))
            .map(([name, info]) => (
              <li key={name}>
                <span className="prov-name">{name}</span>
                <span className="prov-price">{fmtMoney(info.rate, currency, language)}</span>
                {info.url && (
                  <a className="prov-link" href={info.url} target="_blank" rel="noreferrer">→</a>
                )}
              </li>
            ))}
        </ul>
      )}
    </article>
  );
}

export default function HotelList({ hotels, currency, language, checkin, checkout, region }) {
  const { t } = useTranslation();
  if (!hotels || hotels.length === 0) {
    return <p className="empty">{t("noResults")}</p>;
  }
  return (
    <div className="hotel-list">
      {hotels.map((h, i) => (
        <HotelCard
          key={`${h.name}-${i}`}
          hotel={h}
          currency={currency}
          language={language}
          checkin={checkin}
          checkout={checkout}
          region={region}
        />
      ))}
    </div>
  );
}
