import { useState } from "react";
import { useTranslation } from "react-i18next";

const today = () => new Date().toISOString().slice(0, 10);
const inDays = n => {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
};

export default function SearchForm({ onSubmit, busy }) {
  const { t } = useTranslation();
  const [location, setLocation] = useState("");
  const [checkin, setCheckin] = useState(inDays(30));
  const [checkout, setCheckout] = useState(inDays(31));
  const [minStars, setMinStars] = useState(5);
  const [maxPrice, setMaxPrice] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (!location.trim()) return;
    onSubmit({
      location: location.trim(),
      checkin,
      checkout,
      minStars,
      maxPrice: maxPrice ? Number(maxPrice) : undefined,
    });
  }

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <label className="grow">
        {t("destination")}
        <input
          type="text"
          value={location}
          onChange={e => setLocation(e.target.value)}
          placeholder={t("destinationPlaceholder")}
          autoFocus
        />
      </label>

      <label>
        {t("checkin")}
        <input
          type="date"
          value={checkin}
          min={today()}
          onChange={e => setCheckin(e.target.value)}
        />
      </label>

      <label>
        {t("checkout")}
        <input
          type="date"
          value={checkout}
          min={checkin}
          onChange={e => setCheckout(e.target.value)}
        />
      </label>

      <label>
        {t("minStars")}
        <select value={minStars} onChange={e => setMinStars(Number(e.target.value))}>
          {[3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>

      <label>
        {t("maxPrice")}
        <input
          type="number"
          inputMode="numeric"
          value={maxPrice}
          onChange={e => setMaxPrice(e.target.value)}
          placeholder="—"
        />
      </label>

      <button type="submit" disabled={busy || !location.trim()}>
        {busy ? t("searching") : t("search")}
      </button>
    </form>
  );
}
