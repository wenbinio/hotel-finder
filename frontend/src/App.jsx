import { useState } from "react";
import { useTranslation } from "react-i18next";
import LocaleBar from "./components/LocaleBar";
import SearchForm from "./components/SearchForm";
import HotelList from "./components/HotelList";
import { searchOne } from "./api";

const stored = key => {
  if (typeof localStorage === "undefined") return null;
  return localStorage.getItem(key);
};
const persist = (key, value) => {
  if (typeof localStorage === "undefined") return;
  if (value == null) localStorage.removeItem(key);
  else localStorage.setItem(key, value);
};

export default function App() {
  const { t, i18n } = useTranslation();

  const [locale, setLocale] = useState({
    language: i18n.language || "en",
    currency: stored("hf.currency") || "USD",
    region: stored("hf.region") || null,
  });

  const [results, setResults] = useState(null);
  const [searchInput, setSearchInput] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  function updateLocale(patch) {
    const next = { ...locale, ...patch };
    setLocale(next);
    if (patch.currency !== undefined) persist("hf.currency", next.currency);
    if (patch.region !== undefined) persist("hf.region", next.region);
  }

  async function runSearch(input) {
    setBusy(true);
    setError(null);
    setSearchInput(input);
    try {
      const r = await searchOne({ ...input, ...locale });
      setResults(r);
    } catch (e) {
      setError(e.message);
      setResults(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t("title")}</h1>
        <p className="subtitle">{t("subtitle")}</p>
        <LocaleBar
          language={locale.language}
          currency={locale.currency}
          region={locale.region}
          onChange={updateLocale}
        />
      </header>

      <main>
        <SearchForm onSubmit={runSearch} busy={busy} />

        {error && (
          <div className="error">
            <strong>{t("errorTitle")}:</strong> {error}
          </div>
        )}

        {results && (
          <section>
            <h2>{t("resultsFor", { location: results.location })}</h2>
            <HotelList
              hotels={results.hotels}
              currency={results.currency || locale.currency}
              language={locale.language}
              checkin={searchInput?.checkin}
              checkout={searchInput?.checkout}
              region={locale.region}
            />
          </section>
        )}
      </main>
    </div>
  );
}
