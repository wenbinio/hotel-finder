import { useTranslation } from "react-i18next";
import { SUPPORTED_LANGUAGES } from "../i18n";

// Common ISO 4217 codes the backend price-extractor recognizes.
export const SUPPORTED_CURRENCIES = [
  "USD", "EUR", "GBP", "JPY", "CNY", "INR", "THB", "AUD", "CAD",
  "SGD", "MYR", "IDR", "VND", "KRW", "HKD", "TWD", "PHP", "CHF",
  "NZD", "BRL", "MXN", "ZAR", "TRY", "AED", "SAR", "PLN", "SEK",
  "NOK", "DKK", "CZK",
];

export const SUPPORTED_REGIONS = [
  ["", "regionAuto"],
  ["us", "United States"],
  ["gb", "United Kingdom"],
  ["fr", "France"],
  ["de", "Germany"],
  ["es", "Spain"],
  ["it", "Italy"],
  ["jp", "Japan"],
  ["sg", "Singapore"],
  ["au", "Australia"],
  ["ca", "Canada"],
  ["br", "Brazil"],
  ["mx", "Mexico"],
  ["in", "India"],
  ["th", "Thailand"],
  ["id", "Indonesia"],
  ["my", "Malaysia"],
  ["vn", "Vietnam"],
];

export default function LocaleBar({ language, currency, region, onChange }) {
  const { t, i18n } = useTranslation();

  return (
    <div className="locale-bar">
      <label>
        {t("language")}
        <select
          value={language}
          onChange={e => {
            const v = e.target.value;
            i18n.changeLanguage(v);
            onChange({ language: v });
          }}
        >
          {SUPPORTED_LANGUAGES.map(l => (
            <option key={l.code} value={l.code}>{l.name}</option>
          ))}
        </select>
      </label>

      <label>
        {t("currency")}
        <select
          value={currency}
          onChange={e => onChange({ currency: e.target.value })}
        >
          {SUPPORTED_CURRENCIES.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </label>

      <label>
        {t("region")}
        <select
          value={region || ""}
          onChange={e => onChange({ region: e.target.value || null })}
        >
          {SUPPORTED_REGIONS.map(([code, name]) => (
            <option key={code} value={code}>
              {code === "" ? t("regionAuto") : name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
