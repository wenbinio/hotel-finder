import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import es from "./locales/es.json";
import fr from "./locales/fr.json";
import de from "./locales/de.json";
import ja from "./locales/ja.json";

export const SUPPORTED_LANGUAGES = [
  { code: "en", name: "English" },
  { code: "es", name: "Español" },
  { code: "fr", name: "Français" },
  { code: "de", name: "Deutsch" },
  { code: "ja", name: "日本語" },
];

const stored =
  typeof localStorage !== "undefined"
    ? localStorage.getItem("hf.lang")
    : null;
const browser =
  typeof navigator !== "undefined"
    ? (navigator.language || "en").slice(0, 2)
    : "en";
const initial = stored || (SUPPORTED_LANGUAGES.find(l => l.code === browser) ? browser : "en");

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    es: { translation: es },
    fr: { translation: fr },
    de: { translation: de },
    ja: { translation: ja },
  },
  lng: initial,
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

i18n.on("languageChanged", lng => {
  if (typeof localStorage !== "undefined") localStorage.setItem("hf.lang", lng);
  if (typeof document !== "undefined") document.documentElement.lang = lng;
});

if (typeof document !== "undefined") document.documentElement.lang = initial;

export default i18n;
