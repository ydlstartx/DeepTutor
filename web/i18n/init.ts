import i18n, { type ResourceLanguage } from "i18next";
import { initReactI18next } from "react-i18next";

export type AppLanguage = "en" | "zh";

export function normalizeLanguage(lang: unknown): AppLanguage {
  if (!lang) return "en";
  const s = String(lang).toLowerCase();
  if (s === "zh" || s === "cn" || s === "chinese") return "zh";
  return "en";
}

// i18next calls each plugin's `init` during i18n.init() — that's what makes
// react-i18next's useTranslation find the instance (getI18n). We must init
// SYNCHRONOUSLY at module load so every component that calls useTranslation
// before language bundles finish loading gets a registered instance instead
// of the NO_I18NEXT_INSTANCE warning. Resources are intentionally empty here;
// they load on demand via ensureLanguage() (the default `en` keys are the
// English text itself, so first paint is indistinguishable from translated
// output until the bundle arrives and changeLanguage re-renders).
i18n.use(initReactI18next).init({
  resources: {},
  lng: "en",
  fallbackLng: "en",
  // Use a single default namespace to keep lookups simple.
  // We intentionally keep keySeparator disabled so keys like "Generating..." remain valid.
  defaultNS: "app",
  ns: ["app"],
  keySeparator: false,
  interpolation: {
    escapeValue: false,
  },
  returnEmptyString: false,
  returnNull: false,
});

let _initPromise: Promise<typeof i18n> | null = null;

export function initI18n(language?: unknown): Promise<typeof i18n> {
  if (_initPromise) return _initPromise;
  const lng = normalizeLanguage(language);
  _initPromise = (async () => {
    await ensureLanguage(lng);
    if (i18n.language !== lng) {
      await i18n.changeLanguage(lng);
    }
    return i18n;
  })();
  return _initPromise;
}

async function fetchBundle(language: AppLanguage): Promise<ResourceLanguage> {
  // `default` interop differs between bundlers (webpack ESM) and compiled CJS
  // (tsc emits `.default` on a plain JSON require); accept both shapes.
  const mod = language === "zh"
    ? await import("@/locales/zh/app.json")
    : await import("@/locales/en/app.json");
  return ((mod as { default?: unknown }).default ?? mod) as ResourceLanguage;
}

async function loadResourceBundle(language: AppLanguage): Promise<void> {
  if (i18n.hasResourceBundle(language, "app")) return;
  i18n.addResourceBundle(language, "app", await fetchBundle(language), true, true);
}

export async function ensureLanguage(language: AppLanguage) {
  // The requested language plus the fallback (en) so missing keys in
  // partial translations still resolve instead of leaking raw keys.
  const wanted = language === "zh" ? (["zh", "en"] as const) : (["en"] as const);
  await Promise.all(wanted.map(loadResourceBundle));
}
