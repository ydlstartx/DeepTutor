import i18n, { type ResourceLanguage } from "i18next";
import { initReactI18next } from "react-i18next";

export type AppLanguage = "en" | "zh";

export function normalizeLanguage(lang: unknown): AppLanguage {
  if (!lang) return "en";
  const s = String(lang).toLowerCase();
  if (s === "zh" || s === "cn" || s === "chinese") return "zh";
  return "en";
}

// Register the React plugin eagerly (harmless, no init side effects) so
// useTranslation works even before initI18n resolves.
i18n.use(initReactI18next);

// Bundles fetched before i18n.init() resolves (hasResourceBundle/addResource
// live on the store, which only exists after init) are handed to init() via
// its `resources` option; after init they go straight into the store.
const _pendingBundles: Record<string, ResourceLanguage> = {};

let _initPromise: Promise<typeof i18n> | null = null;

export function initI18n(language?: unknown): Promise<typeof i18n> {
  if (_initPromise) return _initPromise;
  const lng = normalizeLanguage(language);
  _initPromise = (async () => {
    await ensureLanguage(lng);
    // i18next's resources shape is { lng: { ns: bundle } } — wrap the flat
    // bundles fetched before init into the namespace layer.
    const resources = Object.fromEntries(
      Object.entries(_pendingBundles).map(([lang, bundle]) => [
        lang,
        { app: bundle },
      ]),
    );
    i18n.init({
      resources,
      lng,
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
  if (i18n.store) {
    // Already initialized: instance-level resource helpers are live.
    if (i18n.hasResourceBundle(language, "app")) return;
    i18n.addResourceBundle(language, "app", await fetchBundle(language), true, true);
    return;
  }
  if (_pendingBundles[language]) return;
  _pendingBundles[language] = await fetchBundle(language);
}

export async function ensureLanguage(language: AppLanguage) {
  // The requested language plus the fallback (en) so missing keys in
  // partial translations still resolve instead of leaking raw keys.
  const wanted = language === "zh" ? (["zh", "en"] as const) : (["en"] as const);
  await Promise.all(wanted.map(loadResourceBundle));
}
