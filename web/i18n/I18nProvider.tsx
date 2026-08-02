"use client";

import { useEffect } from "react";
import i18n from "i18next";

import {
  ensureLanguage,
  initI18n,
  normalizeLanguage,
  type AppLanguage,
} from "./init";

/**
 * Initializes i18next asynchronously (language bundles are code-split chunks
 * now) and applies the active language. Until the first init resolves,
 * `t()` falls back to raw keys — for the default `en` the keys ARE the
 * English text, so first paint is indistinguishable from translated output;
 * the init event then re-renders with real translations.
 */
export function I18nProvider({
  language,
  children,
}: {
  language: AppLanguage | string;
  children: React.ReactNode;
}) {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await initI18n(language);
      const nextLang = normalizeLanguage(language);
      await ensureLanguage(nextLang);
      if (cancelled) return;
      if (i18n.language !== nextLang) {
        await i18n.changeLanguage(nextLang);
      }
      // Keep <html lang="..."> in sync for accessibility & Intl defaults.
      document.documentElement.lang = nextLang;
    })();
    return () => {
      cancelled = true;
    };
  }, [language]);

  return children;
}
