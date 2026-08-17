import {
  APACHE_LICENSE_NAME,
  APACHE_LICENSE_URL,
  ICP_RECORD_NUMBER,
  ICP_RECORD_URL,
  PUBLIC_FOOTER_ATTRIBUTION_PREFIX,
  PUBLIC_FOOTER_ATTRIBUTION_SUFFIX,
  PUBLIC_FOOTER_ARIA_LABEL,
  PUBLIC_FOOTER_NON_OFFICIAL,
  PUBLIC_PRODUCT_NAME,
  UPSTREAM_PROJECT_NAME,
  UPSTREAM_PROJECT_URL,
} from "@/lib/public-brand";

const externalLinkClass =
  "underline decoration-current/30 underline-offset-2 transition-colors hover:text-[var(--foreground)]";

export default function PublicSiteFooter() {
  return (
    <footer
      aria-label={PUBLIC_FOOTER_ARIA_LABEL}
      className="w-full shrink-0 border-t border-[var(--border)]/45 bg-[var(--background)]/95 px-3 py-1.5 text-center text-[10px] leading-4 text-[var(--muted-foreground)]"
    >
      <div className="flex flex-col items-center justify-center gap-x-2 sm:flex-row sm:flex-wrap">
        <span>
          © 2026 {PUBLIC_PRODUCT_NAME} ·{" "}
          <a
            href={ICP_RECORD_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={externalLinkClass}
          >
            {ICP_RECORD_NUMBER}
          </a>
        </span>
        <span>
          {PUBLIC_FOOTER_ATTRIBUTION_PREFIX}{" "}
          <a
            href={UPSTREAM_PROJECT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={externalLinkClass}
          >
            {UPSTREAM_PROJECT_NAME}
          </a>{" "}
          {PUBLIC_FOOTER_ATTRIBUTION_SUFFIX} ·{" "}
          <a
            href={APACHE_LICENSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={externalLinkClass}
          >
            {APACHE_LICENSE_NAME}
          </a>{" "}
          · {PUBLIC_FOOTER_NON_OFFICIAL}
        </span>
      </div>
    </footer>
  );
}
