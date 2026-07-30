"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import * as pdfjs from "pdfjs-dist";
import type { FilePreviewSource } from "../previewerFor";
import FallbackPreview from "./FallbackPreview";

/**
 * PDF preview rendered with Mozilla's pdf.js (canvas), replacing the earlier
 * <iframe> approach: Safari shows a bare "Open" placeholder instead of its
 * PDF viewer inside iframes, so the native viewer was never reliable
 * cross-browser. Pages render lazily as they scroll near the viewport so
 * book-length documents stay cheap; the worker + library are only loaded
 * when a PDF is actually previewed (this component is dynamically imported).
 */
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const MAX_RENDER_WIDTH = 1200;
const SIDE_PADDING = 24;
// Render pages a couple of screens before they enter the viewport.
const LAZY_ROOT_MARGIN = "600px";

export default function PdfPreview({
  url,
  filename,
}: {
  url: string;
  filename: FilePreviewSource["filename"];
}) {
  const { t } = useTranslation();
  const [doc, setDoc] = useState<pdfjs.PDFDocumentProxy | null>(null);
  const [failed, setFailed] = useState(false);
  const [pageRatio, setPageRatio] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const task = pdfjs.getDocument({ url, withCredentials: true });
    task.promise
      .then((loaded) => {
        if (cancelled) return;
        setDoc(loaded);
        // Book pages are uniform — one viewport ratio serves every
        // placeholder, avoiding a getPage() per page up front.
        void loaded.getPage(1).then((page) => {
          if (cancelled) return;
          const viewport = page.getViewport({ scale: 1 });
          setPageRatio(viewport.height / viewport.width);
        });
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      void task.destroy();
    };
  }, [url]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  if (failed) {
    return <FallbackPreview filename={filename} url={url} reason="unsupported" />;
  }

  return (
    <div
      ref={containerRef}
      className="h-full w-full overflow-y-auto bg-[var(--muted)]/30"
    >
      {!doc ? (
        <div className="flex h-full items-center justify-center gap-2 text-[12px] text-[var(--muted-foreground)]">
          <Loader2 size={14} className="animate-spin" />
          <span>{t("Loading preview…")}</span>
        </div>
      ) : (
        <div className="mx-auto flex flex-col items-center gap-3 py-3">
          {Array.from({ length: doc.numPages }, (_, index) => (
            <PdfPage
              key={index + 1}
              doc={doc}
              pageNumber={index + 1}
              width={containerWidth}
              placeholderRatio={pageRatio}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PdfPage({
  doc,
  pageNumber,
  width,
  placeholderRatio,
}: {
  doc: pdfjs.PDFDocumentProxy;
  pageNumber: number;
  width: number;
  placeholderRatio: number | null;
}) {
  const holderRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderedSize, setRenderedSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  useEffect(() => {
    if (!width) return;
    const holder = holderRef.current;
    if (!holder) return;

    let cancelled = false;
    let renderTask: pdfjs.RenderTask | null = null;

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting || cancelled) return;
        observer.disconnect();
        void (async () => {
          try {
            const page = await doc.getPage(pageNumber);
            if (cancelled) return;
            const baseViewport = page.getViewport({ scale: 1 });
            const targetWidth = Math.min(
              Math.max(width - SIDE_PADDING, 200),
              MAX_RENDER_WIDTH,
            );
            const scale = targetWidth / baseViewport.width;
            const outputScale = window.devicePixelRatio || 1;
            const viewport = page.getViewport({ scale: scale * outputScale });

            const canvas = canvasRef.current;
            if (!canvas || cancelled) return;
            canvas.width = Math.floor(viewport.width);
            canvas.height = Math.floor(viewport.height);
            const cssWidth = Math.floor(baseViewport.width * scale);
            const cssHeight = Math.floor(baseViewport.height * scale);
            canvas.style.width = `${cssWidth}px`;
            canvas.style.height = `${cssHeight}px`;

            const context = canvas.getContext("2d");
            if (!context || cancelled) return;
            renderTask = page.render({
              canvas,
              canvasContext: context,
              viewport,
            });
            await renderTask.promise;
            if (!cancelled) setRenderedSize({ width: cssWidth, height: cssHeight });
          } catch {
            // RenderTask.cancel() rejects here on teardown — not an error.
          }
        })();
      },
      { rootMargin: LAZY_ROOT_MARGIN },
    );
    observer.observe(holder);

    return () => {
      cancelled = true;
      observer.disconnect();
      renderTask?.cancel();
    };
  }, [doc, pageNumber, width]);

  const placeholderWidth = width
    ? Math.min(Math.max(width - SIDE_PADDING, 200), MAX_RENDER_WIDTH)
    : undefined;

  return (
    <div
      ref={holderRef}
      className="overflow-hidden rounded-[2px] bg-white shadow-sm"
      style={
        renderedSize
          ? { width: renderedSize.width, height: renderedSize.height }
          : {
              width: placeholderWidth ?? "100%",
              aspectRatio: placeholderRatio
                ? `1 / ${placeholderRatio}`
                : "210 / 297",
            }
      }
    >
      <canvas ref={canvasRef} className="block" />
    </div>
  );
}
