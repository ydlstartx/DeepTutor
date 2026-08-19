"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  ChevronRight,
  Database,
  FolderOpen,
  Loader2,
} from "lucide-react";
import Modal from "@/components/common/Modal";
import {
  listKnowledgeImportFolders,
  probeKnowledgeImportFolder,
  type KnowledgeImportFolderListing,
  type KnowledgeImportProbe,
  type KnowledgeImportResult,
} from "@/lib/knowledge-api";

interface ImportKbModalProps {
  isOpen: boolean;
  uploadDirectory: string;
  onClose: () => void;
  onImport: (params: {
    path: string;
    name: string;
  }) => Promise<KnowledgeImportResult>;
}

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

export default function ImportKbModal({
  isOpen,
  uploadDirectory,
  onClose,
  onImport,
}: ImportKbModalProps) {
  const { t } = useTranslation();
  const [listing, setListing] = useState<KnowledgeImportFolderListing | null>(
    null,
  );
  const [probe, setProbe] = useState<KnowledgeImportProbe | null>(null);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [probing, setProbing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    setProbe(null);
    setName("");
    try {
      setListing(await listKnowledgeImportFolders(path));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    setListing(null);
    setProbe(null);
    setName("");
    setError(null);
    void browse("");
  }, [browse, isOpen]);

  const handleProbe = async () => {
    if (!listing?.path || probing) return;
    setProbing(true);
    setError(null);
    try {
      const result = await probeKnowledgeImportFolder(listing.path);
      setProbe(result);
      setName(result.suggested_name);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setProbing(false);
    }
  };

  const handleImport = async () => {
    if (!probe?.ok || !name.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onImport({ path: probe.path, name: name.trim() });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const displayPath = listing?.path
    ? `${uploadDirectory}/${listing.path}`
    : uploadDirectory;

  return (
    <Modal
      isOpen={isOpen}
      onClose={submitting ? () => {} : onClose}
      title={t("Import existing knowledge base")}
      titleIcon={<FolderOpen size={16} />}
      width="lg"
      closeOnBackdrop={!submitting}
      closeOnEscape={!submitting}
      footer={
        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded-md px-3 py-1.5 text-[12.5px] font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-40"
          >
            {t("Cancel")}
          </button>
          <button
            type="button"
            onClick={() => void handleImport()}
            disabled={!probe?.ok || !name.trim() || submitting}
            className="inline-flex items-center gap-1.5 rounded-md bg-[var(--primary)] px-3.5 py-1.5 text-[12.5px] font-medium text-[var(--primary-foreground)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Database className="h-3.5 w-3.5" />
            )}
            {t("Import")}
          </button>
        </div>
      }
    >
      <div className="space-y-4 px-5 py-4">
        <div className="rounded-xl border border-sky-200 bg-sky-50 px-3 py-2.5 text-[12px] leading-relaxed text-sky-800 dark:border-sky-900/60 dark:bg-sky-950/20 dark:text-sky-200">
          {t(
            "Choose a complete knowledge base from the server upload directory. DeepTutor copies its existing index and does not rebuild it.",
          )}
        </div>

        <div className="overflow-hidden rounded-xl border border-[var(--border)]">
          <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--muted)]/30 px-3 py-2">
            <button
              type="button"
              onClick={() => void browse(listing?.parent ?? "")}
              disabled={
                loading || probing || listing?.parent === null || !listing
              }
              aria-label={t("Parent folder")}
              className="rounded-md p-1 text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-30"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
            </button>
            <FolderOpen className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />
            <span className="truncate font-mono text-[11.5px] text-[var(--foreground)]">
              {displayPath}
            </span>
            {loading && (
              <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-[var(--muted-foreground)]" />
            )}
          </div>

          <div className="max-h-64 min-h-36 overflow-y-auto p-1.5">
            {!loading && listing?.folders.length === 0 ? (
              <div className="flex min-h-32 items-center justify-center px-4 text-center text-[12px] text-[var(--muted-foreground)]">
                {listing.candidate
                  ? t("This folder contains a knowledge base.")
                  : t("No folders found here.")}
              </div>
            ) : (
              listing?.folders.map((folder) => (
                <button
                  key={folder.path}
                  type="button"
                  onClick={() => void browse(folder.path)}
                  disabled={loading || probing || submitting}
                  className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-[var(--muted)] disabled:opacity-50"
                >
                  <FolderOpen className="h-4 w-4 shrink-0 text-[var(--muted-foreground)]" />
                  <span className="min-w-0 flex-1 truncate text-[12.5px] text-[var(--foreground)]">
                    {folder.name}
                  </span>
                  {folder.candidate && (
                    <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300">
                      {t("Knowledge base")}
                    </span>
                  )}
                  <ChevronRight className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />
                </button>
              ))
            )}
          </div>

          {listing?.candidate && !probe && (
            <div className="flex items-center justify-between gap-3 border-t border-[var(--border)] px-3 py-2.5">
              <span className="text-[11.5px] text-[var(--muted-foreground)]">
                {t("Validate this folder before importing it.")}
              </span>
              <button
                type="button"
                onClick={() => void handleProbe()}
                disabled={probing || loading}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 py-1.5 text-[12px] font-medium text-[var(--foreground)] hover:border-[var(--ring)] disabled:opacity-40"
              >
                {probing ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                )}
                {t("Use this folder")}
              </button>
            </div>
          )}
        </div>

        {probe && (
          <div
            className={`rounded-xl border px-3 py-3 ${
              probe.ok
                ? "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/20"
                : "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/20"
            }`}
          >
            <div className="flex items-start gap-2">
              {probe.ok ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
              ) : (
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
              )}
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] font-medium text-[var(--foreground)]">
                  {probe.ok
                    ? t("Ready to import")
                    : t("This folder cannot be imported")}
                </div>
                {probe.ok ? (
                  <>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-[var(--muted-foreground)]">
                      <span>{probe.provider}</span>
                      <span>
                        {t("{{count}} index versions", {
                          count: probe.ready_version_count,
                        })}
                      </span>
                      {probe.document_count !== null && (
                        <span>
                          {t("{{count}} indexed items", {
                            count: probe.document_count,
                          })}
                        </span>
                      )}
                      <span>{formatBytes(probe.size_bytes)}</span>
                    </div>
                    {probe.warnings.length > 0 && (
                      <ul className="mt-2 space-y-1 text-[11px] text-amber-700 dark:text-amber-300">
                        {probe.warnings.map((warning) => (
                          <li key={warning}>• {warning}</li>
                        ))}
                      </ul>
                    )}
                  </>
                ) : (
                  <p className="mt-1 text-[11.5px] text-red-700 dark:text-red-300">
                    {probe.error}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {probe?.ok && (
          <div>
            <label className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
              {t("Knowledge base name")}
            </label>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={submitting}
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-[13px] text-[var(--foreground)] outline-none transition-colors focus:border-[var(--foreground)]/25 disabled:opacity-50"
            />
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[12px] text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">
            {error}
          </div>
        )}
      </div>
    </Modal>
  );
}
