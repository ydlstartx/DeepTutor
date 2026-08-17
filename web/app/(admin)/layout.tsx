import PublicSiteFooter from "@/components/layout/PublicSiteFooter";

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col bg-[var(--background)]">
      <div className="min-h-0 flex-1">{children}</div>
      <PublicSiteFooter />
    </div>
  );
}
