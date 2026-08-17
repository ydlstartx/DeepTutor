import PublicSiteFooter from "@/components/layout/PublicSiteFooter";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col bg-[var(--background)]">
      <div className="flex min-h-0 flex-1 items-center justify-center px-4 py-8">
        {children}
      </div>
      <PublicSiteFooter />
    </div>
  );
}
