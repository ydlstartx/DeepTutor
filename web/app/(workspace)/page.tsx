import { redirect } from "next/navigation";

/**
 * Root page now redirects to /home.
 * Handles backward compatibility for /?session=xxx URLs.
 * Server-side redirect: no client bundle + hydration round-trip on first load.
 */
export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const sessionId = typeof params.session === "string" ? params.session : undefined;
  const capability = typeof params.capability === "string" ? params.capability : undefined;
  const tools = Array.isArray(params.tool)
    ? params.tool.filter((t): t is string => typeof t === "string")
    : typeof params.tool === "string"
      ? [params.tool]
      : [];

  let target = sessionId ? `/home/${sessionId}` : "/home";

  const query: string[] = [];
  if (capability) query.push(`capability=${encodeURIComponent(capability)}`);
  tools.forEach((t) => query.push(`tool=${encodeURIComponent(t)}`));
  if (query.length) target += `?${query.join("&")}`;

  redirect(target);
}
