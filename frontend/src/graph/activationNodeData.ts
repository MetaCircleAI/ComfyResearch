/** Read server-side activation fields from node data (camelCase or snake_case). */

export function readActivationRunId(data: Record<string, unknown> | null | undefined): string {
  if (!data) return "";
  const camel = data["activationRunId"];
  const snake = data["activation_run_id"];
  const v = typeof camel === "string" ? camel : typeof snake === "string" ? snake : "";
  return v.trim();
}

export type ActivationManifestJson = Record<string, { shape: number[] }>;

export function readActivationManifest(
  data: Record<string, unknown> | null | undefined,
): ActivationManifestJson | null {
  if (!data) return null;
  const camel = data["activationManifest"];
  const snake = data["activation_manifest"];
  const m = camel ?? snake;
  if (m && typeof m === "object" && !Array.isArray(m)) {
    return m as ActivationManifestJson;
  }
  return null;
}
