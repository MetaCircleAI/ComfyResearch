/** Accepts decimal (0.0001) and scientific (1e-4) notation; value must be finite and positive. */
export function parsePositiveFloat(raw: string): number | null {
  const t = raw.trim().replaceAll(",", "");
  if (t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n;
}
