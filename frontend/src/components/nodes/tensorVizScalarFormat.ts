/** Shared formatting for scalar (0D) tensor display in tensor viz nodes. */
export function formatTensorScalarDisplay(v: number): string {
  if (!Number.isFinite(v)) {
    return Number.isNaN(v) ? "NaN" : v > 0 ? "∞" : "-∞";
  }
  if (Object.is(v, -0)) return "-0";
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return String(v);
  if (Math.abs(v) !== 0 && (Math.abs(v) < 1e-6 || Math.abs(v) >= 1e8)) {
    return v.toExponential(6);
  }
  const s = v.toPrecision(8);
  return s.includes("e") ? v.toExponential(6) : s;
}
