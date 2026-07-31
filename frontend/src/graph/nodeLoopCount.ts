/** Persisted on `Node.data.loopCount` when the user sets a loop count from the canvas context menu. */
export function readGraphNodeLoopCount(data: unknown): number | null {
  if (data == null || typeof data !== "object") return null;
  const raw = (data as Record<string, unknown>).loopCount;
  if (typeof raw === "number" && Number.isFinite(raw)) return Math.floor(raw);
  if (typeof raw === "string" && raw.trim()) {
    const n = Math.floor(Number(raw));
    if (Number.isFinite(n)) return n;
  }
  return null;
}

/** When true, looped blocks share one parameter set (default: false = separate copies). */
export function readGraphNodeLoopShareParams(data: unknown): boolean {
  if (data == null || typeof data !== "object") return false;
  const raw = (data as Record<string, unknown>).loopShareParams;
  return raw === true;
}
