export type FlattenNodeData = {
  /** Dimension index to keep as its own axis (0-based; negatives count from end). All other axes merge. `null` = full flatten to one dimension. */
  exceptDim: number | null;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function readFlattenExceptDim(raw: unknown): number | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === "string") {
    const t = raw.trim().toLowerCase();
    if (t === "" || t === "null") return null;
    const n = Number.parseInt(t, 10);
    return Number.isFinite(n) ? n : null;
  }
  if (typeof raw === "number" && Number.isFinite(raw)) return Math.trunc(raw);
  if (Array.isArray(raw) && raw.length) return readFlattenExceptDim(raw[0]);
  return null;
}

export function defaultFlattenNodeData(): FlattenNodeData {
  return {
    exceptDim: null,
    ioMode: "input-output",
    levelMode: "high",
  };
}
