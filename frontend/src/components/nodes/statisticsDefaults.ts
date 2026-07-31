import type { CollectedActivationTensor } from "./activationDefaults";

export type StatisticsReductionOp =
  | "mean"
  | "median"
  | "max"
  | "min"
  | "l2_norm"
  | "l1_norm";

export type StatisticsNodeData = {
  /**
   * NumPy-style single-operand einsum subscripts, e.g. `ij->j` (mean/max/… over `i`).
   * Letters map to axes in order; only `a-z` / `A-Z`, one label per axis on the left.
   */
  einsumSubscripts: string;
  reductionOp: StatisticsReductionOp;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

/** Older graphs used axis checkboxes; migrated when computing or previewing. */
export type StatisticsNodeDataWithLegacy = StatisticsNodeData & {
  reductionAxes?: number[];
  reductionAxis?: number;
};

export function defaultStatisticsData(): StatisticsNodeData {
  return {
    einsumSubscripts: "ab->b",
    reductionOp: "mean",
    outputTensor: null,
    lastError: null,
  };
}

/** Legacy axis list → Einstein string using consecutive labels a, b, c, … */
export function legacyReductionAxesToEinsum(axes: number[], rank: number): string {
  const letters = Array.from({ length: rank }, (_, i) => String.fromCharCode(97 + i));
  const drop = new Set(axes.map((a) => Math.floor(a)).filter((a) => a >= 0 && a < rank));
  const rhs = letters.filter((_, i) => !drop.has(i)).join("");
  return `${letters.join("")}->${rhs}`;
}

export function reductionAxesFromNodeData(
  raw: Partial<StatisticsNodeDataWithLegacy>,
  _fallback: StatisticsNodeData,
): number[] {
  const fromList = raw.reductionAxes;
  if (Array.isArray(fromList) && fromList.length > 0) {
    const xs = [...new Set(fromList.map((x) => Math.floor(Number(x))))].filter((x) => Number.isFinite(x));
    if (xs.length > 0) {
      return xs.sort((a, b) => a - b);
    }
  }
  const legacy = raw.reductionAxis;
  if (typeof legacy === "number" && Number.isFinite(legacy)) {
    return [Math.floor(legacy)];
  }
  return [0];
}

export function clampReductionAxesForRank(axes: number[], rank: number): number[] {
  if (rank < 1) return [];
  const s = new Set<number>();
  for (const a of axes) {
    const ai = Math.floor(Number(a));
    if (Number.isFinite(ai) && ai >= 0 && ai < rank) {
      s.add(ai);
    }
  }
  const out = [...s].sort((a, b) => a - b);
  return out.length > 0 ? out : [0];
}

/**
 * Effective subscripts for compute / preview: explicit `einsumSubscripts`, else legacy axes, else default.
 */
export function resolvedEinsumSubscripts(
  raw: Partial<StatisticsNodeDataWithLegacy>,
  rank: number | null,
  fallback: StatisticsNodeData,
): string {
  const trimmed = typeof raw.einsumSubscripts === "string" ? raw.einsumSubscripts.trim().replace(/\s+/g, "") : "";
  if (trimmed.length > 0) {
    return trimmed;
  }
  if (rank != null && rank >= 1 && (raw.reductionAxes != null || raw.reductionAxis != null)) {
    const axes = clampReductionAxesForRank(reductionAxesFromNodeData(raw, fallback), rank);
    return legacyReductionAxesToEinsum(axes, rank);
  }
  return fallback.einsumSubscripts.trim().replace(/\s+/g, "") || "ab->b";
}
