import type { CollectedActivationTensor } from "./activationDefaults";
import type { Statistics2PairOp } from "../../graph/statistics2Pair";

export type { Statistics2PairOp };

export type Statistics2NodeData = {
  /**
   * Two-operand NumPy-style subscripts, e.g. `ij,ik->jk` (same as tensordot on axis 0).
   * Letters label axes; sizes must agree for shared labels.
   */
  einsumSubscripts: string;
  pairReduction: Statistics2PairOp;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

/** Older graphs stored per-tensor axis indices. */
export type Statistics2NodeDataWithLegacy = Statistics2NodeData & {
  tensor1Axis?: number;
  tensor2Axis?: number;
};

export function defaultStatistics2Data(): Statistics2NodeData {
  return {
    einsumSubscripts: "ij,ik->jk",
    pairReduction: "dot",
    outputTensor: null,
    lastError: null,
  };
}

/**
 * Same semantics as the old tensordot-on-one-axis-per-tensor pair, encoded as NumPy einsum.
 * Contracted dimension uses the label `i`; free axes get distinct letters across both operands.
 */
export function legacyAxesToPairEinsum(axisA: number, axisB: number, rank: number): string {
  const pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ".split("");
  const used = new Set<string>(["i"]);
  const pickFree = (): string => {
    for (const c of pool) {
      if (!used.has(c)) {
        used.add(c);
        return c;
      }
    }
    return "?";
  };
  const l0 = Array.from({ length: rank }, (_, d) => (d === axisA ? "i" : pickFree())).join("");
  const l1 = Array.from({ length: rank }, (_, d) => (d === axisB ? "i" : pickFree())).join("");
  const out0 = Array.from({ length: rank }, (_, d) => (d === axisA ? "" : l0[d]!)).join("");
  const out1 = Array.from({ length: rank }, (_, d) => (d === axisB ? "" : l1[d]!)).join("");
  return `${l0},${l1}->${out0}${out1}`;
}

export function resolvedStatistics2Einsum(
  raw: Partial<Statistics2NodeDataWithLegacy>,
  rank: number | null,
  fallback: Statistics2NodeData,
): string {
  const trimmed = typeof raw.einsumSubscripts === "string" ? raw.einsumSubscripts.trim().replace(/\s+/g, "") : "";
  if (trimmed.length > 0) {
    return trimmed;
  }
  if (
    rank != null &&
    rank >= 1 &&
    (typeof raw.tensor1Axis === "number" || typeof raw.tensor2Axis === "number")
  ) {
    const a1 = Math.floor(Number(raw.tensor1Axis ?? 0));
    const a2 = Math.floor(Number(raw.tensor2Axis ?? 0));
    const ax1 = Math.max(0, Math.min(rank - 1, a1));
    const ax2 = Math.max(0, Math.min(rank - 1, a2));
    return legacyAxesToPairEinsum(ax1, ax2, rank);
  }
  return fallback.einsumSubscripts.trim().replace(/\s+/g, "") || "ij,ik->jk";
}
