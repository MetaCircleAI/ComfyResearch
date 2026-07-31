/** KAN regression node: pykan ``KAN`` with width ``[input] + depth×[hidden] + [output]`` (see pykan README). */

import type { ListOr1 } from "./multiValueUtils";

export type KanBaseFunId = "silu" | "identity" | "zero";

export type KanModelNodeData = {
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  /** Number of hidden KAN layers (same convention as MLP depth). */
  depth: ListOr1<number>;
  /** Hidden layer width (each internal layer uses this width). */
  width: ListOr1<number>;
  /** B-spline grid intervals (pykan ``grid``). */
  grid: ListOr1<number>;
  /** Spline polynomial order (pykan ``k``). */
  k: ListOr1<number>;
  /** Residual base activation name accepted by MultKAN (string form). */
  baseFun: ListOr1<KanBaseFunId>;
  /** PyTorch / pykan RNG seed for initialization. */
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export const KAN_BASE_FUN_OPTIONS: { id: KanBaseFunId; label: string }[] = [
  { id: "silu", label: "SiLU" },
  { id: "identity", label: "Identity" },
  { id: "zero", label: "Zero" },
];

export const DEFAULT_KAN_OUTPUT_DIM = 1;

export function defaultKanModelData(): KanModelNodeData {
  return {
    inputDim: 10,
    outputDim: DEFAULT_KAN_OUTPUT_DIM,
    depth: 2,
    width: 5,
    grid: 3,
    k: 3,
    baseFun: "silu",
    seed: 0,
  };
}
