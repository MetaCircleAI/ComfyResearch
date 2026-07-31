import type { CollectedActivationTensor } from "./activationDefaults";
import type { ListOr1 } from "./multiValueUtils";

/** How to fill the tensor before optional manual edits (future). */
export type TensorConstantInit = "zero" | "uniform_m11" | "gaussian";

export type TensorConstantNodeData = {
  /** Tensor layout (positive sizes). */
  shape: number[];
  init: TensorConstantInit;
  /** RNG seed for uniform / Gaussian fills (`zeros` ignores). Same sweep pattern as model `seed`. */
  initSeed: ListOr1<number>;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

export function defaultTensorConstantData(partial?: Partial<TensorConstantNodeData>): TensorConstantNodeData {
  return {
    shape: partial?.shape?.length ? [...partial.shape] : [2, 3],
    init: partial?.init ?? "zero",
    initSeed: partial?.initSeed !== undefined ? partial.initSeed : 0,
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
