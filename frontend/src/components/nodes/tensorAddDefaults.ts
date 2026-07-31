import type { CollectedActivationTensor } from "./activationDefaults";

export type TensorAddNodeData = {
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

export function defaultTensorAddData(partial?: Partial<TensorAddNodeData>): TensorAddNodeData {
  return {
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
