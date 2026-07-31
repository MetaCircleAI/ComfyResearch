import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";

export type EffectiveRankNodeData = {
  outputTensor: WeightTensorPayload | null;
  lastError: string | null;
};

export function defaultEffectiveRankData(partial?: Partial<EffectiveRankNodeData>): EffectiveRankNodeData {
  return {
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
