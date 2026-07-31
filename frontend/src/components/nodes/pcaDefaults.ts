import type { CollectedActivationTensor } from "./activationDefaults";

export type PcaNodeData = {
  /** Filled after Run; optional legacy override if present and valid in upstream collected tensors. */
  representationId: string;
  /** 0 = use all components (min(samples, features)); default 2 for typical 2D projection. */
  nComponents: number;
  /** Centered input projected onto principal axes (samples × k), after Run. */
  transformedTensor: CollectedActivationTensor | null;
  principalComponents: CollectedActivationTensor | null;
  explainedVarianceRatio: number[] | null;
  pcaSummary: string | null;
};

export function defaultPcaData(): PcaNodeData {
  return {
    representationId: "",
    nComponents: 2,
    transformedTensor: null,
    principalComponents: null,
    explainedVarianceRatio: null,
    pcaSummary: null,
  };
}
