import type { ListOr1 } from "./multiValueUtils";
import type { InputDistributionId, OutputDistributionId } from "./linearDatasetDefaults";

/** Describes how inputs x are drawn and optionally jittered before a teacher model sees them. */
export type RandomInputDistributionNodeData = {
  inputDim: ListOr1<number>;
  inputDistribution: ListOr1<InputDistributionId>;
  /** Extra isotropic Gaussian on x after the base draw (same semantics as linear dataset output noise). */
  noiseDistribution: ListOr1<OutputDistributionId>;
  noiseLevel: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export const defaultRandomInputDistributionData = (): RandomInputDistributionNodeData => ({
  inputDim: 10,
  inputDistribution: "standard_normal",
  noiseDistribution: "deterministic",
  noiseLevel: 0,
  seed: 0,
});
