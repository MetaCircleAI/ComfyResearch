import type { ListOr1 } from "./multiValueUtils";

/** Samples x tensors from an upstream random_input_distribution node. */
export type InputSamplerNodeData = {
  numSamples: ListOr1<number>;
};

export const defaultInputSamplerData = (): InputSamplerNodeData => ({
  numSamples: 800,
});
