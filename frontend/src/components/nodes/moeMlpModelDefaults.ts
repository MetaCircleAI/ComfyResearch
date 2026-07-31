import type { ListOr1 } from "./multiValueUtils";
import type { MlpActivationId } from "./mlpModelDefaults";

/** MoE MLP: softmax gate over K experts; each expert is an MLP block stack. */
export type MoeMlpModelNodeData = {
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  depth: ListOr1<number>;
  width: ListOr1<number>;
  numExperts: ListOr1<number>;
  activation: ListOr1<MlpActivationId>;
  /** PyTorch RNG seed for weight initialization. */
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultMoeMlpModelData(): MoeMlpModelNodeData {
  return {
    inputDim: 10,
    outputDim: 1,
    depth: 2,
    width: 32,
    numExperts: 4,
    activation: "silu",
    seed: 0,
  };
}
