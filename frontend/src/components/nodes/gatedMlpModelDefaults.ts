import type { ListOr1 } from "./multiValueUtils";
import type { MlpActivationId } from "./mlpModelDefaults";

/** Gated MLP: repeated blocks `act(Wg x) * (Wv x)` then final output projection. */
export type GatedMlpModelNodeData = {
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  depth: ListOr1<number>;
  width: ListOr1<number>;
  activation: ListOr1<MlpActivationId>;
  /** PyTorch RNG seed for weight initialization. */
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultGatedMlpModelData(): GatedMlpModelNodeData {
  return {
    inputDim: 10,
    outputDim: 1,
    depth: 2,
    width: 64,
    activation: "silu",
    seed: 0,
  };
}
