import type { ListOr1 } from "./multiValueUtils";
import type { MlpActivationId } from "./mlpModelDefaults";

export type ActivationLayerNodeData = {
  activation: ListOr1<MlpActivationId>;
  /** LeakyReLU only: PyTorch ``negative_slope`` in [-1, 1] (0 → ReLU, 1 → linear, -1 → abs on the negative side). */
  leakyP?: number;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultActivationLayerData(): ActivationLayerNodeData {
  return {
    activation: "relu",
    leakyP: 0,
  };
}
