import type { ListOr1 } from "./multiValueUtils";
import type { MlpActivationId } from "./mlpModelDefaults";

export type ResidualLnMode = "pre_ln" | "post_ln";

export type ResidualLnModelNodeData = {
  dim: ListOr1<number>;
  depth: ListOr1<number>;
  alpha: ListOr1<number>;
  lnMode: ListOr1<ResidualLnMode>;
  activation: ListOr1<MlpActivationId>;
  /** PyTorch RNG seed for weight initialization. */
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultResidualLnModelData(): ResidualLnModelNodeData {
  return {
    dim: 256,
    depth: 100,
    alpha: 1,
    lnMode: "pre_ln",
    activation: "relu",
    seed: 0,
  };
}
