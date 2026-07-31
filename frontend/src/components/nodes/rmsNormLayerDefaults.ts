import type { ListOr1 } from "./multiValueUtils";

export type RmsNormLayerNodeData = {
  normalizedShape: ListOr1<number>;
  eps: ListOr1<number>;
  /** 1 = learnable scale, 0 = no affine parameter. */
  elementwiseAffine: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultRmsNormLayerData(): RmsNormLayerNodeData {
  return {
    normalizedShape: 64,
    eps: 1e-6,
    elementwiseAffine: 1,
  };
}
