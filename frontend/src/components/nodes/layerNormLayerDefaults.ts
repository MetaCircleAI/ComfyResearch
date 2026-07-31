import type { ListOr1 } from "./multiValueUtils";

export type LayerNormLayerNodeData = {
  normalizedShape: ListOr1<number>;
  eps: ListOr1<number>;
  /** 1 = learnable affine, 0 = ``elementwise_affine=False``. */
  elementwiseAffine: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultLayerNormLayerData(): LayerNormLayerNodeData {
  return {
    normalizedShape: 64,
    eps: 1e-5,
    elementwiseAffine: 1,
  };
}
