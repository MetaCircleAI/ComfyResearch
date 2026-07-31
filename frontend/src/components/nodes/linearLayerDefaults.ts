import type { ListOr1 } from "./multiValueUtils";

export type LinearLayerNodeData = {
  inFeatures: ListOr1<number>;
  outFeatures: ListOr1<number>;
  /** 1 = with bias, 0 = no bias (matches PyTorch ``bias`` flag). */
  bias: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultLinearLayerData(): LinearLayerNodeData {
  return {
    inFeatures: 10,
    outFeatures: 10,
    bias: 1,
    seed: 0,
  };
}
