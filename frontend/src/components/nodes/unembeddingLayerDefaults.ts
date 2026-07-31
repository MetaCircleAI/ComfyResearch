import type { ListOr1 } from "./multiValueUtils";

/** LM-head style projection: ``(…, inFeatures)`` → ``(…, outFeatures)`` (typically logits over vocab). */
export type UnembeddingLayerNodeData = {
  inFeatures: ListOr1<number>;
  outFeatures: ListOr1<number>;
  bias: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultUnembeddingLayerData(): UnembeddingLayerNodeData {
  return {
    inFeatures: 64,
    outFeatures: 4096,
    bias: 1,
    seed: 0,
  };
}
