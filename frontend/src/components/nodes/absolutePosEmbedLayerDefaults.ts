import type { ListOr1 } from "./multiValueUtils";

export type AbsolutePosEmbedLayerNodeData = {
  maxSeqLen: ListOr1<number>;
  embeddingDim: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultAbsolutePosEmbedLayerData(): AbsolutePosEmbedLayerNodeData {
  return {
    maxSeqLen: 512,
    embeddingDim: 64,
    seed: 0,
  };
}
