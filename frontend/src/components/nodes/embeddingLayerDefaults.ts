import type { ListOr1 } from "./multiValueUtils";

export type EmbeddingLayerNodeData = {
  /** Vocabulary / index range size ``[0, numEmbeddings)``. */
  numEmbeddings: ListOr1<number>;
  /** Dimension of each embedding vector. */
  embeddingDim: ListOr1<number>;
  /** Last dimension of the incoming index tensor (e.g. 1 for shape ``(batch,)``). */
  numIndexColumns: ListOr1<number>;
  /** ``-1`` = no padding row; else row index treated as padding by ``nn.Embedding``. */
  paddingIdx: ListOr1<number>;
  /** 1 = ``scale_grad_by_freq`` in ``nn.Embedding``. */
  scaleGradByFreq: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultEmbeddingLayerData(): EmbeddingLayerNodeData {
  return {
    numEmbeddings: 4096,
    embeddingDim: 64,
    numIndexColumns: 1,
    paddingIdx: -1,
    scaleGradByFreq: 0,
    seed: 0,
  };
}
