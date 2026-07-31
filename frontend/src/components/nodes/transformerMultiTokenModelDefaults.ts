import type { ListOr1 } from "./multiValueUtils";

export type TransformerMultiTokenCausalId = "yes" | "no";
export type TransformerMultiTokenTieId = "yes" | "no";

export type TransformerMultiTokenModelNodeData = {
  vocabSize: ListOr1<number>;
  contextLength: ListOr1<number>;
  tokensPerPosition: ListOr1<number>;
  modelDim: ListOr1<number>;
  numHeads: ListOr1<number>;
  numLayers: ListOr1<number>;
  ffDim: ListOr1<number>;
  encoderBackend: ListOr1<"pytorch" | "stable">;
  encoderDropout: ListOr1<number>;
  spectralNormLinears: ListOr1<TransformerMultiTokenTieId>;
  lmLogitScale: ListOr1<number>;
  stableQkNorm: ListOr1<TransformerMultiTokenTieId>;
  stableAttnTemperature: ListOr1<number>;
  stableAttnLogitCap: ListOr1<number>;
  stableAttnDropout: ListOr1<number>;
  tieEmbeddingLmHead: ListOr1<TransformerMultiTokenTieId>;
  causalAttention: ListOr1<TransformerMultiTokenCausalId>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultTransformerMultiTokenModelData(): TransformerMultiTokenModelNodeData {
  return {
    vocabSize: 100,
    contextLength: 4,
    tokensPerPosition: 2,
    modelDim: 32,
    numHeads: 1,
    numLayers: 1,
    ffDim: 64,
    encoderBackend: "pytorch",
    encoderDropout: 0,
    spectralNormLinears: "no",
    lmLogitScale: 1,
    stableQkNorm: "no",
    stableAttnTemperature: 1,
    stableAttnLogitCap: 0,
    stableAttnDropout: 0,
    tieEmbeddingLmHead: "no",
    causalAttention: "yes",
    seed: 0,
  };
}
