import type { ListOr1 } from "./multiValueUtils";

export type TransformerTokenCausalId = "yes" | "no";
export type TransformerTokenTieId = "yes" | "no";

/** FFN activations for the native Pre-LN encoder stack (explicit MHA + two-layer FFN; nanoGPT-style). */
export type TransformerTokenEncoderActivationId = "gelu" | "relu" | "silu";

export type TransformerEncoderBackendId = "pytorch" | "stable";

export const TRANSFORMER_ENCODER_ACTIVATION_OPTIONS: {
  id: TransformerTokenEncoderActivationId;
  label: string;
}[] = [
  { id: "gelu", label: "GELU" },
  { id: "relu", label: "ReLU" },
  { id: "silu", label: "SiLU" },
];

export const TRANSFORMER_ENCODER_BACKEND_OPTIONS: { id: TransformerEncoderBackendId; label: string }[] = [
  { id: "pytorch", label: "Pre-LN encoder (default)" },
  { id: "stable", label: "Pre-LN + attention stability knobs" },
];

export type TransformerTokenModelNodeData = {
  vocabSize: ListOr1<number>;
  contextLength: ListOr1<number>;
  modelDim: ListOr1<number>;
  numHeads: ListOr1<number>;
  numLayers: ListOr1<number>;
  ffDim: ListOr1<number>;
  /** FFN activation inside each encoder layer. */
  activation: ListOr1<TransformerTokenEncoderActivationId>;
  encoderBackend: ListOr1<TransformerEncoderBackendId>;
  encoderDropout: ListOr1<number>;
  spectralNormLinears: ListOr1<TransformerTokenTieId>;
  lmLogitScale: ListOr1<number>;
  stableQkNorm: ListOr1<TransformerTokenTieId>;
  stableAttnTemperature: ListOr1<number>;
  stableAttnLogitCap: ListOr1<number>;
  stableAttnDropout: ListOr1<number>;
  tieEmbeddingLmHead: ListOr1<TransformerTokenTieId>;
  causalAttention: ListOr1<TransformerTokenCausalId>;
  /** Causal depthwise conv along sequence (0/1/2 = off; odd kernel ≥3 enables Canon-lite mixing). */
  localMixingKernel: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultTransformerTokenModelData(): TransformerTokenModelNodeData {
  return {
    vocabSize: 100,
    contextLength: 4,
    modelDim: 32,
    numHeads: 1,
    numLayers: 1,
    ffDim: 64,
    activation: "gelu",
    encoderBackend: "pytorch",
    encoderDropout: 0,
    spectralNormLinears: "no",
    lmLogitScale: 1,
    stableQkNorm: "no",
    stableAttnTemperature: 1,
    stableAttnLogitCap: 0,
    stableAttnDropout: 0,
    tieEmbeddingLmHead: "yes",
    causalAttention: "yes",
    localMixingKernel: 0,
    seed: 0,
  };
}
