import type { ListOr1 } from "./multiValueUtils";

/** ``yes``: causal self-attention; ``no``: full-sequence (bidirectional) attention. */
export type AttentionOnlyCausalId = "yes" | "no";
export type AttentionOnlyYesNoId = "yes" | "no";

export type AttentionOnlyModelNodeData = {
  vocabSize: ListOr1<number>;
  /** Per-token activation width ``d`` (``[batch, L, d]`` in / out). */
  embedDim: ListOr1<number>;
  numHeads: ListOr1<number>;
  contextLength: ListOr1<number>;
  causalAttention: ListOr1<AttentionOnlyCausalId>;
  /** Causal depthwise conv after embedding in token CE bundle (0–2 = off). */
  localMixingKernel: ListOr1<number>;
  qkNorm: ListOr1<AttentionOnlyYesNoId>;
  attnTemperature: ListOr1<number>;
  attnLogitCap: ListOr1<number>;
  attnDropout: ListOr1<number>;
  /** PyTorch init seed for attention weights (training and weight export). */
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultAttentionOnlyModelData(): AttentionOnlyModelNodeData {
  return {
    vocabSize: 100,
    embedDim: 32,
    numHeads: 4,
    contextLength: 4,
    causalAttention: "yes",
    localMixingKernel: 0,
    qkNorm: "no",
    attnTemperature: 1,
    attnLogitCap: 0,
    attnDropout: 0,
    seed: 0,
  };
}
