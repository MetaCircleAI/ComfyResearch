import type { ListOr1 } from "./multiValueUtils";

/** ``yes``: causal self-attention (position i sees tokens ≤ i). ``no``: full bidirectional encoder. */
export type NumericTransformerCausalId = "yes" | "no";

export type NumericTransformerModelNodeData = {
  /** Sequence length T (number of context positions). */
  contextLength: ListOr1<number>;
  /** Per-position input width D_in (flattened training uses T * D_in). */
  inputDim: ListOr1<number>;
  /** Per-position output width D_out. */
  outputDim: ListOr1<number>;
  modelDim: ListOr1<number>;
  numHeads: ListOr1<number>;
  numLayers: ListOr1<number>;
  ffDim: ListOr1<number>;
  activation: ListOr1<"gelu" | "relu" | "silu">;
  encoderBackend: ListOr1<"pytorch" | "stable">;
  encoderDropout: ListOr1<number>;
  spectralNormLinears: ListOr1<NumericTransformerCausalId>;
  stableQkNorm: ListOr1<NumericTransformerCausalId>;
  stableAttnTemperature: ListOr1<number>;
  stableAttnLogitCap: ListOr1<number>;
  stableAttnDropout: ListOr1<number>;
  causalAttention: ListOr1<NumericTransformerCausalId>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultNumericTransformerModelData(): NumericTransformerModelNodeData {
  return {
    contextLength: 2,
    inputDim: 1,
    outputDim: 1,
    modelDim: 32,
    numHeads: 1,
    numLayers: 1,
    ffDim: 64,
    activation: "gelu",
    encoderBackend: "pytorch",
    encoderDropout: 0,
    spectralNormLinears: "no",
    stableQkNorm: "no",
    stableAttnTemperature: 1,
    stableAttnLogitCap: 0,
    stableAttnDropout: 0,
    causalAttention: "yes",
    seed: 0,
  };
}
