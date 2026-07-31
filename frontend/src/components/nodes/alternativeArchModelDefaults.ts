import type { ListOr1 } from "./multiValueUtils";

export type ArchLmKind =
  | "linear_attention_model"
  | "diagonal_ssm_token_model"
  | "rwkv_time_mix_token_model"
  | "hyena_like_conv_model"
  | "slot_attention_token_model";

export type CausalAttnId = "yes" | "no";

export type AlternativeArchTokenLmNodeData = {
  vocabSize: ListOr1<number>;
  embedDim: ListOr1<number>;
  contextLength: ListOr1<number>;
  seed: ListOr1<number>;
  localMixingKernel: ListOr1<number>;
  numHeads?: ListOr1<number>;
  causalAttention?: ListOr1<CausalAttnId>;
  numLayers?: ListOr1<number>;
  depth?: ListOr1<number>;
  convKernel?: ListOr1<number>;
  ffMult?: ListOr1<number>;
  numSlots?: ListOr1<number>;
  slotIters?: ListOr1<number>;
};

export function defaultAlternativeArchTokenLmData(kind: ArchLmKind): AlternativeArchTokenLmNodeData {
  const base: AlternativeArchTokenLmNodeData = {
    vocabSize: 100,
    embedDim: 32,
    contextLength: 8,
    seed: 0,
    localMixingKernel: 0,
  };
  switch (kind) {
    case "linear_attention_model":
      return { ...base, numHeads: 4, causalAttention: "yes" };
    case "diagonal_ssm_token_model":
      return { ...base, numLayers: 2 };
    case "rwkv_time_mix_token_model":
      return { ...base, depth: 2 };
    case "hyena_like_conv_model":
      return { ...base, depth: 2, convKernel: 7, ffMult: 2 };
    case "slot_attention_token_model":
      return { ...base, numSlots: 4, slotIters: 3 };
    default:
      return base;
  }
}
