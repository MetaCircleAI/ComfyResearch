import type { ListOr1 } from "./multiValueUtils";

export type RotaryEmbedLayerNodeData = {
  /** Must be even; last dimension of activations RoPE is applied to. */
  rotaryDim: ListOr1<number>;
  /** Inverse-frequency base (Llama-style ``theta``). */
  thetaBase: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultRotaryEmbedLayerData(): RotaryEmbedLayerNodeData {
  return {
    rotaryDim: 64,
    thetaBase: 10000,
    seed: 0,
  };
}
