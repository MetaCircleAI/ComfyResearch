import type { ListOr1 } from "./multiValueUtils";

export type LocalMixingLayerNodeData = {
  /** Channel width ``C`` for ``[..., T, C]`` (must match upstream last dim). */
  modelDim: ListOr1<number>;
  /** Odd causal depthwise kernel size (values `< 3` are clamped to `3` on the server). */
  kernelSize: ListOr1<number>;
  seed?: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultLocalMixingLayerData(): LocalMixingLayerNodeData {
  return {
    modelDim: 64,
    kernelSize: 5,
    seed: 0,
  };
}
