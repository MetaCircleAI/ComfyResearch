import type { ListOr1 } from "./multiValueUtils";

export type DiffusionScoreModelNodeData = {
  inputDim: ListOr1<number>;
  hiddenDim: ListOr1<number>;
  depth: ListOr1<number>;
  timeEmbedDim: ListOr1<number>;
  diffusionTimesteps: ListOr1<number>;
  seed: ListOr1<number>;
};

export function defaultDiffusionScoreModelData(): DiffusionScoreModelNodeData {
  return {
    inputDim: 8,
    hiddenDim: 128,
    depth: 3,
    timeEmbedDim: 64,
    diffusionTimesteps: 100,
    seed: 0,
  };
}
