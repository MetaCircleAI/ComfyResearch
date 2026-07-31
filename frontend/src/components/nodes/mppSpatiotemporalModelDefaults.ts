import type { ListOr1 } from "./multiValueUtils";

export type MppSpatiotemporalModelNodeData = {
  contextFrames: ListOr1<number>;
  channels: ListOr1<number>;
  gridSize: ListOr1<number>;
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  patchSize: ListOr1<number>;
  embedDim: ListOr1<number>;
  depth: ListOr1<number>;
  numHeads: ListOr1<number>;
  ffRatio: ListOr1<number>;
  dropout: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

/** Defaults match T=4, C=1, G=16 ⇒ flat 1024; patchSize divides gridSize. */
export function defaultMppSpatiotemporalModelData(): MppSpatiotemporalModelNodeData {
  return {
    contextFrames: 4,
    channels: 1,
    gridSize: 16,
    inputDim: 1024,
    outputDim: 1024,
    patchSize: 4,
    embedDim: 128,
    depth: 4,
    numHeads: 4,
    ffRatio: 4,
    dropout: 0,
    seed: 0,
  };
}
