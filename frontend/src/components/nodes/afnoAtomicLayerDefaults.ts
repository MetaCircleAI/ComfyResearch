import type { ListOr1 } from "./multiValueUtils";

export type AfnoAtomicLayerNodeData = {
  contextFrames: ListOr1<number>;
  channels: ListOr1<number>;
  gridSize: ListOr1<number>;
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  patchSize: ListOr1<number>;
  embedDim: ListOr1<number>;
  numHeads: ListOr1<number>;
  ffRatio: ListOr1<number>;
  dropout: ListOr1<number>;
  numSpectralBlocks: ListOr1<number>;
  maxFrequencyModes: ListOr1<number>;
  spectralShrinkFactor: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultAfnoAtomicLayerData(): AfnoAtomicLayerNodeData {
  return {
    contextFrames: 4,
    channels: 1,
    gridSize: 16,
    inputDim: 1024,
    outputDim: 1024,
    patchSize: 4,
    embedDim: 64,
    numHeads: 4,
    ffRatio: 2,
    dropout: 0,
    numSpectralBlocks: 1,
    maxFrequencyModes: 4,
    spectralShrinkFactor: 1,
    seed: 0,
  };
}

