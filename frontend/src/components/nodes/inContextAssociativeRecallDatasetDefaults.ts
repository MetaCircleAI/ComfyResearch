import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type InContextAssociativeRecallDatasetNodeData = {
  vocabSize: ListOr1<number>;
  numPairs: ListOr1<number>;
  inContextRepeat: ListOr1<number>;
  crossSampleRepeatProb: ListOr1<number>;
  repeatedTokenCount: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultInContextAssociativeRecallDatasetData(): InContextAssociativeRecallDatasetNodeData {
  return {
    vocabSize: 64,
    numPairs: 32,
    inContextRepeat: 1,
    crossSampleRepeatProb: 0.0,
    repeatedTokenCount: 2,
    trainSize: 10_000,
    testSize: 2_000,
    seed: 0,
    samplingMode: "fixed",
  };
}
