import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type CircularMotionDatasetNodeData = {
  vocabSize: ListOr1<number>;
  contextLength: ListOr1<number>;
  radiusMin: ListOr1<number>;
  radiusMax: ListOr1<number>;
  angularVelocity: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultCircularMotionDatasetData(): CircularMotionDatasetNodeData {
  return {
    vocabSize: 128,
    contextLength: 20,
    radiusMin: 0.15,
    radiusMax: 0.35,
    angularVelocity: 0.5,
    trainSize: 4000,
    testSize: 1000,
    seed: 0,
    samplingMode: "fixed",
  };
}
