import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type CircleRandomWalkDatasetNodeData = {
  vocabSize: ListOr1<number>;
  contextLength: ListOr1<number>;
  rightStepProb: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultCircleRandomWalkDatasetData(): CircleRandomWalkDatasetNodeData {
  return {
    vocabSize: 10,
    contextLength: 1,
    rightStepProb: 0.5,
    trainSize: 800,
    testSize: 200,
    seed: 0,
    samplingMode: "fixed",
  };
}
