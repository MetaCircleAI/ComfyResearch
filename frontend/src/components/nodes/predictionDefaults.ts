export type PredictionTensorPayload = {
  shape: number[];
  values: number[];
  truncated?: boolean;
};

export type PredictionRunSplit = "both" | "train" | "test";

export type PredictionNodeData = {
  instanceTitle?: string;
  split: PredictionRunSplit;
  trainPrediction: PredictionTensorPayload | null;
  testPrediction: PredictionTensorPayload | null;
  trainerTask?: string | null;
  lastError?: string | null;
};

export function defaultPredictionNodeData(partial?: Partial<PredictionNodeData>): PredictionNodeData {
  return {
    split: "both",
    trainPrediction: null,
    testPrediction: null,
    trainerTask: null,
    lastError: null,
    ...partial,
  };
}

