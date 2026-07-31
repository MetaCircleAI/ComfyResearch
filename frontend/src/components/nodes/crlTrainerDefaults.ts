export type CrlTrainerNodeData = {
  trainingSteps: number;
  logFrequency: number;
  computeDevice: string;
  /** Replay / SGD batch (transition windows). */
  batchSize: number;
  unrollLength: number;
  sgdStepsPerTrainStep: number;
  gamma: number;
  logsumexpPenaltyCoeff: number;
  entropyParam: number;
  disableEntropy: boolean;
  maxReplayChunks: number;
  seed: number;
  lossHistory?: number[];
  testLossHistory?: number[];
  stepTicks?: number[];
  observableMetricHistories?: Record<string, number[]>;
  memoryCheckpoint_b64?: string;
};

export function defaultCrlTrainerData(): CrlTrainerNodeData {
  return {
    trainingSteps: 40,
    logFrequency: 5,
    computeDevice: "cpu",
    batchSize: 32,
    unrollLength: 24,
    sgdStepsPerTrainStep: 4,
    gamma: 0.99,
    logsumexpPenaltyCoeff: 0.1,
    entropyParam: 0.5,
    disableEntropy: false,
    maxReplayChunks: 200,
    seed: 0,
  };
}
