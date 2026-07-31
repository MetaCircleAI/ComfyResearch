export type CyclicScheduleModeId = "discrete_epoch" | "triangular_step";

export type CyclicLrScheduleNodeData = {
  lrMin: number;
  lrMax: number;
  cycleLengthEpochs: number;
  refBatchSize: number;
  cycleLengthSteps: number;
  scheduleMode: CyclicScheduleModeId;
};

export function defaultCyclicLrScheduleData(): CyclicLrScheduleNodeData {
  return {
    lrMin: 0.001,
    lrMax: 0.005,
    cycleLengthEpochs: 10,
    refBatchSize: 128,
    cycleLengthSteps: 0,
    scheduleMode: "discrete_epoch",
  };
}
