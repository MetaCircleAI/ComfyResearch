export type CyclicScheduleModeId = "discrete_epoch" | "triangular_step";

export type CyclicBatchScheduleNodeData = {
  batchMin: number;
  batchMax: number;
  cycleLengthEpochs: number;
  refBatchSize: number;
  cycleLengthSteps: number;
  scheduleMode: CyclicScheduleModeId;
};

export function defaultCyclicBatchScheduleData(): CyclicBatchScheduleNodeData {
  return {
    batchMin: 128,
    batchMax: 640,
    cycleLengthEpochs: 10,
    refBatchSize: 128,
    cycleLengthSteps: 0,
    scheduleMode: "discrete_epoch",
  };
}
