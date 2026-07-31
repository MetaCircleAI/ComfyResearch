export type LrScheduleKindId = "constant" | "cosine" | "stable_stable_decay" | "exponential_epoch";

export type LrScheduleNodeData = {
  lrWarmupSteps: number;
  lrSchedule: LrScheduleKindId;
  /** Cosine floor as a fraction of each param-group base LR (after warmup). */
  cosineLrMinFraction: number;
  exponentialDecayFactor: number;
  exponentialDecayEpochs: number;
};

export function defaultLrScheduleData(): LrScheduleNodeData {
  return {
    lrWarmupSteps: 0,
    lrSchedule: "constant",
    cosineLrMinFraction: 0,
    exponentialDecayFactor: 0.95,
    exponentialDecayEpochs: 1,
  };
}
