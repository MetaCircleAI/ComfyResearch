export type MupLrScheduleNodeData = {
  mupEmbedLrMult: number;
  mupHiddenLrMult: number;
  mupOutputLrMult: number;
};

export function defaultMupLrScheduleData(): MupLrScheduleNodeData {
  return {
    mupEmbedLrMult: 1,
    mupHiddenLrMult: 1,
    mupOutputLrMult: 1,
  };
}
