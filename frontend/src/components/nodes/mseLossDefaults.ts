import type { ListOr1 } from "./multiValueUtils";

/** How to weight squared errors along the output context axis (see lossMaskContextLength). */
export type MseLossMaskModeId = "all" | "last_context" | "custom";

export type MseLossNodeData = {
  /** Multiplies MSE (training + logs); default 1. */
  lossScale: ListOr1<number>;
  /** Group flat output [batch, T*D] into T context slots of D features each (time-major). Use 1 for per-coordinate masks. */
  lossMaskContextLength: ListOr1<number>;
  lossMaskMode: ListOr1<MseLossMaskModeId>;
  /** When mode is custom: T comma weights if lossMaskContextLength=T>1, else one weight per flat output dim. */
  lossMaskCustom: string;
};

export function defaultMseLossData(): MseLossNodeData {
  return { lossScale: 1, lossMaskContextLength: 1, lossMaskMode: "all", lossMaskCustom: "" };
}
