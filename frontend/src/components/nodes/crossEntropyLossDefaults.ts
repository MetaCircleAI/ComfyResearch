import type { ListOr1 } from "./multiValueUtils";
import type { MseLossMaskModeId } from "./mseLossDefaults";

export type CrossEntropyLossNodeData = {
  lossScale: ListOr1<number>;
  /** Label smoothing for CE (0 = plain cross-entropy). */
  labelSmoothing: ListOr1<number>;
  /** Group flat logits [batch, T*V] into T slots of V classes each (same class label y for every slot). */
  lossMaskContextLength: ListOr1<number>;
  lossMaskMode: ListOr1<MseLossMaskModeId>;
  /** When mode is custom: T comma-separated slot weights (must match lossMaskContextLength when T>1). */
  lossMaskCustom: string;
};

export function defaultCrossEntropyLossData(): CrossEntropyLossNodeData {
  return {
    lossScale: 1,
    labelSmoothing: 0,
    lossMaskContextLength: 1,
    lossMaskMode: "all",
    lossMaskCustom: "",
  };
}
