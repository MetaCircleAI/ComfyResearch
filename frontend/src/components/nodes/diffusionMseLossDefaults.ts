import type { ListOr1 } from "./multiValueUtils";

export type DiffusionMseLossNodeData = {
  lossScale: ListOr1<number>;
};

export function defaultDiffusionMseLossData(): DiffusionMseLossNodeData {
  return { lossScale: 1 };
}
