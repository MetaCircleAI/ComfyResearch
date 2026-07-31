import type { ListOr1 } from "./multiValueUtils";

export type WeightRegNodeData = {
  lossScale: ListOr1<number>;
};

export function defaultWeightRegData(): WeightRegNodeData {
  return { lossScale: 1 };
}
