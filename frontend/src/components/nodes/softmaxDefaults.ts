import type { ListOr1 } from "./multiValueUtils";

export type SoftmaxNodeData = {
  dimension: ListOr1<number>;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function defaultSoftmaxNodeData(): SoftmaxNodeData {
  return {
    dimension: -1,
    ioMode: "input-output",
    levelMode: "high",
  };
}
