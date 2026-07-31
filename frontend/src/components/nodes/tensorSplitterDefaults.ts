import type { ListOr1 } from "./multiValueUtils";

export type TensorSplitterNodeData = {
  splitDimension: ListOr1<number>;
  numParts: ListOr1<number>;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function defaultTensorSplitterNodeData(): TensorSplitterNodeData {
  return {
    splitDimension: -1,
    numParts: 3,
    ioMode: "input-output",
    levelMode: "high",
  };
}
