import type { ListOr1 } from "./multiValueUtils";

export type CausalMaskNodeData = {
  diagonalOffset: ListOr1<number>;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function defaultCausalMaskNodeData(): CausalMaskNodeData {
  return {
    diagonalOffset: 1,
    ioMode: "input-output",
    levelMode: "high",
  };
}
