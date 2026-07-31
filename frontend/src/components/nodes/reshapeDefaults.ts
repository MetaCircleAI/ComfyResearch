import type { ListOr1 } from "./multiValueUtils";

export type ReshapeNodeData = {
  reshapeRule: ListOr1<string>;
  shapeHint: ListOr1<string>;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function defaultReshapeNodeData(): ReshapeNodeData {
  return {
    reshapeRule: "b t d -> b t d",
    shapeHint: "split heads",
    ioMode: "input-output",
    levelMode: "high",
  };
}
