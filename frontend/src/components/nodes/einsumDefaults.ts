import type { ListOr1 } from "./multiValueUtils";

export type EinsumNodeData = {
  equation: ListOr1<string>;
  ioMode?: "model" | "input-output";
  levelMode?: "high" | "low";
};

export function defaultEinsumNodeData(): EinsumNodeData {
  return {
    equation: "b h t d, b h s d -> b h t s",
    ioMode: "input-output",
    levelMode: "high",
  };
}
