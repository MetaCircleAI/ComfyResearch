import type { ListOr1 } from "./multiValueUtils";

export type SgdOptimizerNodeData = {
  learningRate: ListOr1<number>;
  momentum: ListOr1<number>;
  weightDecay: ListOr1<number>;
};

export function defaultSgdOptimizerData(): SgdOptimizerNodeData {
  return {
    learningRate: 0.01,
    momentum: 0,
    weightDecay: 0,
  };
}
