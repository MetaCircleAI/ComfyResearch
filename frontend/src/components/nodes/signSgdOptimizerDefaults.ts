import type { ListOr1 } from "./multiValueUtils";

export type SignSgdOptimizerNodeData = {
  learningRate: ListOr1<number>;
  weightDecay: ListOr1<number>;
};

export function defaultSignSgdOptimizerData(): SignSgdOptimizerNodeData {
  return {
    learningRate: 0.001,
    weightDecay: 0,
  };
}
