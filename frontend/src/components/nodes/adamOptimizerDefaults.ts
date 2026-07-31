import type { ListOr1 } from "./multiValueUtils";

export type AdamOptimizerNodeData = {
  learningRate: ListOr1<number>;
  beta1: ListOr1<number>;
  beta2: ListOr1<number>;
  epsilon: ListOr1<number>;
  weightDecay: ListOr1<number>;
};

export function defaultAdamOptimizerData(): AdamOptimizerNodeData {
  return {
    learningRate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weightDecay: 0,
  };
}
