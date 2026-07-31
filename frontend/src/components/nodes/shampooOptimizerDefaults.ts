import type { ListOr1 } from "./multiValueUtils";

export type ShampooOptimizerNodeData = {
  learningRate: ListOr1<number>;
  momentum: ListOr1<number>;
  epsilon: ListOr1<number>;
  weightDecay: ListOr1<number>;
  preconditionFrequency: ListOr1<number>;
  maxPreconditionerDim: ListOr1<number>;
};

export function defaultShampooOptimizerData(): ShampooOptimizerNodeData {
  return {
    learningRate: 0.01,
    momentum: 0,
    epsilon: 1e-8,
    weightDecay: 0,
    preconditionFrequency: 10,
    maxPreconditionerDim: 1024,
  };
}
