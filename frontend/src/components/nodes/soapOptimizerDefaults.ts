import type { ListOr1 } from "./multiValueUtils";

export type SoapOptimizerNodeData = {
  learningRate: ListOr1<number>;
  beta1: ListOr1<number>;
  beta2: ListOr1<number>;
  epsilon: ListOr1<number>;
  weightDecay: ListOr1<number>;
  preconditionFrequency: ListOr1<number>;
  maxPreconditionerDim: ListOr1<number>;
};

export function defaultSoapOptimizerData(): SoapOptimizerNodeData {
  return {
    learningRate: 0.0003,
    beta1: 0.9,
    beta2: 0.95,
    epsilon: 1e-8,
    weightDecay: 0,
    preconditionFrequency: 10,
    maxPreconditionerDim: 1024,
  };
}
