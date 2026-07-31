import type { MlpActivationId } from "./mlpModelDefaults";
import type { ListOr1 } from "./multiValueUtils";

export type MlpTokenModelNodeData = {
  vocabSize: ListOr1<number>;
  embedDim: ListOr1<number>;
  tokensPerInput: ListOr1<number>;
  depth: ListOr1<number>;
  width: ListOr1<number>;
  numExperts?: ListOr1<number>;
  activation: ListOr1<MlpActivationId>;
  tieWeights: "yes" | "no";
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultMlpTokenModelData(): MlpTokenModelNodeData {
  return {
    vocabSize: 100,
    embedDim: 64,
    tokensPerInput: 1,
    depth: 2,
    width: 64,
    numExperts: 4,
    activation: "relu",
    tieWeights: "yes",
    seed: 0,
  };
}
