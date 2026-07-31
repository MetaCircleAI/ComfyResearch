import type { ListOr1 } from "./multiValueUtils";

export type NumericHyenaModelNodeData = {
  contextLength: ListOr1<number>;
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  modelDim: ListOr1<number>;
  depth: ListOr1<number>;
  convKernel: ListOr1<number>;
  ffMult: ListOr1<number>;
  localMixingKernel: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultNumericHyenaModelData(): NumericHyenaModelNodeData {
  return {
    contextLength: 8,
    inputDim: 2,
    outputDim: 2,
    modelDim: 64,
    depth: 2,
    convKernel: 7,
    ffMult: 2,
    localMixingKernel: 0,
    seed: 0,
  };
}
