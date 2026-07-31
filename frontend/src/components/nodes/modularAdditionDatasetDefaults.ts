import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type ModularAdditionDatasetNodeData = {
  modulus: ListOr1<number>;
  trainFraction: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultModularAdditionDatasetData(): ModularAdditionDatasetNodeData {
  return {
    modulus: 59,
    trainFraction: 0.3,
    seed: 0,
    samplingMode: "fixed",
  };
}
