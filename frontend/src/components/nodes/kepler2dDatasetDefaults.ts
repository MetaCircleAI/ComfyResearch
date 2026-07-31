import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode, OutputDistributionId } from "./linearDatasetDefaults";

export type Kepler2dDatasetNodeData = {
  /** Number of input positions x_0...x_{T-1}; target is x_1...x_T. */
  contextLength: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  semiMajorAxisMin: ListOr1<number>;
  semiMajorAxisMax: ListOr1<number>;
  eccentricityMin: ListOr1<number>;
  eccentricityMax: ListOr1<number>;
  meanMotion: ListOr1<number>;
  outputDistribution: ListOr1<OutputDistributionId>;
  noiseLevel: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultKepler2dDatasetData(): Kepler2dDatasetNodeData {
  return {
    contextLength: 8,
    trainSize: 1600,
    testSize: 400,
    semiMajorAxisMin: 0.7,
    semiMajorAxisMax: 1.3,
    eccentricityMin: 0.0,
    eccentricityMax: 0.55,
    meanMotion: 0.4,
    outputDistribution: "deterministic",
    noiseLevel: 0,
    seed: 0,
    samplingMode: "fixed",
  };
}
