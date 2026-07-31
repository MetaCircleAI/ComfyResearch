import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode, InputDistributionId, OutputDistributionId } from "./linearDatasetDefaults";

export type UniformLinearMotionDatasetNodeData = {
  /** Number of input positions along time: x_0 … x_{T-1}; targets are x_1 … x_T with x_i = x0 + velocity·i. */
  contextLength: ListOr1<number>;
  positionDim: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  positionDistribution: ListOr1<InputDistributionId>;
  /** Per-step velocity v is drawn from this distribution, then scaled by ``velocityScale``. */
  velocityDistribution: ListOr1<InputDistributionId>;
  /** Multiplier applied to each component of the drawn velocity (independent of the distribution’s native width). */
  velocityScale: ListOr1<number>;
  /**
   * Legacy key for the velocity draw; merged into ``velocityDistribution`` when loading older graphs.
   * Not emitted in defaults or new spec code.
   */
  x1Distribution?: ListOr1<InputDistributionId>;
  outputDistribution: ListOr1<OutputDistributionId>;
  noiseLevel: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultUniformLinearMotionDatasetData(): UniformLinearMotionDatasetNodeData {
  return {
    contextLength: 2,
    positionDim: 1,
    trainSize: 800,
    testSize: 200,
    positionDistribution: "standard_normal",
    velocityDistribution: "standard_normal",
    velocityScale: 1,
    outputDistribution: "deterministic",
    noiseLevel: 0,
    seed: 0,
    samplingMode: "fixed",
  };
}
