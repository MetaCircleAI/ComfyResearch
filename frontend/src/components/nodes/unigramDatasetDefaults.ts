import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode, OutputDistributionId } from "./linearDatasetDefaults";

export type UnigramDatasetNodeData = {
  vocabSize: ListOr1<number>;
  /** Same ids as memorization A: uniform / power-law / exponential over token ranks $1..V$. */
  outputDistribution: ListOr1<OutputDistributionId>;
  alpha: ListOr1<number>;
  contextLength: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export const UNIGRAM_OUTPUT_DISTRIBUTION_OPTIONS: { id: OutputDistributionId; label: string }[] = [
  { id: "uniform_class_probs", label: "Uniform classes ($1/V$)" },
  { id: "power_law_class_probs", label: "Power law ($\\propto n^{-\\alpha}$)" },
  { id: "exponential_class_probs", label: "Exponential ($\\propto e^{-\\alpha n}$)" },
];

export function defaultUnigramDatasetData(): UnigramDatasetNodeData {
  return {
    vocabSize: 100,
    outputDistribution: "power_law_class_probs",
    alpha: 1.0,
    contextLength: 1,
    trainSize: 800,
    testSize: 200,
    seed: 0,
    samplingMode: "fixed",
  };
}
