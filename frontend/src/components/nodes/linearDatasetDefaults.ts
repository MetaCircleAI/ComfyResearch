/** Synthetic data: y = W x + σ ε with W ~ Gaussian scaled by 1/√(input dim), ε ~ Normal(0,1). */

import type { ListOr1 } from "./multiValueUtils";

export type InputDistributionId = "standard_normal" | "uniform_neg1_1" | "uniform_0_1";

export type OutputDistributionId =
  | "additive_gaussian"
  | "deterministic"
  | "uniform_class_probs"
  | "power_law_class_probs"
  | "exponential_class_probs";

/** One draw at prepare time vs. resampled train tensors each optimizer step (trainer reads wired dataset). */
export type DatasetSamplingMode = "fixed" | "streaming";

export const DATASET_SAMPLING_MODE_OPTIONS: { id: DatasetSamplingMode; label: string }[] = [
  { id: "fixed", label: "Fixed (sample once)" },
  { id: "streaming", label: "Streaming (new train data each step)" },
];

export type LinearDatasetNodeData = {
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  /** Memorization B unified class vocabulary size (shared by input and output labels). */
  vocabSize?: ListOr1<number>;
  inputDistribution: ListOr1<InputDistributionId>;
  outputDistribution: ListOr1<OutputDistributionId>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  noiseLevel: ListOr1<number>;
  /** Class-prior sharpness for memorization distributions (power-law / exponential). */
  alpha: ListOr1<number>;
  seed: ListOr1<number>;
  /** How the trainer uses this dataset: one fixed draw vs. resample each train step. */
  samplingMode?: DatasetSamplingMode;
  /** Python ``def`` / spec identifier (e.g. LinearDataset_simplified). */
  specCodeName?: string;
  /** Order of fields / parameters (camelCase keys), from last spec parse. */
  paramOrder?: string[];
  /** Parsed unknown parameters from edited code (not used by trainer). */
  extras?: Record<string, string | number | boolean>;
};

export const INPUT_DISTRIBUTION_OPTIONS: { id: InputDistributionId; label: string }[] = [
  { id: "standard_normal", label: "Standard normal" },
  { id: "uniform_neg1_1", label: "Uniform [−1, 1]" },
  { id: "uniform_0_1", label: "Uniform [0, 1]" },
];

export const OUTPUT_DISTRIBUTION_OPTIONS: { id: OutputDistributionId; label: string }[] = [
  { id: "additive_gaussian", label: "Gaussian noise (ε ~ 𝒩(0,1))" },
  { id: "deterministic", label: "Deterministic (no noise)" },
];

export const MEMORIZATION_OUTPUT_DISTRIBUTION_OPTIONS: { id: OutputDistributionId; label: string }[] = [
  { id: "uniform_class_probs", label: "Uniform classes ($1/d_{\\mathrm{out}}$)" },
  { id: "power_law_class_probs", label: "Power law ($\\propto n^{-\\alpha}$)" },
  { id: "exponential_class_probs", label: "Exponential ($\\propto e^{-\\alpha n}$)" },
];

export const defaultLinearDatasetData = (): LinearDatasetNodeData => ({
  inputDim: 10,
  outputDim: 1,
  inputDistribution: "standard_normal",
  outputDistribution: "additive_gaussian",
  trainSize: 800,
  testSize: 200,
  noiseLevel: 0.25,
  alpha: 1.0,
  seed: 0,
  samplingMode: "fixed",
});

export const defaultRandomNoiseDatasetData = (): LinearDatasetNodeData => ({
  inputDim: 10,
  outputDim: 1,
  inputDistribution: "standard_normal",
  outputDistribution: "deterministic",
  trainSize: 800,
  testSize: 200,
  noiseLevel: 0,
  alpha: 1.0,
  seed: 0,
  samplingMode: "fixed",
});

/** Python ``def`` name for the memorization A preset (must not be reused for ``linear_dataset`` nodes). */
export const MEMORIZATION_A_DATASET_SPEC_NAME = "Memorization_A_Dataset";

/** Python ``def`` name for memorization B (categorical input + labels); must not collide with A or linear. */
export const MEMORIZATION_B_DATASET_SPEC_NAME = "Memorization_B_Dataset";

/**
 * Blog preset: "Memory 1 -- How much do linear layers memorize?"
 * https://kindxiaoming.github.io/blog/2026/memory-1/
 */
export const defaultMemorizationADatasetData = (): LinearDatasetNodeData => ({
  inputDim: 40,
  outputDim: 40,
  inputDistribution: "standard_normal",
  outputDistribution: "uniform_class_probs",
  trainSize: 160,
  testSize: 0,
  noiseLevel: 0,
  alpha: 1.0,
  seed: 0,
  samplingMode: "fixed",
  specCodeName: MEMORIZATION_A_DATASET_SPEC_NAME,
});

/** Memorization B: both endpoints are class ids; the trainer uses one-hot float inputs of width ``inputDim`` (input vocabulary). */
export const defaultMemorizationBDatasetData = (): LinearDatasetNodeData => ({
  inputDim: 40,
  outputDim: 40,
  vocabSize: 40,
  inputDistribution: "standard_normal",
  outputDistribution: "uniform_class_probs",
  trainSize: 160,
  testSize: 0,
  noiseLevel: 0,
  alpha: 1.0,
  seed: 0,
  samplingMode: "fixed",
  specCodeName: MEMORIZATION_B_DATASET_SPEC_NAME,
});
