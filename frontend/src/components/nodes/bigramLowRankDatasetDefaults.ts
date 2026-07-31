import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export const BIGRAM_SPECTRUM_DECAY_IDS = ["power_law", "exponential"] as const;
export type BigramSpectrumDecayId = (typeof BIGRAM_SPECTRUM_DECAY_IDS)[number];

export const BIGRAM_SPECTRUM_DECAY_OPTIONS: { id: BigramSpectrumDecayId; label: string }[] = [
  { id: "power_law", label: "Power law $\\lambda_n = n^{-\\alpha}$" },
  { id: "exponential", label: "Exponential $\\lambda_n = e^{-\\alpha n}$" },
];

export type BigramLowRankDatasetNodeData = {
  vocabSize: ListOr1<number>;
  rank: ListOr1<number>;
  logitScale: ListOr1<number>;
  corruptRatio: ListOr1<number>;
  corruptScale: ListOr1<number>;
  decayType: ListOr1<BigramSpectrumDecayId>;
  alpha: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  initSeed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultBigramLowRankDatasetData(): BigramLowRankDatasetNodeData {
  return {
    vocabSize: 100,
    rank: 20,
    logitScale: 1.0,
    corruptRatio: 0.0,
    corruptScale: 5.0,
    decayType: "power_law",
    alpha: 0.0,
    trainSize: 1200,
    testSize: 300,
    seed: 0,
    initSeed: 0,
    samplingMode: "fixed",
  };
}
