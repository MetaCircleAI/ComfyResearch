import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

/** Random tokens in ``0 .. vocabSize-1``; target is the last position (classification). */

export type TokenPredictionDatasetNodeData = {
  /** Retrieval rule: by fixed position index or by nearest content to the last token. */
  retrievalMode?: ListOr1<"position" | "content">;
  vocabSize: ListOr1<number>;
  contextLength: ListOr1<number>;
  /** Python-style index for retrieval target. Default -1 (last token). */
  whichToken: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultTokenPredictionDatasetData(): TokenPredictionDatasetNodeData {
  return {
    retrievalMode: "position",
    vocabSize: 4,
    contextLength: 4,
    whichToken: -1,
    trainSize: 800,
    testSize: 200,
    seed: 0,
    samplingMode: "fixed",
  };
}
