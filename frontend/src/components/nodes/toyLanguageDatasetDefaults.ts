import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export const TOY_LANGUAGE_DATASET_KINDS = [
  "pcfg_dataset",
  "dyck_dataset",
  "ngram_language_dataset",
  "formal_language_suite_dataset",
  "scan_dataset",
  "cogs_dataset",
  "listops_dataset",
  "tinystories_dataset",
  "phi1_style_dataset",
  "biography_lm_dataset",
  "relation_tuple_dataset",
  "synthetic_playground_dataset",
  "multi_hop_fact_chain_dataset",
  "tinyshakespeare_lm_dataset",
] as const;

export type ToyLanguageDatasetKind = (typeof TOY_LANGUAGE_DATASET_KINDS)[number];

export type ToyLanguageDatasetNodeData = {
  vocabSize?: ListOr1<number>;
  vocabCap?: ListOr1<number>;
  contextLength?: ListOr1<number>;
  trainSize?: ListOr1<number>;
  testSize?: ListOr1<number>;
  seed?: ListOr1<number>;
  initSeed?: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
  /** PCFG */
  /** `binary_tree` (default): legacy random binary leaf stream. `cfg_sentence`: weighted CFG over words (see node info). */
  pcfgGenMode?: ListOr1<"binary_tree" | "cfg_sentence">;
  /** Builtin grammar when `pcfgGenMode` is `cfg_sentence` (e.g. `world_model`). */
  pcfgGrammarId?: ListOr1<string>;
  pcfgMaxDepth?: ListOr1<number>;
  pcfgTermProb?: ListOr1<number>;
  /** Dyck: bracket type count k (vocab size v is fixed to 2k on the canvas and in the engine). */
  numBracketTypes?: ListOr1<number>;
  /** Max unmatched opens on the stack in the balanced prefix; 0 = no cap. */
  maxNestingDepth?: ListOr1<number>;
  /** N-gram */
  orderN?: ListOr1<number>;
  dirichletAlpha?: ListOr1<number>;
  /** Formal suite */
  languageType?: ListOr1<string>;
  /** External / corpus */
  dataSource?: ListOr1<"synthetic" | "download">;
  cacheDir?: string;
  scanUrl?: string;
  tinyStoriesUrl?: string;
  downloadUrl?: string;
  tokenizerMode?: ListOr1<"char" | "word">;
  seqLen?: ListOr1<number>;
  stride?: ListOr1<number>;
  domainMix?: ListOr1<string> | string;
  /** Biography LM: template | shuffle_fields | noise_slots */
  biographyAugmentation?: ListOr1<string>;
  slotNoiseProb?: ListOr1<number>;
  /** Relation tuple: forward | inverse */
  relationMode?: ListOr1<string>;
  /** Multi-hop chain length (number of binary facts before query). */
  chainHops?: ListOr1<number>;
  /** Playground preset: depo | brevo | mano | capo | lano (+ aliases). */
  playgroundFamily?: ListOr1<string>;
  depoWindow?: ListOr1<number>;
  manoModulus?: ListOr1<number>;
  lanoNestingDepth?: ListOr1<number>;
  /** Tensor reader / preview: numeric ids (default) or human-readable word view. */
  inspectFormat?: ListOr1<"id" | "word">;
};

export function defaultToyLanguageDatasetData(kind: ToyLanguageDatasetKind): ToyLanguageDatasetNodeData {
  const base: ToyLanguageDatasetNodeData = {
    vocabSize: 32,
    contextLength: 16,
    trainSize: 800,
    testSize: 200,
    seed: 0,
    initSeed: 0,
    samplingMode: "fixed",
    dataSource: "synthetic",
    cacheDir: "",
    inspectFormat: "id",
  };
  if (kind === "pcfg_dataset") {
    return { ...base, pcfgGenMode: "binary_tree", pcfgGrammarId: "world_model", pcfgMaxDepth: 8, pcfgTermProb: 0.35 };
  }
  if (kind === "dyck_dataset") {
    return { ...base, numBracketTypes: 1, maxNestingDepth: 0, vocabSize: 2 };
  }
  if (kind === "ngram_language_dataset") {
    return { ...base, orderN: 3, dirichletAlpha: 1.0 };
  }
  if (kind === "formal_language_suite_dataset") {
    return { ...base, vocabSize: 8, languageType: "anbn" };
  }
  if (kind === "scan_dataset") {
    return { ...base, vocabSize: 64, contextLength: 24 };
  }
  if (kind === "cogs_dataset") {
    return { ...base, vocabSize: 128, contextLength: 32 };
  }
  if (kind === "listops_dataset") {
    return { ...base, vocabSize: 64, contextLength: 48 };
  }
  if (kind === "tinystories_dataset") {
    return {
      ...base,
      vocabCap: 256,
      vocabSize: 256,
      contextLength: 64,
      tokenizerMode: "char",
      seqLen: 512,
      stride: 64,
    };
  }
  if (kind === "phi1_style_dataset") {
    return {
      ...base,
      vocabCap: 256,
      vocabSize: 256,
      contextLength: 96,
      tokenizerMode: "char",
      seqLen: 600,
      stride: 96,
      domainMix: "mixed",
    };
  }
  if (kind === "biography_lm_dataset") {
    return {
      ...base,
      vocabSize: 64,
      contextLength: 32,
      biographyAugmentation: "template",
      slotNoiseProb: 0.0,
    };
  }
  if (kind === "relation_tuple_dataset") {
    return { ...base, vocabSize: 64, contextLength: 24, relationMode: "forward" };
  }
  if (kind === "synthetic_playground_dataset") {
    return {
      ...base,
      vocabSize: 64,
      contextLength: 32,
      playgroundFamily: "depo",
      depoWindow: 4,
      manoModulus: 17,
      lanoNestingDepth: 4,
    };
  }
  if (kind === "multi_hop_fact_chain_dataset") {
    return { ...base, vocabSize: 96, contextLength: 48, chainHops: 3 };
  }
  if (kind === "tinyshakespeare_lm_dataset") {
    return {
      vocabSize: 256,
      contextLength: 32,
      trainSize: 4000,
      testSize: 0,
      seed: 0,
      initSeed: 0,
      stride: 1,
    };
  }
  return base;
}
