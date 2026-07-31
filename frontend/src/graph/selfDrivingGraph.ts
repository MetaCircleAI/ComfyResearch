import type { Edge, Node } from "@xyflow/react";
import { appendResearchNode } from "./nodeInstanceTitle";
import { ensureTrainerAutoVizes } from "./trainerAutoVizSpawn";
import { defaultAdamOptimizerData } from "../components/nodes/adamOptimizerDefaults";
import { defaultAdamWOptimizerData } from "../components/nodes/adamWOptimizerDefaults";
import { defaultSgdOptimizerData } from "../components/nodes/sgdOptimizerDefaults";
import { defaultSignSgdOptimizerData } from "../components/nodes/signSgdOptimizerDefaults";
import { defaultMuonOptimizerData } from "../components/nodes/muonOptimizerDefaults";
import { defaultShampooOptimizerData } from "../components/nodes/shampooOptimizerDefaults";
import { defaultSoapOptimizerData } from "../components/nodes/soapOptimizerDefaults";
import {
  defaultLinearDatasetData,
  defaultMemorizationADatasetData,
} from "../components/nodes/linearDatasetDefaults";
import { defaultSymbolicFuncDatasetData } from "../components/nodes/symbolicFuncDatasetDefaults";
import { defaultUniformLinearMotionDatasetData } from "../components/nodes/uniformLinearMotionDatasetDefaults";
import { defaultTokenPredictionDatasetData } from "../components/nodes/tokenPredictionDatasetDefaults";
import { defaultModularAdditionDatasetData } from "../components/nodes/modularAdditionDatasetDefaults";
import { defaultUnigramDatasetData } from "../components/nodes/unigramDatasetDefaults";
import { defaultCircleRandomWalkDatasetData } from "../components/nodes/circleRandomWalkDatasetDefaults";
import { defaultCircularMotionDatasetData } from "../components/nodes/circularMotionDatasetDefaults";
import { defaultBigramLowRankDatasetData } from "../components/nodes/bigramLowRankDatasetDefaults";
import { defaultInContextAssociativeRecallDatasetData } from "../components/nodes/inContextAssociativeRecallDatasetDefaults";
import {
  defaultToyLanguageDatasetData,
  TOY_LANGUAGE_DATASET_KINDS,
  type ToyLanguageDatasetKind,
} from "../components/nodes/toyLanguageDatasetDefaults";
import { defaultMlpModelData } from "../components/nodes/mlpModelDefaults";
import { defaultGatedMlpModelData } from "../components/nodes/gatedMlpModelDefaults";
import { defaultMoeMlpModelData } from "../components/nodes/moeMlpModelDefaults";
import { defaultKanModelData } from "../components/nodes/kanModelDefaults";
import { defaultMlpTokenModelData } from "../components/nodes/mlpTokenModelDefaults";
import { defaultTransformerMultiTokenModelData } from "../components/nodes/transformerMultiTokenModelDefaults";
import { defaultTransformerTokenModelData } from "../components/nodes/transformerTokenModelDefaults";
import { defaultAttentionOnlyModelData } from "../components/nodes/attentionOnlyModelDefaults";
import {
  defaultAlternativeArchTokenLmData,
  type ArchLmKind,
} from "../components/nodes/alternativeArchModelDefaults";
import { defaultMseLossData } from "../components/nodes/mseLossDefaults";
import { defaultCrossEntropyLossData } from "../components/nodes/crossEntropyLossDefaults";
import { defaultTrainerData } from "../components/nodes/trainerDefaults";

/** Dense vector regression path; ``memorization_a_dataset`` uses discrete labels → cross-entropy (see planner branch). */
const MSE_DATASETS = [
  "linear_dataset",
  "memorization_a_dataset",
  "symbolic_func_dataset",
  "uniform_linear_motion_dataset",
] as const;

const MSE_MODELS = ["mlp_model", "gated_mlp_model", "moe_mlp_model", "kan_model"] as const;

const TOKEN_DATASETS = [
  "token_prediction_dataset",
  "unigram_dataset",
  "circle_random_walk_dataset",
  "circular_motion_dataset",
  "bigram_low_rank_dataset",
  "in_context_associative_recall_dataset",
  "modular_addition_dataset",
  ...TOY_LANGUAGE_DATASET_KINDS,
] as const;

const TOKEN_MODELS = [
  "mlp_token_model",
  "gated_mlp_token_model",
  "moe_mlp_token_model",
  "transformer_token_model",
  "attention_only_model",
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
] as const;

const OPTIMIZERS = [
  "adam_optimizer",
  "adamw_optimizer",
  "sgd_optimizer",
  "signsgd_optimizer",
  "muon_optimizer",
  "shampoo_optimizer",
  "soap_optimizer",
] as const;

/** Compact layout: column gaps + vertical lanes so tall nodes (~360–420px) clear without huge empty bands. */
const COL0 = 0;
const COL1 = 360;
const COL2 = 720;
const ROW_DS = 0;
const ROW_OPT = 420;
const ROW_MD = 0;
const ROW_LOSS = 450;
const ROW_OBS = 700;
const ROW_TR = 260;

function pick<T>(arr: readonly T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)]!;
}

function randInt(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

function newPlannedId(type: string, rng: () => number): string {
  const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
  let s = "";
  for (let i = 0; i < 8; i++) s += alphabet[Math.floor(rng() * alphabet.length)]!;
  return `${type}-${s}`;
}

/** Place a new subgraph to the right of existing nodes so it does not overlap. */
export function computeSelfDrivingAnchor(nodes: Node[]): { x: number; y: number } {
  if (!nodes.length) return { x: 40, y: 40 };
  let maxX = 40;
  for (const n of nodes) {
    maxX = Math.max(maxX, n.position.x + 480);
  }
  return { x: maxX, y: 48 };
}

export type PlannedTrainerNode = {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, unknown>;
  /** Short line for self-driving agent log. */
  logLabel: string;
};

/** One canvas action: add ``node``, then add each ``edge``. Trainer is last so Code mode sees upstream nodes wired in. */
export type PlannedGraphStep = {
  node: PlannedTrainerNode;
  edges: Edge[];
};

export type PlannedRandomTrainer = {
  steps: PlannedGraphStep[];
  trainerId: string;
  trainingVizId: string;
  observableVizId: string;
};

export type RandomTrainerAppendResult = {
  newNodes: Node[];
  newEdges: Edge[];
  trainerId: string;
  visualizationId: string;
  observableVizId: string;
};

function buildEdge(
  source: string,
  target: string,
  sourceHandle: string | null,
  targetHandle: string | null,
): Edge {
  return {
    id: `e-sd-${Math.random().toString(36).slice(2, 11)}`,
    source,
    target,
    sourceHandle,
    targetHandle,
    type: "research_default",
  };
}

/**
 * Plan nodes + edges (with stable ids) for a random trainer subgraph. Does not mutate the canvas.
 */
export function planRandomTrainerSubgraph(anchor: { x: number; y: number }, rng: () => number): PlannedRandomTrainer {
  const ax = anchor.x;
  const ay = anchor.y;

  const trainerFast = { ...defaultTrainerData(), trainingSteps: 1000, logFrequency: 10 };

  const msePath = rng() < 0.55;

  const idDataset = newPlannedId("dataset", rng);
  const idModel = newPlannedId("model", rng);
  const idOpt = newPlannedId("opt", rng);
  const idLoss = newPlannedId("loss", rng);
  const idObs = newPlannedId("obs", rng);
  const idTrainer = newPlannedId("trainer", rng);
  const idOViz = newPlannedId("observable_viz", rng);
  const idTViz = newPlannedId("training_visualization", rng);

  const plannedNodes: PlannedTrainerNode[] = [];

  if (msePath) {
    const dsType = pick(MSE_DATASETS, rng);
    const modelType = pick(MSE_MODELS, rng);
    const optType = pick(OPTIMIZERS, rng);

    const inDim = randInt(rng, 3, 12);
    const sampledOutDim = randInt(rng, 1, 3);
    const memorizationACe = dsType === "memorization_a_dataset";
    const memNumClasses = memorizationACe ? randInt(rng, 8, 40) : null;
    const outDim =
      memNumClasses !== null
        ? memNumClasses
        : dsType === "symbolic_func_dataset" ||
            (modelType === "kan_model" && dsType !== "uniform_linear_motion_dataset")
          ? 1
          : sampledOutDim;

    let datasetData: Record<string, unknown>;
    if (dsType === "uniform_linear_motion_dataset") {
      const posDim = randInt(rng, 1, 2);
      datasetData = {
        ...defaultUniformLinearMotionDatasetData(),
        contextLength: 2,
        positionDim: posDim,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "symbolic_func_dataset") {
      datasetData = {
        ...defaultSymbolicFuncDatasetData(),
        inputDim: inDim,
        outputDim: outDim,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "memorization_a_dataset") {
      datasetData = {
        ...defaultMemorizationADatasetData(),
        inputDim: inDim,
        outputDim: outDim,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else {
      datasetData = {
        ...defaultLinearDatasetData(),
        inputDim: inDim,
        outputDim: outDim,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    }

    let modelData: Record<string, unknown>;
    if (dsType === "uniform_linear_motion_dataset") {
      const posDim = (datasetData as { positionDim?: number }).positionDim ?? 1;
      const flat = 2 * posDim;
      if (modelType === "kan_model") {
        modelData = {
          ...defaultKanModelData(),
          inputDim: flat,
          outputDim: flat,
          depth: 2,
          width: rng() < 0.5 ? 5 : 6,
          seed: randInt(rng, 0, 10_000),
        };
      } else if (modelType === "moe_mlp_model") {
        modelData = {
          ...defaultMoeMlpModelData(),
          inputDim: flat,
          outputDim: flat,
          depth: 2,
          width: 32,
          numExperts: 2,
          seed: randInt(rng, 0, 10_000),
        };
      } else if (modelType === "gated_mlp_model") {
        modelData = {
          ...defaultGatedMlpModelData(),
          inputDim: flat,
          outputDim: flat,
          depth: 2,
          width: 48,
          seed: randInt(rng, 0, 10_000),
        };
      } else {
        modelData = {
          ...defaultMlpModelData(),
          inputDim: flat,
          outputDim: flat,
          depth: 2,
          width: 48,
          seed: randInt(rng, 0, 10_000),
        };
      }
    } else if (modelType === "kan_model") {
      modelData = {
        ...defaultKanModelData(),
        inputDim: inDim,
        outputDim: outDim,
        depth: 2,
        width: rng() < 0.5 ? 5 : 6,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (modelType === "moe_mlp_model") {
      modelData = {
        ...defaultMoeMlpModelData(),
        inputDim: inDim,
        outputDim: outDim,
        depth: 2,
        width: 32,
        numExperts: 2,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (modelType === "gated_mlp_model") {
      modelData = {
        ...defaultGatedMlpModelData(),
        inputDim: inDim,
        outputDim: outDim,
        depth: 2,
        width: 48,
        seed: randInt(rng, 0, 10_000),
      };
    } else {
      modelData = {
        ...defaultMlpModelData(),
        inputDim: inDim,
        outputDim: outDim,
        depth: 2,
        width: 48,
        seed: randInt(rng, 0, 10_000),
      };
    }

    let lossData = { ...defaultMseLossData() };
    if (!memorizationACe && dsType === "uniform_linear_motion_dataset") {
      lossData = {
        ...lossData,
        lossMaskContextLength: 2,
        lossMaskMode: "all",
      };
    }

    const optData =
      optType === "adamw_optimizer"
        ? { ...defaultAdamWOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "sgd_optimizer"
        ? { ...defaultSgdOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "signsgd_optimizer"
          ? { ...defaultSignSgdOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "muon_optimizer"
          ? { ...defaultMuonOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "shampoo_optimizer"
          ? { ...defaultShampooOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "soap_optimizer"
          ? { ...defaultSoapOptimizerData(), seed: randInt(rng, 0, 10_000) }
          : { ...defaultAdamOptimizerData(), seed: randInt(rng, 0, 10_000) };

    plannedNodes.push(
      {
        id: idDataset,
        type: dsType,
        position: { x: ax + COL0, y: ay + ROW_DS },
        data: datasetData,
        logLabel: `Add dataset (${dsType})`,
      },
      {
        id: idModel,
        type: modelType,
        position: { x: ax + COL1, y: ay + ROW_MD },
        data: modelData,
        logLabel: `Add model (${modelType})`,
      },
      {
        id: idOpt,
        type: optType,
        position: { x: ax + COL0, y: ay + ROW_OPT },
        data: optData as Record<string, unknown>,
        logLabel: `Add optimizer (${optType})`,
      },
      {
        id: idLoss,
        type: memorizationACe ? "cross_entropy_loss" : "mse_loss",
        position: { x: ax + COL1, y: ay + ROW_LOSS },
        data: (memorizationACe ? defaultCrossEntropyLossData() : lossData) as Record<string, unknown>,
        logLabel: memorizationACe ? "Add cross-entropy loss (Memorization A)" : "Add MSE loss",
      },
      {
        id: idObs,
        type: memorizationACe ? "observable_accuracy" : "observable_weight_l2",
        position: { x: ax + COL1, y: ay + ROW_OBS },
        data: memorizationACe ? {} : { normAggregation: "global" },
        logLabel: memorizationACe ? "Add Accuracy observable" : "Add Weight L2 observable",
      },
      {
        id: idTrainer,
        type: "trainer",
        position: { x: ax + COL2, y: ay + ROW_TR },
        data: trainerFast as Record<string, unknown>,
        logLabel: "Add trainer",
      },
    );
  } else {
    const dsType = pick(TOKEN_DATASETS, rng);
    const modelType =
      dsType === "circular_motion_dataset" ? "transformer_multi_token_model" : pick(TOKEN_MODELS, rng);
    const optType = pick(OPTIMIZERS, rng);

    const vocab = randInt(rng, 12, 48);
    const ctx = randInt(rng, 3, 6);

    let datasetData: Record<string, unknown>;
    if (dsType === "token_prediction_dataset") {
      datasetData = {
        ...defaultTokenPredictionDatasetData(),
        vocabSize: vocab,
        contextLength: ctx,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "unigram_dataset") {
      datasetData = {
        ...defaultUnigramDatasetData(),
        vocabSize: vocab,
        contextLength: ctx,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "circle_random_walk_dataset") {
      datasetData = {
        ...defaultCircleRandomWalkDatasetData(),
        vocabSize: vocab,
        contextLength: ctx,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "circular_motion_dataset") {
      datasetData = {
        ...defaultCircularMotionDatasetData(),
        vocabSize: vocab,
        contextLength: ctx,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "bigram_low_rank_dataset") {
      const rankMax = Math.max(1, Math.min(20, vocab));
      datasetData = {
        ...defaultBigramLowRankDatasetData(),
        vocabSize: vocab,
        rank: randInt(rng, 1, rankMax),
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (dsType === "modular_addition_dataset") {
      const modulus = randInt(rng, 31, 97);
      datasetData = {
        ...defaultModularAdditionDatasetData(),
        modulus,
        trainFraction: 0.3,
        seed: randInt(rng, 0, 10_000),
      };
    } else if ((TOY_LANGUAGE_DATASET_KINDS as readonly string[]).includes(dsType)) {
      datasetData = {
        ...defaultToyLanguageDatasetData(dsType as ToyLanguageDatasetKind),
        vocabSize: vocab,
        contextLength: ctx,
        trainSize: 400,
        testSize: 80,
        seed: randInt(rng, 0, 10_000),
      };
    } else {
      const numPairs = randInt(rng, 2, 4);
      datasetData = {
        ...defaultInContextAssociativeRecallDatasetData(),
        vocabSize: vocab,
        numPairs,
        inContextRepeat: 1,
        crossSampleRepeatProb: 0,
        repeatedTokenCount: Math.min(2, vocab - 1),
        trainSize: 800,
        testSize: 120,
        seed: randInt(rng, 0, 10_000),
      };
    }

    const ctxModel =
      dsType === "in_context_associative_recall_dataset"
        ? 2 * ((datasetData as { numPairs?: number }).numPairs ?? 2) + 1
        : dsType === "bigram_low_rank_dataset"
          ? 1
          : dsType === "modular_addition_dataset"
            ? 2
          : ctx;
    const modelVocab =
      dsType === "modular_addition_dataset"
        ? Number((datasetData as { modulus?: number }).modulus ?? vocab)
        : vocab;

    let modelData: Record<string, unknown>;
    if (
      modelType === "mlp_token_model" ||
      modelType === "gated_mlp_token_model" ||
      modelType === "moe_mlp_token_model"
    ) {
      modelData = {
        ...defaultMlpTokenModelData(),
        vocabSize: modelVocab,
        embedDim: 32,
        tokensPerInput: ctxModel,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (modelType === "transformer_token_model") {
      modelData = {
        ...defaultTransformerTokenModelData(),
        vocabSize: modelVocab,
        contextLength: ctxModel,
        modelDim: 64,
        numHeads: 4,
        numLayers: 1,
        ffDim: 128,
        seed: randInt(rng, 0, 10_000),
      };
    } else if (modelType === "transformer_multi_token_model") {
      modelData = {
        ...defaultTransformerMultiTokenModelData(),
        vocabSize: modelVocab,
        contextLength: ctxModel,
        tokensPerPosition: 2,
        modelDim: 64,
        numHeads: 4,
        numLayers: 1,
        ffDim: 128,
        tieEmbeddingLmHead: "no",
        seed: randInt(rng, 0, 10_000),
      };
    } else if (modelType === "attention_only_model") {
      modelData = {
        ...defaultAttentionOnlyModelData(),
        vocabSize: modelVocab,
        embedDim: 32,
        numHeads: 4,
        contextLength: ctxModel,
        seed: randInt(rng, 0, 10_000),
      };
    } else {
      const archKind = modelType as ArchLmKind;
      modelData = {
        ...defaultAlternativeArchTokenLmData(archKind),
        vocabSize: modelVocab,
        embedDim: 32,
        contextLength: ctxModel,
        seed: randInt(rng, 0, 10_000),
      };
    }

    const optData =
      optType === "adamw_optimizer"
        ? { ...defaultAdamWOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "sgd_optimizer"
        ? { ...defaultSgdOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "signsgd_optimizer"
          ? { ...defaultSignSgdOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "muon_optimizer"
          ? { ...defaultMuonOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "shampoo_optimizer"
          ? { ...defaultShampooOptimizerData(), seed: randInt(rng, 0, 10_000) }
        : optType === "soap_optimizer"
          ? { ...defaultSoapOptimizerData(), seed: randInt(rng, 0, 10_000) }
          : { ...defaultAdamOptimizerData(), seed: randInt(rng, 0, 10_000) };

    plannedNodes.push(
      {
        id: idDataset,
        type: dsType,
        position: { x: ax + COL0, y: ay + ROW_DS },
        data: datasetData,
        logLabel: `Add dataset (${dsType})`,
      },
      {
        id: idModel,
        type: modelType,
        position: { x: ax + COL1, y: ay + ROW_MD },
        data: modelData,
        logLabel: `Add model (${modelType})`,
      },
      {
        id: idOpt,
        type: optType,
        position: { x: ax + COL0, y: ay + ROW_OPT },
        data: optData as Record<string, unknown>,
        logLabel: `Add optimizer (${optType})`,
      },
      {
        id: idLoss,
        type: "cross_entropy_loss",
        position: { x: ax + COL1, y: ay + ROW_LOSS },
        data: defaultCrossEntropyLossData() as Record<string, unknown>,
        logLabel: "Add cross-entropy loss",
      },
      {
        id: idObs,
        type: "observable_accuracy",
        position: { x: ax + COL1, y: ay + ROW_OBS },
        data: {},
        logLabel: "Add Accuracy observable",
      },
      {
        id: idTrainer,
        type: "trainer",
        position: { x: ax + COL2, y: ay + ROW_TR },
        data: trainerFast as Record<string, unknown>,
        logLabel: "Add trainer",
      },
    );
  }

  const edges: Edge[] = [
    buildEdge(idDataset, idTrainer, "dataset", "dataset"),
    buildEdge(idModel, idTrainer, "model", "model"),
    buildEdge(idOpt, idTrainer, "optimizer", "optimizer"),
    buildEdge(idLoss, idTrainer, "loss", "loss"),
    buildEdge(idObs, idTrainer, "observables", "observables"),
  ];

  const nodeById = new Map(plannedNodes.map((n) => [n.id, n]));
  const steps: PlannedGraphStep[] = [
    { node: nodeById.get(idDataset)!, edges: [] },
    { node: nodeById.get(idModel)!, edges: [] },
    { node: nodeById.get(idOpt)!, edges: [] },
    { node: nodeById.get(idLoss)!, edges: [] },
    { node: nodeById.get(idObs)!, edges: [] },
    { node: nodeById.get(idTrainer)!, edges },
  ];

  return {
    steps,
    trainerId: idTrainer,
    trainingVizId: idTViz,
    observableVizId: idOViz,
  };
}

/** Append planned subgraph in one shot (used by random generate). */
export function appendRandomTrainerSubgraph(
  existingNodes: Node[],
  anchor: { x: number; y: number },
  rng: () => number,
): RandomTrainerAppendResult {
  const plan = planRandomTrainerSubgraph(anchor, rng);
  let pool = [...existingNodes];
  const newNodes: Node[] = [];
  const newEdges: Edge[] = [];
  for (const step of plan.steps) {
    const n = appendResearchNode(pool, step.node.type, step.node.position, step.node.data, step.node.id);
    pool = [...pool, n];
    newNodes.push(n);
    newEdges.push(...step.edges);
  }
  const fin = ensureTrainerAutoVizes(pool, newEdges, plan.trainerId, plan.observableVizId, plan.trainingVizId);
  return {
    newNodes: fin.nodes.slice(existingNodes.length),
    newEdges: fin.edges,
    trainerId: plan.trainerId,
    visualizationId: plan.trainingVizId,
    observableVizId: plan.observableVizId,
  };
}

/** @deprecated Use ``appendRandomTrainerSubgraph``. */
export function appendSelfDrivingTrainerSubgraph(
  existingNodes: Node[],
  anchor: { x: number; y: number },
  rng: () => number,
): RandomTrainerAppendResult {
  return appendRandomTrainerSubgraph(existingNodes, anchor, rng);
}
