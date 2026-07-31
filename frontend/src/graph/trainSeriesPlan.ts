import type { Edge, Node } from "@xyflow/react";
import { BIGRAM_SPECTRUM_DECAY_IDS } from "../components/nodes/bigramLowRankDatasetDefaults";
import { enumChoices, floatChoices, intChoices } from "../components/nodes/multiValueUtils";
import { TOY_LANGUAGE_DATASET_KINDS } from "../components/nodes/toyLanguageDatasetDefaults";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";

export const MAX_TRAIN_SERIES = 256;

export type TrainSeriesAssignment = { nodeId: string; key: string; value: unknown };

function incomingSource(edges: Edge[], trainerId: string, handle: string): string | null {
  const e = edges.find((x) => x.target === trainerId && (x.targetHandle ?? null) === handle);
  return e?.source ?? null;
}

function incomingSources(edges: Edge[], trainerId: string, handle: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const e of edges) {
    if (e.target !== trainerId || (e.targetHandle ?? null) !== handle) continue;
    const src = e.source;
    if (seen.has(src)) continue;
    seen.add(src);
    out.push(src);
  }
  return out;
}

function trainerDatasetSourceId(edges: Edge[], trainerId: string): string | null {
  return incomingSource(edges, trainerId, "dataset") ?? incomingSource(edges, trainerId, "train_dataset");
}

function incomingSourceToNode(edges: Edge[], targetNodeId: string, handle: string): string | null {
  const e = edges.find((x) => x.target === targetNodeId && (x.targetHandle ?? null) === handle);
  return e?.source ?? null;
}

type Axis = { nodeId: string; key: string; values: unknown[] };

/** Generic sweep axes for generated NodeDef nodes.
 * A spec without fields has no axes; only a missing spec uses specialized logic. */
function axesFromGeneratedSpec(nodeId: string, type: string, d: Record<string, unknown>): Axis[] {
  const spec = GENERATED_NODE_SPECS[type];
  if (!spec?.fields?.length) return [];
  const out: Axis[] = [];
  for (const f of spec.fields) {
    if ((f as { sweepable?: boolean }).sweepable === false) continue; // 非 sweep 旋钮
    if (f.kind === "int") out.push({ nodeId, key: f.key, values: intChoices(d[f.key], Number(f.defaultValue)) });
    else if (f.kind === "float")
      out.push({
        nodeId, key: f.key,
        // sweepKind: "int" makes the runtime interpret this float field as an integer.
        values: (f as { sweepKind?: string }).sweepKind === "int"
          ? intChoices(d[f.key], Number(f.defaultValue))
          : floatChoices(d[f.key], Number(f.defaultValue)),
      });
    else if (f.kind === "enum" && "options" in f)
      out.push({ nodeId, key: f.key, values: enumChoices(d[f.key], new Set(f.options), String(f.defaultValue)) });
    // Optimizer list fields accept comma-separated float or integer choices.
    else if (f.kind === "floatList") out.push({ nodeId, key: f.key, values: floatChoices(d[f.key], Number(f.defaultValue)) });
    else if (f.kind === "intList") out.push({ nodeId, key: f.key, values: intChoices(d[f.key], Number(f.defaultValue)) });
  }
  return out;
}

// Model types whose generated fields define sweep axes. Other generated model
// types intentionally expose no axes. kan_reg axes belong only to observables.
const LOSS_SWEEP_ALLOWLIST = new Set([
  "mse_loss", "cross_entropy_loss", "binary_cross_entropy_with_logits_loss",
  "diffusion_mse_loss", "l1_reg", "l2_reg", "l2_projection",
]);

const MODEL_SWEEP_ALLOWLIST = new Set([
  "mlp_model", "gated_mlp_model", "moe_mlp_model", "crl_residual_mlp", "kan_model",
  "attention_only_model", "linear_attention_model", "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model", "hyena_like_conv_model", "slot_attention_token_model",
  "diffusion_score_model", "numeric_transformer_model", "mpp_spatiotemporal_model",
  "afno_lite_spatiotemporal_model", "transformer_token_model", "transformer_multi_token_model",
  "vgg11_cifar_model", "small_inception_cifar_model",
]);

const INPUT_DIST_IDS = new Set(["standard_normal", "uniform_neg1_1", "uniform_0_1"]);
const NUMERIC_TRANSFORMER_CAUSAL_IDS = new Set(["yes", "no"]);
const TRANSFORMER_TOKEN_TIE_IDS = new Set(["yes", "no"]);
const TRANSFORMER_TOKEN_ENCODER_ACTIVATION_IDS = new Set(["gelu", "relu", "silu"]);
const TRANSFORMER_ENCODER_BACKEND_IDS = new Set(["pytorch", "stable"]);
const OUTPUT_DIST_IDS = new Set(["additive_gaussian", "deterministic"]);
const MEM_OUTPUT_DIST_IDS = new Set(["uniform_class_probs", "power_law_class_probs", "exponential_class_probs"]);
const ACT_IDS = new Set(["relu", "gelu", "tanh", "sigmoid", "leaky_relu", "silu", "identity"]);
const BIGRAM_DECAY_IDS = new Set<string>(BIGRAM_SPECTRUM_DECAY_IDS);
const KAN_BASE_FUN_IDS = new Set(["silu", "identity", "zero"]);

const TOY_LANGUAGE_DATASET_TYPE_SET = new Set<string>(TOY_LANGUAGE_DATASET_KINDS);

const FORMAL_LANG_FOR_SWEEP_IDS = new Set(["anbn", "anbncn", "palindrome", "parity"]);
const PCFG_GEN_MODE_IDS = new Set(["binary_tree", "cfg_sentence"]);
const TOY_DATA_SOURCE_IDS = new Set(["synthetic", "download"]);
const TOY_TOKENIZER_MODE_IDS = new Set(["char", "word"]);

function cartesianPick<T>(arrays: T[][]): T[][] {
  if (!arrays.length) return [[]];
  const [head, ...tail] = arrays;
  const rest = cartesianPick(tail);
  const out: T[][] = [];
  for (const h of head) {
    for (const r of rest) {
      out.push([h, ...r]);
    }
  }
  return out;
}













function axesForTeacherDataset(nodeId: string, d: Record<string, unknown>): Axis[] {
  void nodeId;
  void d;
  return [];
}

function axesForInputSampler(nodeId: string, d: Record<string, unknown>): Axis[] {
  return [
    { nodeId, key: "numSamples", values: intChoices(d.numSamples, 800) },
  ];
}


function axesForRandomInputDistribution(nodeId: string, d: Record<string, unknown>): Axis[] {
  return [
    { nodeId, key: "inputDim", values: intChoices(d.inputDim, 10) },
    {
      nodeId,
      key: "inputDistribution",
      values: enumChoices(d.inputDistribution, INPUT_DIST_IDS, "standard_normal"),
    },
    {
      nodeId,
      key: "noiseDistribution",
      values: enumChoices(d.noiseDistribution, OUTPUT_DIST_IDS, "deterministic"),
    },
    { nodeId, key: "noiseLevel", values: floatChoices(d.noiseLevel, 0) },
    { nodeId, key: "seed", values: intChoices(d.seed, 0) },
  ];
}









const CRL_ENV_PRESET_IDS = new Set(["point_u4_maze", "ant_u4_maze"]);

function axesForCrlEnvConfig(nodeId: string, d: Record<string, unknown>): Axis[] {
  return [
    { nodeId, key: "preset", values: enumChoices(d.preset, CRL_ENV_PRESET_IDS, "point_u4_maze") },
    { nodeId, key: "numEnvs", values: intChoices(d.numEnvs, 8) },
    { nodeId, key: "episodeLength", values: intChoices(d.episodeLength, 200) },
    { nodeId, key: "mazeSizeScaling", values: floatChoices(d.mazeSizeScaling, 4) },
    { nodeId, key: "seed", values: intChoices(d.seed, 0) },
  ];
}

function collectAxes(nodes: Node[], edges: Edge[], trainerId: string): Axis[] {
  const axes: Axis[] = [];
  const trainerNode = nodes.find((x) => x.id === trainerId);
  const isCrlTrainer = trainerNode?.type === "crl_trainer";

  if (!isCrlTrainer) {
  const trainDs = trainerDatasetSourceId(edges, trainerId);
  if (trainDs) {
    const n = nodes.find((x) => x.id === trainDs);
    // Generated dataset fields define sweep axes. teacher_dataset is excluded
    // because its sampler/distribution satellite chain owns those axes.
    if (n?.type != null && n.type !== "teacher_dataset" && GENERATED_NODE_SPECS[String(n.type)]) {
      axes.push(...axesFromGeneratedSpec(n.id, String(n.type), (n.data ?? {}) as Record<string, unknown>));
    } else if (n?.type === "teacher_dataset") {
      axes.push(...axesForTeacherDataset(n.id, (n.data ?? {}) as Record<string, unknown>));
      const trainSampler = incomingSourceToNode(edges, n.id, "train_input");
      const testSampler = incomingSourceToNode(edges, n.id, "test_input");
      const seenSampler = new Set<string>();
      for (const samplerId of [trainSampler, testSampler]) {
        if (!samplerId) continue;
        if (seenSampler.has(samplerId)) continue;
        seenSampler.add(samplerId);
        const samplerNode = nodes.find((x) => x.id === samplerId);
        if (samplerNode?.type !== "input_sampler") continue;
        axes.push(...axesForInputSampler(samplerNode.id, (samplerNode.data ?? {}) as Record<string, unknown>));
        const rid = incomingSourceToNode(edges, samplerNode.id, "distribution");
        if (!rid) continue;
        const rn = nodes.find((x) => x.id === rid);
        if (rn?.type === "random_input_distribution") {
          axes.push(...axesForRandomInputDistribution(rn.id, (rn.data ?? {}) as Record<string, unknown>));
        }
      }
    }
  }
  const testDs = incomingSource(edges, trainerId, "test_dataset");
  if (testDs && testDs !== trainDs) {
    const n = nodes.find((x) => x.id === testDs);
    // NodeDef 通道 generated-first(同 train 侧)。
    // teacher_dataset 排除(hazard 1):其 sweep 是图遍历卫星链(sampler→
    // distribution 6 轴只经此链存在),generated-first 截胡会静默清空。
    if (n?.type != null && n.type !== "teacher_dataset" && GENERATED_NODE_SPECS[String(n.type)]) {
      axes.push(...axesFromGeneratedSpec(n.id, String(n.type), (n.data ?? {}) as Record<string, unknown>));
    } else if (n?.type === "teacher_dataset") {
      axes.push(...axesForTeacherDataset(n.id, (n.data ?? {}) as Record<string, unknown>));
      const trainSampler = incomingSourceToNode(edges, n.id, "train_input");
      const testSampler = incomingSourceToNode(edges, n.id, "test_input");
      const seenSampler = new Set<string>();
      for (const samplerId of [trainSampler, testSampler]) {
        if (!samplerId) continue;
        if (seenSampler.has(samplerId)) continue;
        seenSampler.add(samplerId);
        const samplerNode = nodes.find((x) => x.id === samplerId);
        if (samplerNode?.type !== "input_sampler") continue;
        axes.push(...axesForInputSampler(samplerNode.id, (samplerNode.data ?? {}) as Record<string, unknown>));
        const rid = incomingSourceToNode(edges, samplerNode.id, "distribution");
        if (!rid) continue;
        const rn = nodes.find((x) => x.id === rid);
        if (rn?.type === "random_input_distribution") {
          axes.push(...axesForRandomInputDistribution(rn.id, (rn.data ?? {}) as Record<string, unknown>));
        }
      }
    }
  }
  }

  if (isCrlTrainer) {
    const envId = incomingSource(edges, trainerId, "env");
    if (envId) {
      const en = nodes.find((x) => x.id === envId);
      if (en?.type === "crl_env_config") {
        axes.push(...axesForCrlEnvConfig(en.id, (en.data ?? {}) as Record<string, unknown>));
      }
    }
    if (trainerNode?.type != null && GENERATED_NODE_SPECS[String(trainerNode.type)]) {
      // NodeDef 通道 generated-first:crl_trainer 轴=可 sweep 字段
      // (computeDevice sweepable=False、disableEntropy boolean 天然无轴)。
      axes.push(...axesFromGeneratedSpec(trainerId, String(trainerNode.type), (trainerNode.data ?? {}) as Record<string, unknown>));
    }
  }

  const modelId = incomingSource(edges, trainerId, "model");
  if (modelId) {
    const n = nodes.find((x) => x.id === modelId);
    // Only allowlisted model types derive sweep axes from generated fields.
    if (n?.type != null && GENERATED_NODE_SPECS[String(n.type)]) {
      if (MODEL_SWEEP_ALLOWLIST.has(String(n.type))) {
        const md = (n.data ?? {}) as Record<string, unknown>;
        axes.push(...axesFromGeneratedSpec(n.id, String(n.type), md));
        // gated/moe manifests omit outputScale, but stale user data may still
        // contain it and therefore expose a sweep axis. mlp_model is field-driven.
        if (n.type !== "mlp_model" && "outputScale" in md && ["gated_mlp_model", "moe_mlp_model"].includes(String(n.type))) {
          axes.push({ nodeId: n.id, key: "outputScale", values: floatChoices(md.outputScale, 1) });
        }
      }
    }
  }
  const optId = incomingSource(edges, trainerId, "optimizer");
  if (optId) {
    const n = nodes.find((x) => x.id === optId);
    // NodeDef 通道 generated-first:optimizer 9/9 全员有轴(dataset 教义
    // 轴=fields,与 model allowlist 相反);空集恒 miss,else-if 链逐批递减。
    if (n?.type != null && GENERATED_NODE_SPECS[String(n.type)]) {
      axes.push(...axesFromGeneratedSpec(n.id, String(n.type), (n.data ?? {}) as Record<string, unknown>));
    }
    // lr/mup 卫星双路径迁 generated-first;**mupLrAxisSources 去重结构
    // The same muP node may connect to two handles; collect its axes only once.
    const mupLrAxisSources = new Set<string>();
    const lrSchedId = incomingSourceToNode(edges, optId, "lr_schedule");
    if (lrSchedId) {
      const ls = nodes.find((x) => x.id === lrSchedId);
      if (ls?.type != null && GENERATED_NODE_SPECS[String(ls.type)]) {
        axes.push(...axesFromGeneratedSpec(ls.id, String(ls.type), (ls.data ?? {}) as Record<string, unknown>));
        if (ls.type === "mup_lr_schedule") mupLrAxisSources.add(ls.id);
      }
    }
    const mupLrSchedId = incomingSourceToNode(edges, optId, "mup_lr_schedule");
    if (mupLrSchedId && !mupLrAxisSources.has(mupLrSchedId)) {
      const ms = nodes.find((x) => x.id === mupLrSchedId);
      if (ms?.type === "mup_lr_schedule" && GENERATED_NODE_SPECS["mup_lr_schedule"]) {
        axes.push(...axesFromGeneratedSpec(ms.id, "mup_lr_schedule", (ms.data ?? {}) as Record<string, unknown>));
      }
    }
  }
  const lossIds = incomingSources(edges, trainerId, "loss");
  if (!isCrlTrainer && lossIds.length > 0) {
    for (const lossId of lossIds) {
      const n = nodes.find((x) => x.id === lossId);
      if (!n) continue;
      // NodeDef 通道 generated-first，门到 LOSS_SWEEP_ALLOWLIST；kan_reg
      // 进入 GENERATED 后，在 loss handle 上不得生成轴。
      // 不得新造轴——它的轴只属 observables handle)。
      if (n.type != null && LOSS_SWEEP_ALLOWLIST.has(String(n.type)) && GENERATED_NODE_SPECS[String(n.type)]) {
        axes.push(...axesFromGeneratedSpec(n.id, String(n.type), (n.data ?? {}) as Record<string, unknown>));
      }
    }
  }
  for (const e of edges) {
    if (e.target !== trainerId || (e.targetHandle ?? null) !== "observables") continue;
    const n = nodes.find((x) => x.id === e.source);
    if (n && n.type != null && GENERATED_NODE_SPECS[String(n.type)]) {
      axes.push(...axesFromGeneratedSpec(n.id, String(n.type), (n.data ?? {}) as Record<string, unknown>));
      continue;
    }
  }
  if (!isCrlTrainer && trainerNode?.type === "trainer" && GENERATED_NODE_SPECS["trainer"]) {
    // Trainer generated fields provide four sweep axes.
    axes.push(...axesFromGeneratedSpec(trainerNode.id, "trainer", (trainerNode.data ?? {}) as Record<string, unknown>));
  }
  return axes;
}

/** Stable id for a sweep axis (node + field). */
export function trainSeriesAxisKey(nodeId: string, key: string): string {
  return `${nodeId}\x1f${key}`;
}

/** Axes that actually vary in the current graph (multi-valued fields). */
export function getSweptAxisIdSet(nodes: Node[], edges: Edge[], trainerId: string): Set<string> {
  const axes = collectAxes(nodes, edges, trainerId);
  const out = new Set<string>();
  for (const a of axes) {
    if (a.values.length > 1) out.add(trainSeriesAxisKey(a.nodeId, a.key));
  }
  return out;
}

/** Each inner array is one experiment: assignments to apply onto node.data before POST /api/train. */
export function planTrainSeriesAssignments(nodes: Node[], edges: Edge[], trainerId: string): TrainSeriesAssignment[][] {
  const axes = collectAxes(nodes, edges, trainerId);
  if (!axes.length) return [[]];
  const pickLists = axes.map((a) => a.values.map((value) => ({ nodeId: a.nodeId, key: a.key, value })));
  let product = 1;
  for (const pl of pickLists) {
    product *= Math.max(1, pl.length);
    if (product > MAX_TRAIN_SERIES) {
      throw new Error(
        `Too many train combinations (${product} product of sweep axes). Max is ${MAX_TRAIN_SERIES}. Reduce multi-valued parameters.`,
      );
    }
  }
  const picks = cartesianPick(pickLists);
  if (picks.length > MAX_TRAIN_SERIES) {
    throw new Error(
      `Too many train combinations (${picks.length}). Max is ${MAX_TRAIN_SERIES}. Reduce multi-valued parameters.`,
    );
  }
  return picks;
}

export type SerializedTrainNode = {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, unknown>;
  /** Required for ``combined_model``: backend resolves inner layers via React Flow parent ids. */
  parentId?: string;
};

export type SerializedTrainEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle: string | null;
  targetHandle: string | null;
};

type ExecutionGraphNode = Pick<Node, "id" | "type" | "position" | "data" | "parentId">;
type ExecutionGraphEdge = Pick<
  Edge,
  "id" | "source" | "target" | "sourceHandle" | "targetHandle"
>;

const TRAINER_RUNTIME_DATA_KEYS = new Set([
  "hostTrainUi",
  "trainUi",
  "trainPauseCheckpoint",
  "trainCompleteCheckpoint",
  "trainSeriesPauseCtx",
  "lossHistory",
  "testLossHistory",
  "regLossHistory",
  "stepTicks",
  "epochTicks",
  "observableMetricHistories",
  "memoryCheckpoint_b64",
  "checkpoint_b64",
  "targetCurveStepTicks",
  "targetCurveLossHistory",
  "lastTrainLoopSeconds",
  "lastAutoTuneSummary",
  "autoTuneComparisonResult",
]);

/** Shape sent to ``POST /api/train`` (subset of XYFlow node fields the engine needs). */
export function serializeNodeForTrain(n: ExecutionGraphNode): SerializedTrainNode {
  const raw = (n.data as Record<string, unknown>) ?? {};
  /** Canvas-only UI; must not affect train parity (manual vs sweep auto-train / graph assist). */
  let data: Record<string, unknown> = raw;
  if (n.type === "trainer" || n.type === "crl_trainer") {
    data = {};
    for (const [key, value] of Object.entries(raw)) {
      if (!TRAINER_RUNTIME_DATA_KEYS.has(key)) data[key] = value;
    }
  } else if (
    n.type === "model_checkpoint" &&
    typeof raw.checkpoint_b64 === "string" &&
    raw.checkpoint_b64 !== "" &&
    raw.checkpoint_b64 === raw.memoryCheckpoint_b64
  ) {
    const { memoryCheckpoint_b64: _duplicate, ...rest } = raw;
    data = rest;
  }
  const base: SerializedTrainNode = {
    id: n.id,
    type: n.type as string,
    position: { x: n.position.x, y: n.position.y },
    data,
  };
  if (n.parentId != null && n.parentId !== "") {
    base.parentId = String(n.parentId);
  }
  return base;
}

/** Minimal executable graph for one target: upstream wires plus combined-model structure. */
export function serializeExecutionGraphForTarget(
  nodes: ExecutionGraphNode[],
  edges: ExecutionGraphEdge[],
  targetId: string,
): { nodes: SerializedTrainNode[]; edges: SerializedTrainEdge[] } {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const included = new Set<string>();
  if (nodeById.has(targetId)) included.add(targetId);

  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of edges) {
      if (
        included.has(edge.target) &&
        nodeById.get(edge.target)?.type !== "model_checkpoint" &&
        nodeById.has(edge.source) &&
        !included.has(edge.source)
      ) {
        included.add(edge.source);
        changed = true;
      }
    }
    for (const node of nodes) {
      if (included.has(node.id) && node.parentId && nodeById.has(String(node.parentId))) {
        const parentId = String(node.parentId);
        if (!included.has(parentId)) {
          included.add(parentId);
          changed = true;
        }
      }
      if (node.parentId && included.has(String(node.parentId))) {
        const parent = nodeById.get(String(node.parentId));
        if (parent?.type === "combined_model" && !included.has(node.id)) {
          included.add(node.id);
          changed = true;
        }
      }
    }
  }

  return {
    nodes: nodes.filter((node) => included.has(node.id)).map(serializeNodeForTrain),
    edges: edges
      .filter((edge) => included.has(edge.source) && included.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle ?? null,
        targetHandle: edge.targetHandle ?? null,
      })),
  };
}

export function applyAssignmentsToNodes(
  nodes: SerializedTrainNode[],
  combo: TrainSeriesAssignment[],
): SerializedTrainNode[] {
  const out: SerializedTrainNode[] = structuredClone(nodes);
  for (const c of combo) {
    const n = out.find((x) => x.id === c.nodeId);
    if (!n) continue;
    n.data = { ...n.data, [c.key]: c.value };
  }
  return out;
}

function formatScalar(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return String(value);
    return String(Number(value.toPrecision(6)));
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null) return "null";
  return String(value);
}

const TYPE_SHORT: Record<string, string> = {
  mlp_model: "model",
  moe_mlp_model: "model",
  attention_only_model: "model",
  linear_attention_model: "model",
  diagonal_ssm_token_model: "model",
  rwkv_time_mix_token_model: "model",
  hyena_like_conv_model: "model",
  slot_attention_token_model: "model",
  diffusion_score_model: "model",
  numeric_transformer_model: "model",
  mpp_spatiotemporal_model: "model",
  afno_lite_spatiotemporal_model: "model",
  transformer_token_model: "model",
  transformer_multi_token_model: "model",
  vgg11_cifar_model: "model",
  small_inception_cifar_model: "model",
  adam_optimizer: "adam",
  adamw_optimizer: "adamw",
  sgd_optimizer: "sgd",
  signsgd_optimizer: "signsgd",
  muon_optimizer: "muon",
  shampoo_optimizer: "shampoo",
  soap_optimizer: "soap",
  lr_schedule: "lrsched",
  mup_lr_schedule: "mup_lr",
  mup_initialization: "mup_init",
  mse_loss: "loss",
  diffusion_mse_loss: "loss",
  cross_entropy_loss: "loss",
  binary_cross_entropy_with_logits_loss: "loss",
  crl_env_config: "env",
  crl_residual_mlp: "crl_model",
  crl_trainer: "crl",
  token_prediction_dataset: "dataset",
  memorization_a_dataset: "dataset",
  memorization_b_dataset: "dataset",
  circle_random_walk_dataset: "dataset",
  circular_motion_dataset: "dataset",
  bigram_low_rank_dataset: "dataset",
  in_context_associative_recall_dataset: "dataset",
  teacher_dataset: "dataset",
  uniform_linear_motion_dataset: "dataset",
  diffusion_pde_dataset: "dataset",
  reaction_diffusion_dataset: "dataset",
  advection_dataset: "dataset",
  modular_addition_dataset: "dataset",
  random_input_distribution: "xdist",
  input_sampler: "sampler",
};

/** One line per swept parameter (only axes with multiple values). */
export function formatTrainSeriesSweptLines(
  combo: TrainSeriesAssignment[],
  nodes: SerializedTrainNode[],
  sweptAxisIds: Set<string>,
): string[] {
  const filtered = combo.filter((c) => sweptAxisIds.has(trainSeriesAxisKey(c.nodeId, c.key)));
  if (!filtered.length) return ["defaults"];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const datasetIds: string[] = [];
  const parts: string[] = [];
  for (const c of filtered) {
    const node = byId.get(c.nodeId);
    const t = (node?.type as string) ?? "?";
    let prefix: string;
    if (
      t === "linear_dataset" ||
      t === "memorization_a_dataset" ||
      t === "memorization_b_dataset" ||
      t === "symbolic_func_dataset" ||
      t === "token_prediction_dataset" ||
      t === "circle_random_walk_dataset" ||
      t === "circular_motion_dataset" ||
      t === "bigram_low_rank_dataset" ||
      t === "in_context_associative_recall_dataset" ||
      t === "teacher_dataset" ||
      t === "uniform_linear_motion_dataset" ||
      t === "diffusion_pde_dataset" ||
      t === "reaction_diffusion_dataset" ||
      t === "advection_dataset" ||
      t === "modular_addition_dataset" ||
      TOY_LANGUAGE_DATASET_TYPE_SET.has(t)
    ) {
      if (!datasetIds.includes(c.nodeId)) datasetIds.push(c.nodeId);
      const di = datasetIds.indexOf(c.nodeId);
      prefix = di === 0 ? "dataset" : "test";
    } else {
      prefix = TYPE_SHORT[t] ?? t.replace(/_model$/, "").replace(/_optimizer$/, "").replace(/_loss$/, "");
    }
    parts.push(`${prefix}.${c.key}=${formatScalar(c.value)}`);
  }
  return parts;
}

/**
 * Human-readable list of swept parameters for one combo (comma-separated; use
 * {@link formatTrainSeriesSweptLines} for trainer UI lines).
 */
export function formatTrainSeriesComboCaption(
  combo: TrainSeriesAssignment[],
  nodes: SerializedTrainNode[],
  sweptAxisIds: Set<string>,
): string {
  return formatTrainSeriesSweptLines(combo, nodes, sweptAxisIds).join(", ");
}
