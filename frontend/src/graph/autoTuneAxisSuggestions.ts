/**
 * Build coordinate-descent axis suggestions from live node data (numeric fields only).
 * Dataset nodes are excluded; trainer UI / curve blobs are skipped under known paths.
 */

import { nodeRegistryTypesWithFamily } from "./nodeRegistrySpec";

/** Dataset + input sources: excluded from auto-tune per product request. */
export const AUTO_TUNE_EXCLUDED_DATASET_TYPES = new Set<string>([
  "linear_dataset",
  "memorization_a_dataset",
  "memorization_b_dataset",
  "symbolic_func_dataset",
  "teacher_dataset",
  "token_prediction_dataset",
  "circle_random_walk_dataset",
  "circular_motion_dataset",
  "unigram_dataset",
  "bigram_low_rank_dataset",
  "in_context_associative_recall_dataset",
  "uniform_linear_motion_dataset",
  "modular_addition_dataset",
  "dataset_mixer",
  "dataset_mixer_b",
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
  "mnist_dataset",
  "gaussian_blob_dataset",
  "shape_world_dataset",
  "hole_counting_dataset",
  "random_input_distribution",
  "input_sampler",
]);

const ATOMIC_LAYER_NODE_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
]);

const FULL_MODEL_STRIP_NODE_TYPES = new Set([
  "mlp_model",
  "gated_mlp_model",
  "moe_mlp_model",
  "mlp_token_model",
  "gated_mlp_token_model",
  "moe_mlp_token_model",
  "kan_model",
  "residual_ln_model",
  "attention_only_model",
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
  "diffusion_score_model",
]);

/** Model, optimizer, trainer, and loss nodes whose numeric `data` fields may affect training. */
const AUTO_TUNE_NODE_TYPES = new Set<string>([
  ...ATOMIC_LAYER_NODE_TYPES,
  ...FULL_MODEL_STRIP_NODE_TYPES,
  "resnet_model",
  "vit_model",
  "numeric_transformer_model",
  "transformer_token_model",
  "transformer_multi_token_model",
  "combined_model",
  "tensor_splitter",
  "reshape",
  "flatten",
  "einsum",
  "softmax",
  "causal_mask",
  "tensor_add",
  "basic_calculator",
  "tensor_stack",
  "tensor_concat",
  "tensor_constant",
  "tensor_linspace",
  "elementwise_transform",
  "fake_tensor",
  "adam_optimizer",
  "adamw_optimizer",
  "sgd_optimizer",
  "signsgd_optimizer",
  "muon_optimizer",
  "shampoo_optimizer",
  "soap_optimizer",
  "lr_schedule",
  "mup_lr_schedule",
  "mup_initialization",
  "idnns_initialization",
  "trainer",
  "mse_loss",
  "l1_reg",
  "l2_reg",
  "cross_entropy_loss",
  "binary_cross_entropy_with_logits_loss",
  "diffusion_mse_loss",
  "kan_reg",
]);

const TRAINER_SKIP_EXACT = new Set([
  "trainingSteps",
  "logFrequency",
  "targetCurveStepTicks",
  "targetCurveLossHistory",
]);

const TRAINER_SKIP_PREFIXES = [
  "lossHistory",
  "testLossHistory",
  "stepTicks",
  "observableMetricHistories",
  "memoryCheckpoint",
  "hostTrainUi",
  "trainUi",
  "trainPauseCheckpoint",
  "trainCompleteCheckpoint",
  "trainSeriesPauseCtx",
  "lastAutoTune",
  "autoTune",
];

function shouldSkipTrainerPath(path: string): boolean {
  if (TRAINER_SKIP_EXACT.has(path)) return true;
  for (const p of TRAINER_SKIP_PREFIXES) {
    if (path === p || path.startsWith(`${p}.`) || path.startsWith(p)) return true;
  }
  return false;
}

function shouldSkipKey(key: string): boolean {
  return key.startsWith("__");
}

/** MoE-only fields stored on shared schemas (e.g. token MLP defaults include numExperts for all variants). */
const MOE_NODE_TYPES_WITH_EXPERTS = new Set<string>(nodeRegistryTypesWithFamily("moe_model"));

function pathLeafSegment(path: string): string {
  const parts = path.split(".").filter(Boolean);
  return parts.length ? parts[parts.length - 1]! : path;
}

function shouldSkipIrrelevantNumericPath(nodeType: string, path: string): boolean {
  if (pathLeafSegment(path) !== "numExperts") return false;
  return !MOE_NODE_TYPES_WITH_EXPERTS.has(nodeType);
}

export type AutoTuneAxisSuggestion = {
  key: string;
  nodeId: string;
  path: string;
  currentValue: number;
  /** Comma-separated defaults for the text field when the modal opens. */
  defaultCandidates: string;
};

function pathLooksLikeSeed(path: string): boolean {
  return path.toLowerCase().includes("seed");
}

function pathLooksIntish(path: string, currentValue: number): boolean {
  if (pathLooksLikeSeed(path)) return true;
  if (/(^|\.)(depth|width|inputdim|outputdim|dim|rank|ranks|head|heads|layer|layers|nlayer)/i.test(path)) {
    return true;
  }
  if (Number.isInteger(currentValue) && Math.abs(currentValue) >= 2) return true;
  const r = Math.round(currentValue);
  if (Math.abs(currentValue - r) < 1e-6 && Math.abs(r) >= 2) return true;
  return false;
}

/** Default comma-separated candidate list for lazy users; tuned to path name and scale. */
export function autoTuneDefaultCandidates(path: string, currentValue: number): string {
  if (!Number.isFinite(currentValue)) return "0, 1, 2";

  if (pathLooksLikeSeed(path)) {
    const s0 = Math.round(currentValue);
    const delta = Math.max(1, Math.floor(Math.abs(s0) * 0.25 + 256));
    const a = s0 - delta;
    const b = s0;
    const c = s0 + delta;
    return `${a}, ${b}, ${c}`;
  }

  if (pathLooksIntish(path, currentValue)) {
    const n = Math.max(1, Math.round(currentValue));
    const lo = Math.max(1, Math.round(n * 0.5));
    const hi = Math.max(1, Math.round(n * 1.5));
    const uniq = [...new Set([lo, n, hi])].sort((x, y) => x - y);
    return uniq.join(", ");
  }

  const x = currentValue;
  if (x === 0) return "0, 0.01, 0.1";

  const fmt = (v: number) => {
    const t = Math.abs(v);
    if (t >= 1e-3 && t < 1e6) return String(+v.toPrecision(4));
    return String(v);
  };

  if (x > 0) {
    return `${fmt(x * 0.5)}, ${fmt(x)}, ${fmt(x * 1.5)}`;
  }
  return `${fmt(x * 1.5)}, ${fmt(x)}, ${fmt(x * 0.5)}`;
}

const MAX_CRAWL_DEPTH = 14;
const MAX_AXES = 512;

type CrawlOut = { nodeId: string; path: string; currentValue: number };

function pushAxis(out: CrawlOut[], seen: Set<string>, nodeId: string, path: string, currentValue: number) {
  const key = `${nodeId}.${path}`;
  if (seen.has(key)) return;
  seen.add(key);
  out.push({ nodeId, path, currentValue });
}

function crawlNumericFields(
  nodeId: string,
  nodeType: string,
  path: string,
  value: unknown,
  depth: number,
  out: CrawlOut[],
  seen: Set<string>,
) {
  if (out.length >= MAX_AXES) return;
  if (depth > MAX_CRAWL_DEPTH) return;

  if (typeof value === "number" && Number.isFinite(value)) {
    if (nodeType === "trainer" && shouldSkipTrainerPath(path)) return;
    if (shouldSkipIrrelevantNumericPath(nodeType, path)) return;
    pushAxis(out, seen, nodeId, path, value);
    return;
  }

  if (Array.isArray(value)) {
    const nums = value.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
    if (nums.length > 0) {
      if (nodeType === "trainer" && shouldSkipTrainerPath(path)) return;
      if (shouldSkipIrrelevantNumericPath(nodeType, path)) return;
      pushAxis(out, seen, nodeId, path, nums[0]!);
    }
    return;
  }

  if (!value || typeof value !== "object") return;

  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (shouldSkipKey(k)) continue;
    const nextPath = path ? `${path}.${k}` : k;
    crawlNumericFields(nodeId, nodeType, nextPath, v, depth + 1, out, seen);
    if (out.length >= MAX_AXES) return;
  }
}

export function buildAutoTuneAxisSuggestions(
  nodes: ReadonlyArray<{ id: string; type?: string | null; data?: unknown }>,
): AutoTuneAxisSuggestion[] {
  const out: CrawlOut[] = [];
  const seen = new Set<string>();

  for (const n of nodes) {
    const t = String(n.type ?? "");
    if (!t || AUTO_TUNE_EXCLUDED_DATASET_TYPES.has(t)) continue;
    if (!AUTO_TUNE_NODE_TYPES.has(t)) continue;
    if (!n.data || typeof n.data !== "object") continue;
    crawlNumericFields(n.id, t, "", n.data as Record<string, unknown>, 0, out, seen);
    if (out.length >= MAX_AXES) break;
  }

  out.sort((a, b) => {
    const c = a.nodeId.localeCompare(b.nodeId);
    if (c !== 0) return c;
    return a.path.localeCompare(b.path);
  });

  return out.map((row) => ({
    key: `${row.nodeId}.${row.path}`,
    nodeId: row.nodeId,
    path: row.path,
    currentValue: row.currentValue,
    defaultCandidates: autoTuneDefaultCandidates(row.path, row.currentValue),
  }));
}
