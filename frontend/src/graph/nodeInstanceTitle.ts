import type { Edge as RFEdge, Node as RFNode } from "@xyflow/react";
import { centroidClipAbsolute, clipAbsolutePosition } from "./combineNodesUtils";

/** Canvas-only display title; persisted in node `data` and shown in the node header. */
export const INSTANCE_TITLE_KEY = "instanceTitle" as const;

const DISPLAY_BASE_OVERRIDES: Record<string, string> = {
  tensor_viz_general: "Tensor viz",
  tensor_viz_1d: "Tensor viz 1D",
  tensor_viz_2d: "Tensor viz 2D",
  mlp_model: "MLP",
  gated_mlp_model: "Gated MLP",
  moe_mlp_model: "MoE MLP",
  gated_mlp_token_model: "Gated MLP_token model",
  moe_mlp_token_model: "MoE MLP_token model",
  kan_model: "KAN",
  visualize_kan: "Visualize KAN",
  pca: "PCA",
  svd: "SVD",
  adam_optimizer: "Adam",
  adamw_optimizer: "AdamW",
  sgd_optimizer: "SGD",
  signsgd_optimizer: "SignSGD",
  muon_optimizer: "Muon",
  shampoo_optimizer: "Shampoo",
  soap_optimizer: "SOAP",
  lr_schedule: "LR schedule",
  cyclic_lr_schedule: "Cyclic LR schedule",
  cyclic_batch_schedule: "Cyclic batch schedule",
  mup_lr_schedule: "MuP LR schedule",
  mup_initialization: "MuP initialization",
  idnns_initialization: "IDNNs initialization",
  mse_loss: "MSE loss",
  l1_reg: "L1 reg",
  l2_reg: "L2 reg",
  l2_projection: "L2 projection",
  kan_reg: "KAN reg",
  cross_entropy_loss: "Cross-entropy loss",
  binary_cross_entropy_with_logits_loss: "Binary cross-entropy (logits)",
  training_visualization: "Training viz",
  image_dataset_displayer: "Image dataset displayer",
  sweep_data_table: "Sweep data table",
  table_viz: "Table viz",
  curve_series_table: "Curve series table",
  curve_series_viz: "Curve series viz",
  parametric_path_sampler: "Parametric path sampler",
  shape_checker: "Shape checker",
  tensor_reader: "Tensor reader",
  tensor_viz_0d: "0D tensor viz",
  model_checkpoint: "Model checkpoint",
  crl_trainer: "CRL trainer",
  crl_env_config: "CRL environment",
  crl_residual_mlp: "CRL residual MLP",
  curve_annotator: "Curve annotator",
  url_node: "URL",
  model_weight_tensors: "Model weight tensors",
  effective_rank: "Effective rank",
  series_endpoint_gap: "Series endpoint gap",
  smoothing_curve: "Smoothing curve",
  derivative_curve: "Derivative curve",
  dimension_permutator: "Dimension permutator",
  tensor_slicing: "Tensor slicing",
  observable_relu_nonlinear_count: "ReLU nonlinear count",
  observable_embedding_evolution: "Embedding evolution",
  observable_embedding_trajectory: "Embedding trajectory",
  observable_capacity: "Capacity",
  observable_weight_l1: "Weight L1",
  observable_weight_l2: "Weight L2",
  observable_hessian_eigenvalues: "Hessian eigenvalues",
  observable_gradient_norm: "Gradient norm",
  observable_train_test_gap: "Train vs test gap",
  observable_activation_stats: "Activation mean/std",
  observable_embedding_effective_rank: "Embedding effective rank",
  observable_embedding_feature_drift: "Embedding feature drift",
  observable_sink_attention_mass: "Sink attention mass",
  observable_attention_entropy_mean: "Attention entropy (mean)",
  observable_attention_max_weight_mean: "Attention max weight (mean)",
  observable_attention_head_sink_max: "Attention head sink max",
  observable_attention_position_bias_ratio: "Attention position bias ratio",
  observable_activation_norm_mean: "Activation norm (mean)",
  observable_activation_outlier_ratio: "Activation outlier ratio",
  observable_user: "User observable",
  observable_viz: "Observable viz",
  tensor_selector: "Tensor selector",
  tensor_add: "Tensor add",
  tensor_stack: "Tensor stack",
  tensor_concat: "Tensor concat",
  basic_calculator: "Basic calculator",
  tensor_constant: "Tensor constant",
  tensor_linspace: "Tensor linspace",
  fake_tensor: "Fake tensor",
  token_prediction_dataset: "Token Retrieval (position) dataset",
  circle_random_walk_dataset: "Circle random walk dataset",
  circular_motion_dataset: "Circular motion dataset",
  kepler_2d_dataset: "Kepler 2D dataset",
  unigram_dataset: "Unigram dataset",
  bigram_low_rank_dataset: "Bigram low-rank dataset",
  random_input_distribution: "Random input distribution",
  input_sampler: "Input sampler",
  attention_only_model: "Attention layer",
  linear_attention_model: "Linear attention LM",
  diagonal_ssm_token_model: "Diagonal SSM LM",
  rwkv_time_mix_token_model: "RWKV-lite LM",
  hyena_like_conv_model: "Hyena-like conv LM",
  slot_attention_token_model: "Slot attention LM",
  diffusion_score_model: "Diffusion score MLP",
  diffusion_mse_loss: "Diffusion MSE loss",
  mlp_token_model: "MLP_token model",
  symbolic_func_dataset: "Symbolic func dataset",
  linear_dataset: "Linear Dataset",
  random_noise_dataset: "Random noise dataset",
  memorization_a_dataset: "Memorization A dataset",
  memorization_b_dataset: "Memorization B dataset",
  teacher_dataset: "Teacher Dataset",
  in_context_associative_recall_dataset: "In-context associative recall dataset",
  uniform_linear_motion_dataset: "Uniform linear motion dataset",
  modular_addition_dataset: "Modular addition dataset",
  pcfg_dataset: "PCFG toy LM dataset",
  dyck_dataset: "Dyck language dataset",
  ngram_language_dataset: "N-gram language dataset",
  formal_language_suite_dataset: "Formal language suite",
  scan_dataset: "SCAN toy dataset",
  cogs_dataset: "COGS-style toy dataset",
  listops_dataset: "ListOps toy dataset",
  tinystories_dataset: "TinyStories-style dataset",
  phi1_style_dataset: "phi-1 style dataset",
  biography_lm_dataset: "Biography synthetic LM",
  relation_tuple_dataset: "Relation tuple synthetic LM",
  synthetic_playground_dataset: "Synthetic playground LM",
  multi_hop_fact_chain_dataset: "Multi-hop fact chain LM",
  mnist_dataset: "MNIST dataset",
  cifar10_dataset: "CIFAR-10 dataset",
  gaussian_blob_dataset: "Gaussian blob dataset",
  shape_world_dataset: "Shape world dataset",
  hole_counting_dataset: "Hole counting dataset",
  diffusion_pde_dataset: "Diffusion PDE dataset",
  reaction_diffusion_dataset: "Reaction–diffusion dataset",
  advection_dataset: "Advection dataset",
  mpp_spatiotemporal_model: "MPP spatiotemporal ViT",
  afno_lite_spatiotemporal_model: "AFNO-lite spatiotemporal model",
  resnet_model: "ResNet",
  keskar_c1_c2_cnn_model: "Keskar C1/C2 CNN",
  vgg11_cifar_model: "VGG-11 (CIFAR)",
  small_inception_cifar_model: "Small Inception (CIFAR)",
  vit_model: "ViT",
  residual_ln_model: "Residual LN model",
  numeric_transformer_model: "Transformer (numerics)",
  numeric_hyena_model: "Hyena-like LM (numerics)",
  transformer_token_model: "Transformer (tokens)",
  transformer_multi_token_model: "Transformer (multiple tokens)",
  combined_model: "Combined model",
  embedding_layer: "Embedding layer",
  unembedding_layer: "Unembedding layer",
  absolute_pos_embed_layer: "Absolute positional embedding",
  rotary_embed_layer: "Rotary positional embedding",
  local_mixing_layer: "Local mixing layer",
  afno_patch_embed_layer: "AFNO patch embed layer",
  afno_spectral_mixer_layer: "AFNO spectral mixer layer",
  afno_encoder_block_layer: "AFNO encoder block layer",
  afno_patch_decode_layer: "AFNO patch decode layer",
};

function humanizeSnake(type: string): string {
  return type
    .split("_")
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase() : ""))
    .join(" ");
}

export function defaultInstanceTitleBase(nodeType: string): string {
  return DISPLAY_BASE_OVERRIDES[nodeType] ?? humanizeSnake(nodeType);
}

function parseTrailingIndex(title: unknown): { base: string; index: number } | null {
  if (typeof title !== "string") return null;
  const m = title.trim().match(/^(.*)\s(\d+)$/);
  if (!m) return null;
  return { base: m[1]!.trimEnd(), index: Number.parseInt(m[2]!, 10) };
}

function titleBaseFromPrior(priorTitle: unknown, nodeType: string): string {
  const p = parseTrailingIndex(priorTitle);
  if (p) return p.base;
  return defaultInstanceTitleBase(nodeType);
}

/** Title for a newly added node: `<Base> <n>` where `n` is the current count of that node type (0-based). */
export function nextInstanceTitleForCreate(nodes: RFNode[], nodeType: string): string {
  const base = defaultInstanceTitleBase(nodeType);
  const idx = nodes.filter((n) => n.type === nodeType).length;
  return `${base} ${idx}`;
}

/** Next display index after the highest same-base suffix among nodes of this type (for paste / fork without sharing). */
export function nextInstanceTitleForDuplicate(
  allNodes: RFNode[],
  nodeType: string,
  priorTitle: unknown,
): string {
  const base = titleBaseFromPrior(priorTitle, nodeType);
  let maxIdx = -1;
  for (const n of allNodes) {
    if (n.type !== nodeType) continue;
    const raw = (n.data as Record<string, unknown> | undefined)?.[INSTANCE_TITLE_KEY];
    const p = parseTrailingIndex(raw);
    if (p && p.base === base) maxIdx = Math.max(maxIdx, p.index);
  }
  return `${base} ${maxIdx + 1}`;
}

export function readInstanceTitle(data: unknown, fallback: string): string {
  const v = (data as Record<string, unknown> | null | undefined)?.[INSTANCE_TITLE_KEY];
  if (typeof v === "string" && v.trim()) return v.trim();
  return fallback;
}

/** Older combined nodes used ``Combined model <n>`` while ``displayName`` holds the user label. */
export function migrateCombinedModelInstanceTitles(nodes: RFNode[]): RFNode[] {
  return nodes.map((n) => {
    if (n.type !== "combined_model") return n;
    const d = (n.data ?? {}) as Record<string, unknown>;
    const dn = typeof d.displayName === "string" ? d.displayName.trim() : "";
    if (!dn) return n;
    const title = d[INSTANCE_TITLE_KEY];
    if (typeof title !== "string") return n;
    const m = title.match(/^Combined model (\d+)$/i);
    if (!m) return n;
    return { ...n, data: { ...d, [INSTANCE_TITLE_KEY]: `${dn} ${m[1]}` } };
  });
}

export function withNewInstanceTitle(
  nodes: RFNode[],
  nodeType: string,
  data: Record<string, unknown>,
): Record<string, unknown> {
  if (nodeType === "observable_attention_relation_score") {
    return { ...data, [INSTANCE_TITLE_KEY]: `score${nodes.filter((n) => n.type === nodeType).length}` };
  }
  if (nodeType === "observable_viz") {
    const pairedId = typeof data.pairedObservableId === "string" ? data.pairedObservableId : "";
    const paired = nodes.find((n) => n.id === pairedId);
    if (paired?.type === "observable_attention_relation_score") {
      const scoreTitle = readInstanceTitle(paired.data, "");
      const scoreIndex = /^score(\d+)$/.exec(scoreTitle)?.[1];
      if (scoreIndex != null) return { ...data, [INSTANCE_TITLE_KEY]: `viz${scoreIndex}` };
    }
  }
  if (nodeType === "combined_model") {
    const dn = typeof data.displayName === "string" ? data.displayName.trim() : "";
    const base = dn || defaultInstanceTitleBase(nodeType);
    const idx = nodes.filter((n) => n.type === nodeType).length;
    return {
      ...data,
      [INSTANCE_TITLE_KEY]: `${base} ${idx}`,
    };
  }
  return {
    ...data,
    [INSTANCE_TITLE_KEY]: nextInstanceTitleForCreate(nodes, nodeType),
  };
}

export function appendResearchNode(
  existing: RFNode[],
  nodeType: string,
  position: { x: number; y: number },
  data: Record<string, unknown>,
  /** When set (e.g. planned subgraphs), wire edges to this id before the node exists in ``existing``. */
  nodeId?: string,
): RFNode {
  return {
    id: nodeId?.trim() ? nodeId.trim() : `${nodeType}-${Math.random().toString(36).slice(2, 10)}`,
    type: nodeType,
    position: { ...position },
    data: withNewInstanceTitle(existing, nodeType, data),
  };
}

export type ClipboardSubgraph = { nodes: RFNode[]; edges: RFEdge[] };

/** Deep-clone nodes + internal edges; remap ids; optionally assign fresh instance titles per type. */
export function cloneSubgraphForPaste(
  allNodes: RFNode[],
  clip: ClipboardSubgraph,
  anchor: { x: number; y: number },
  shareParams: boolean,
): { nodes: RFNode[]; edges: RFEdge[] } {
  const idMap = new Map<string, string>();
  for (const n of clip.nodes) {
    idMap.set(n.id, `${String(n.type)}-${Math.random().toString(36).slice(2, 10)}`);
  }
  const clipById = new Map(clip.nodes.map((n) => [n.id, n]));
  const clipIds = new Set(clip.nodes.map((n) => n.id));
  const c = centroidClipAbsolute(clip.nodes, clipById, clipIds);
  let pool: RFNode[] = [...allNodes];
  const nextNodes: RFNode[] = clip.nodes.map((n) => {
    const newId = idMap.get(n.id)!;
    const data = structuredClone((n.data ?? {}) as Record<string, unknown>);
    const t = String(n.type);
    if (!shareParams) {
      data[INSTANCE_TITLE_KEY] = nextInstanceTitleForDuplicate(pool, t, data[INSTANCE_TITLE_KEY]);
    }
    const parentInClip = Boolean(n.parentId && clipIds.has(String(n.parentId)));
    const abs = clipAbsolutePosition(n, clipById, clipIds);
    const position = parentInClip
      ? { x: n.position.x, y: n.position.y }
      : { x: anchor.x + (abs.x - c.x), y: anchor.y + (abs.y - c.y) };
    const nextParentId =
      n.parentId && clipIds.has(String(n.parentId)) ? idMap.get(String(n.parentId)) : undefined;
    const nn: RFNode = {
      ...n,
      id: newId,
      selected: true,
      position,
      data,
      parentId: nextParentId,
      extent: nextParentId ? n.extent : undefined,
    };
    pool = [...pool, nn];
    return nn;
  });
  const nextEdges: RFEdge[] = clip.edges.map((e) => ({
    ...e,
    id: `e-${Math.random().toString(36).slice(2, 12)}`,
    source: idMap.get(e.source) ?? e.source,
    target: idMap.get(e.target) ?? e.target,
  }));
  return { nodes: nextNodes, edges: nextEdges };
}
