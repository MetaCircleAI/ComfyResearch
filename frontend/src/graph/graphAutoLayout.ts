import type { Edge, Node } from "@xyflow/react";

type LayoutNode = {
  node: Node;
  order: number;
  width: number;
  height: number;
  baseRank: number;
};

type Lane = "main" | "bottom" | "notes";

type BottomDirectedEdge = {
  source: string;
  target: string;
  gap: number;
};

type BottomLayoutUnit =
  | {
      kind: "chain";
      items: LayoutNode[];
      minRank: number;
      minY: number;
      minOrder: number;
    }
  | {
      kind: "loose";
      items: LayoutNode[];
      minRank: number;
      minY: number;
      minOrder: number;
      lane: "observables" | "analysis";
    };

export type GraphAutoLayoutResult = {
  nodes: Node[];
  changed: boolean;
  laidOutNodeIds: string[];
};

const COL_GAP = 170;
const ROW_GAP = 72;
const BOTTOM_LANE_GAP = 160;
const BOTTOM_COL_GAP = 72;
const BOTTOM_ROW_GAP = 96;
const NOTES_LANE_GAP = 170;
const GRID = 20;
const DEFAULT_NODE_W = 320;
const DEFAULT_NODE_H = 180;
const MIN_NODE_W = 220;
const MIN_NODE_H = 90;

const DATASET_TYPES = new Set([
  "linear_dataset",
  "random_noise_dataset",
  "memorization_a_dataset",
  "memorization_b_dataset",
  "symbolic_func_dataset",
  "token_prediction_dataset",
  "circle_random_walk_dataset",
  "circular_motion_dataset",
  "kepler_2d_dataset",
  "unigram_dataset",
  "bigram_low_rank_dataset",
  "random_input_distribution",
  "input_sampler",
  "teacher_dataset",
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
  "diffusion_pde_dataset",
  "reaction_diffusion_dataset",
  "advection_dataset",
]);

const MODEL_TYPES = new Set([
  "mlp_model",
  "gated_mlp_model",
  "moe_mlp_model",
  "mlp_token_model",
  "gated_mlp_token_model",
  "moe_mlp_token_model",
  "numeric_transformer_model",
  "numeric_hyena_model",
  "mpp_spatiotemporal_model",
  "afno_lite_spatiotemporal_model",
  "transformer_token_model",
  "transformer_multi_token_model",
  "kan_model",
  "residual_ln_model",
  "attention_only_model",
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
  "diffusion_score_model",
  "resnet_model",
  "vit_model",
  "combined_model",
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
  "afno_patch_embed_layer",
  "afno_spectral_mixer_layer",
  "afno_encoder_block_layer",
  "afno_patch_decode_layer",
  "crl_residual_mlp",
]);

const OPTIMIZER_TYPES = new Set([
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
]);

const LOSS_TYPES = new Set([
  "mse_loss",
  "cross_entropy_loss",
  "binary_cross_entropy_with_logits_loss",
  "diffusion_mse_loss",
  "l1_reg",
  "l2_reg",
  "l2_projection",
  "kan_reg",
  "crl_env_config",
]);

const TRAINER_TYPES = new Set(["trainer", "crl_trainer"]);

const VISUALIZATION_TYPES = new Set([
  "observable_viz",
  "training_visualization",
  "image_dataset_displayer",
  "protein_structure_displayer",
  "protein_structure_comparison_viz",
  "interatomic_eval_viz",
  "docking_pose_viz",
  "agent_trace_viz",
  "tensor_viz_0d",
  "tensor_viz_general",
  "tensor_viz_1d",
  "tensor_viz_2d",
  "tensor_viz_scatter",
  "visualize_kan",
  "sweep_data_table",
  "table_viz",
  "tensor_reader",
]);

const TENSOR_TOOL_TYPES = new Set([
  "activation",
  "pca",
  "svd",
  "statistics",
  "statistics2",
  "tensor_add",
  "tensor_stack",
  "tensor_concat",
  "basic_calculator",
  "tensor_constant",
  "tensor_linspace",
  "elementwise_transform",
  "fake_tensor",
  "model_weight_tensors",
  "effective_rank",
  "series_endpoint_gap",
  "smoothing_curve",
  "derivative_curve",
  "prediction",
  "tensor_selector",
  "dimension_permutator",
  "tensor_slicing",
  "tensor_splitter",
  "reshape",
  "flatten",
  "einsum",
  "softmax",
  "causal_mask",
  "regressor",
  "curve_annotator",
  "shape_checker",
]);

function nodeType(node: Node): string {
  return String(node.type ?? "");
}

function readNumeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function readNodeSize(node: Node): { width: number; height: number } {
  const raw = node as Node & {
    measured?: { width?: number; height?: number };
    width?: number;
    height?: number;
  };
  const style = (node.style ?? {}) as Record<string, unknown>;
  const width =
    readNumeric(raw.measured?.width) ??
    readNumeric(raw.width) ??
    readNumeric(style.width) ??
    readNumeric(style.minWidth) ??
    DEFAULT_NODE_W;
  const height =
    readNumeric(raw.measured?.height) ??
    readNumeric(raw.height) ??
    readNumeric(style.height) ??
    readNumeric(style.minHeight) ??
    DEFAULT_NODE_H;
  return {
    width: Math.max(MIN_NODE_W, Math.ceil(width)),
    height: Math.max(MIN_NODE_H, Math.ceil(height)),
  };
}

function isTopLevelVisible(node: Node): boolean {
  return node.hidden !== true && (node.parentId == null || node.parentId === "");
}

function isObservableNodeType(type: string): boolean {
  return type === "observable" || (type.startsWith("observable_") && !VISUALIZATION_TYPES.has(type));
}

function isBottomLaneType(type: string): boolean {
  return isObservableNodeType(type) || VISUALIZATION_TYPES.has(type);
}

function canPromoteToBottomLane(type: string): boolean {
  return isBottomLaneType(type) || TENSOR_TOOL_TYPES.has(type);
}

function laneForType(type: string): Lane {
  if (type === "comment" || type === "hypothesis" || type === "url_node" || type === "graph_assist_failure_overlay") {
    return "notes";
  }
  return isBottomLaneType(type) ? "bottom" : "main";
}

function baseRankForType(type: string): number {
  if (DATASET_TYPES.has(type)) return 0;
  if (MODEL_TYPES.has(type)) return 1;
  if (TENSOR_TOOL_TYPES.has(type)) return 2;
  if (OPTIMIZER_TYPES.has(type) || LOSS_TYPES.has(type)) return 3;
  if (TRAINER_TYPES.has(type)) return 4;
  if (isBottomLaneType(type)) return 5;
  return 2;
}

function sortKeyForType(type: string): number {
  if (DATASET_TYPES.has(type)) return 0;
  if (MODEL_TYPES.has(type)) return 10;
  if (TENSOR_TOOL_TYPES.has(type)) return 20;
  if (OPTIMIZER_TYPES.has(type)) return 30;
  if (LOSS_TYPES.has(type)) return 40;
  if (TRAINER_TYPES.has(type)) return 50;
  if (isObservableNodeType(type)) return 60;
  if (VISUALIZATION_TYPES.has(type)) return 70;
  return 90;
}

function snap(value: number): number {
  return Math.round(value / GRID) * GRID;
}

function compareStable(a: LayoutNode, b: LayoutNode): number {
  const at = nodeType(a.node);
  const bt = nodeType(b.node);
  return (
    sortKeyForType(at) - sortKeyForType(bt) ||
    a.node.position.y - b.node.position.y ||
    a.node.position.x - b.node.position.x ||
    a.order - b.order ||
    a.node.id.localeCompare(b.node.id)
  );
}

function computeMainRanks(main: LayoutNode[], edges: Edge[]): Map<string, number> {
  const byId = new Map(main.map((item) => [item.node.id, item]));
  const rank = new Map(main.map((item) => [item.node.id, item.baseRank]));
  const outgoing = new Map<string, string[]>();
  const indegree = new Map(main.map((item) => [item.node.id, 0]));
  const seen = new Set<string>();

  for (const edge of edges) {
    if (!byId.has(edge.source) || !byId.has(edge.target) || edge.source === edge.target) continue;
    const key = `${edge.source}->${edge.target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const list = outgoing.get(edge.source) ?? [];
    list.push(edge.target);
    outgoing.set(edge.source, list);
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1);
  }

  const queue = main
    .filter((item) => (indegree.get(item.node.id) ?? 0) === 0)
    .sort(compareStable)
    .map((item) => item.node.id);
  const processed = new Set<string>();

  while (queue.length) {
    const id = queue.shift()!;
    if (processed.has(id)) continue;
    processed.add(id);
    const nextRank = (rank.get(id) ?? 0) + 1;
    for (const target of outgoing.get(id) ?? []) {
      rank.set(target, Math.max(rank.get(target) ?? 0, nextRank));
      const d = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, d);
      if (d === 0) queue.push(target);
    }
    queue.sort((a, b) => compareStable(byId.get(a)!, byId.get(b)!));
  }

  return rank;
}

function relatedMainRank(nodeId: string, edges: Edge[], mainRanks: Map<string, number>): number | null {
  const related: number[] = [];
  for (const edge of edges) {
    if (edge.source === nodeId) {
      const r = mainRanks.get(edge.target);
      if (r != null) related.push(r);
    }
    if (edge.target === nodeId) {
      const r = mainRanks.get(edge.source);
      if (r != null) related.push(r);
    }
  }
  if (!related.length) return null;
  return Math.round(related.reduce((sum, v) => sum + v, 0) / related.length);
}

function readStringData(node: Node, key: string): string | null {
  const data = (node.data ?? {}) as Record<string, unknown>;
  const value = data[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function computeBottomLaneIds(layoutNodes: LayoutNode[], edges: Edge[]): Set<string> {
  const byId = new Map(layoutNodes.map((item) => [item.node.id, item]));
  const bottomIds = new Set<string>();

  for (const item of layoutNodes) {
    if (laneForType(nodeType(item.node)) === "bottom") {
      bottomIds.add(item.node.id);
    }
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of edges) {
      const source = byId.get(edge.source);
      const target = byId.get(edge.target);
      if (!source || !target || source.node.id === target.node.id) continue;

      if (bottomIds.has(source.node.id) && canPromoteToBottomLane(nodeType(target.node)) && !bottomIds.has(target.node.id)) {
        bottomIds.add(target.node.id);
        changed = true;
      }
      if (bottomIds.has(target.node.id) && canPromoteToBottomLane(nodeType(source.node)) && !bottomIds.has(source.node.id)) {
        bottomIds.add(source.node.id);
        changed = true;
      }
    }

    for (const item of layoutNodes) {
      if (nodeType(item.node) !== "observable_viz") continue;
      const pairedObservableId = readStringData(item.node, "pairedObservableId");
      if (!pairedObservableId || !byId.has(pairedObservableId)) continue;
      if (bottomIds.has(item.node.id) && !bottomIds.has(pairedObservableId)) {
        bottomIds.add(pairedObservableId);
        changed = true;
      }
      if (bottomIds.has(pairedObservableId) && !bottomIds.has(item.node.id)) {
        bottomIds.add(item.node.id);
        changed = true;
      }
    }
  }

  return bottomIds;
}

function buildBottomRelations(
  bottom: LayoutNode[],
  edges: Edge[],
): { directed: BottomDirectedEdge[]; undirected: Array<[string, string]> } {
  const bottomIds = new Set(bottom.map((item) => item.node.id));
  const directed: BottomDirectedEdge[] = [];
  const undirected: Array<[string, string]> = [];
  const seenDirected = new Set<string>();
  const seenUndirected = new Set<string>();

  const addDirected = (source: string, target: string, gap: number) => {
    if (source === target || !bottomIds.has(source) || !bottomIds.has(target)) return;
    const key = `${source}->${target}:${gap}`;
    if (!seenDirected.has(key)) {
      directed.push({ source, target, gap });
      seenDirected.add(key);
    }
    const ukey = source < target ? `${source}|${target}` : `${target}|${source}`;
    if (!seenUndirected.has(ukey)) {
      undirected.push([source, target]);
      seenUndirected.add(ukey);
    }
  };

  for (const edge of edges) {
    addDirected(edge.source, edge.target, 1);
  }

  for (const item of bottom) {
    if (nodeType(item.node) !== "observable_viz") continue;
    const pairedObservableId = readStringData(item.node, "pairedObservableId");
    if (!pairedObservableId) continue;
    // Leave the trainer column between an observable source and its paired viz.
    addDirected(pairedObservableId, item.node.id, 2);
  }

  return { directed, undirected };
}

function seedBottomRank(
  item: LayoutNode,
  edges: Edge[],
  byId: Map<string, LayoutNode>,
  mainRanks: Map<string, number>,
): number {
  const candidates: number[] = [];
  for (const edge of edges) {
    if (edge.source === item.node.id) {
      const target = byId.get(edge.target);
      const targetRank = mainRanks.get(edge.target);
      if (target && targetRank != null) {
        candidates.push(TRAINER_TYPES.has(nodeType(target.node)) ? targetRank - 1 : targetRank + 1);
      }
    }
    if (edge.target === item.node.id) {
      const source = byId.get(edge.source);
      const sourceRank = mainRanks.get(edge.source);
      if (source && sourceRank != null) {
        candidates.push(sourceRank + 1);
      }
    }
  }
  if (candidates.length) {
    return Math.max(0, Math.round(candidates.reduce((sum, value) => sum + value, 0) / candidates.length));
  }
  return baseRankForType(nodeType(item.node));
}

function computeBottomRanks(
  bottom: LayoutNode[],
  edges: Edge[],
  byId: Map<string, LayoutNode>,
  mainRanks: Map<string, number>,
  directed: BottomDirectedEdge[],
): Map<string, number> {
  const bottomById = new Map(bottom.map((item) => [item.node.id, item]));
  const rank = new Map(bottom.map((item) => [item.node.id, seedBottomRank(item, edges, byId, mainRanks)]));
  const outgoing = new Map<string, BottomDirectedEdge[]>();
  const indegree = new Map(bottom.map((item) => [item.node.id, 0]));

  for (const rel of directed) {
    if (!bottomById.has(rel.source) || !bottomById.has(rel.target)) continue;
    const list = outgoing.get(rel.source) ?? [];
    list.push(rel);
    outgoing.set(rel.source, list);
    indegree.set(rel.target, (indegree.get(rel.target) ?? 0) + 1);
  }

  const queue = bottom
    .filter((item) => (indegree.get(item.node.id) ?? 0) === 0)
    .sort(compareStable)
    .map((item) => item.node.id);
  const processed = new Set<string>();

  while (queue.length) {
    const id = queue.shift()!;
    if (processed.has(id)) continue;
    processed.add(id);
    for (const rel of outgoing.get(id) ?? []) {
      rank.set(rel.target, Math.max(rank.get(rel.target) ?? 0, (rank.get(id) ?? 0) + rel.gap));
      const d = (indegree.get(rel.target) ?? 0) - 1;
      indegree.set(rel.target, d);
      if (d === 0) queue.push(rel.target);
    }
    queue.sort((a, b) => compareStable(bottomById.get(a)!, bottomById.get(b)!));
  }

  return rank;
}

function connectedBottomComponents(bottom: LayoutNode[], undirected: Array<[string, string]>): LayoutNode[][] {
  const byId = new Map(bottom.map((item) => [item.node.id, item]));
  const adjacency = new Map(bottom.map((item) => [item.node.id, [] as string[]]));
  for (const [a, b] of undirected) {
    if (!byId.has(a) || !byId.has(b)) continue;
    adjacency.get(a)?.push(b);
    adjacency.get(b)?.push(a);
  }

  const seen = new Set<string>();
  const components: LayoutNode[][] = [];
  for (const item of bottom.sort(compareStable)) {
    if (seen.has(item.node.id)) continue;
    const stack = [item.node.id];
    const ids: string[] = [];
    seen.add(item.node.id);
    while (stack.length) {
      const id = stack.pop()!;
      ids.push(id);
      for (const next of adjacency.get(id) ?? []) {
        if (seen.has(next)) continue;
        seen.add(next);
        stack.push(next);
      }
    }
    components.push(ids.map((id) => byId.get(id)!).sort(compareStable));
  }
  return components;
}

function minRankForItems(items: LayoutNode[], ranks: Map<string, number>): number {
  return Math.min(...items.map((item) => ranks.get(item.node.id) ?? item.baseRank));
}

function minYForItems(items: LayoutNode[]): number {
  return Math.min(...items.map((item) => item.node.position.y));
}

function minOrderForItems(items: LayoutNode[]): number {
  return Math.min(...items.map((item) => item.order));
}

function relatedPlacedY(
  nodeId: string,
  edges: Edge[],
  positions: Map<string, { x: number; y: number }>,
): number | null {
  const ys: number[] = [];
  for (const edge of edges) {
    if (edge.source === nodeId) {
      const p = positions.get(edge.target);
      if (p) ys.push(p.y);
    }
    if (edge.target === nodeId) {
      const p = positions.get(edge.source);
      if (p) ys.push(p.y);
    }
  }
  if (!ys.length) return null;
  return Math.min(...ys);
}

function putPosition(out: Map<string, { x: number; y: number }>, item: LayoutNode, x: number, y: number) {
  out.set(item.node.id, { x: snap(x), y: snap(y) });
}

export function layoutResearchGraphNodes(nodes: Node[], edges: Edge[]): GraphAutoLayoutResult {
  const layoutNodes: LayoutNode[] = [];
  nodes.forEach((node, order) => {
    if (!isTopLevelVisible(node)) return;
    const size = readNodeSize(node);
    layoutNodes.push({
      node,
      order,
      width: size.width,
      height: size.height,
      baseRank: baseRankForType(nodeType(node)),
    });
  });

  if (layoutNodes.length === 0) {
    return { nodes, changed: false, laidOutNodeIds: [] };
  }

  const minX = Math.min(...layoutNodes.map((item) => item.node.position.x));
  const minY = Math.min(...layoutNodes.map((item) => item.node.position.y));
  const anchorX = snap(minX);
  const anchorY = snap(minY);

  const bottomLaneIds = computeBottomLaneIds(layoutNodes, edges);
  const main = layoutNodes.filter((item) => laneForType(nodeType(item.node)) === "main" && !bottomLaneIds.has(item.node.id));
  const bottom = layoutNodes.filter((item) => bottomLaneIds.has(item.node.id));
  const notes = layoutNodes.filter((item) => laneForType(nodeType(item.node)) === "notes");
  const nextPositions = new Map<string, { x: number; y: number }>();
  const layoutById = new Map(layoutNodes.map((item) => [item.node.id, item]));

  const mainRanks = computeMainRanks(main, edges);
  const rankValues = new Set<number>(main.map((item) => mainRanks.get(item.node.id) ?? item.baseRank));
  if (rankValues.size === 0) rankValues.add(0);

  const orderedRanks = [...rankValues].sort((a, b) => a - b);
  const itemsByRank = new Map<number, LayoutNode[]>();
  for (const rank of orderedRanks) itemsByRank.set(rank, []);
  for (const item of main) {
    const rank = mainRanks.get(item.node.id) ?? item.baseRank;
    const list = itemsByRank.get(rank) ?? [];
    list.push(item);
    itemsByRank.set(rank, list);
  }

  const xByRank = new Map<number, number>();
  const colWidthByRank = new Map<number, number>();
  let cursorX = anchorX;
  for (const rank of orderedRanks) {
    const colItems = itemsByRank.get(rank) ?? [];
    const colWidth = Math.max(DEFAULT_NODE_W, ...colItems.map((item) => item.width));
    xByRank.set(rank, cursorX);
    colWidthByRank.set(rank, colWidth);
    cursorX += colWidth + COL_GAP;
  }

  let mainBottom = anchorY;
  for (const rank of orderedRanks) {
    const colItems = (itemsByRank.get(rank) ?? []).sort(compareStable);
    let y = anchorY;
    for (const item of colItems) {
      putPosition(nextPositions, item, xByRank.get(rank) ?? anchorX, y);
      y += item.height + ROW_GAP;
    }
    mainBottom = Math.max(mainBottom, y - ROW_GAP);
  }

  let mainRight = anchorX + DEFAULT_NODE_W;
  for (const rank of orderedRanks) {
    const x = xByRank.get(rank) ?? anchorX;
    const w = colWidthByRank.get(rank) ?? DEFAULT_NODE_W;
    mainRight = Math.max(mainRight, x + w);
  }
  if (main.length === 0 && bottom.length > 0) {
    const firstRowEstimate = Math.min(4, bottom.length);
    const averageWidth =
      bottom.reduce((sum, item) => sum + item.width, 0) / Math.max(1, bottom.length);
    mainRight = anchorX + Math.max(DEFAULT_NODE_W * 2, averageWidth * firstRowEstimate + BOTTOM_COL_GAP * (firstRowEstimate - 1));
  }

  let bottomBottom = mainBottom;
  let bottomRight = mainRight;
  if (bottom.length) {
    const relations = buildBottomRelations(bottom, edges);
    const bottomRanks = computeBottomRanks(bottom, edges, layoutById, mainRanks, relations.directed);
    const components = connectedBottomComponents(bottom, relations.undirected);
    const bottomRankWidths = new Map<number, number>();
    for (const item of bottom) {
      const rank = bottomRanks.get(item.node.id) ?? item.baseRank;
      bottomRankWidths.set(rank, Math.max(bottomRankWidths.get(rank) ?? DEFAULT_NODE_W, item.width));
    }
    const minMainRank = orderedRanks[0] ?? 0;
    const maxMainRank = orderedRanks[orderedRanks.length - 1] ?? minMainRank;
    const xForBottomRank = (rank: number): number => {
      const direct = xByRank.get(rank);
      if (direct != null) return direct;
      if (rank > maxMainRank) {
        let x = mainRight + COL_GAP;
        for (let r = maxMainRank + 1; r < rank; r += 1) {
          x += Math.max(DEFAULT_NODE_W, bottomRankWidths.get(r) ?? DEFAULT_NODE_W) + BOTTOM_COL_GAP;
        }
        return x;
      }
      if (rank < minMainRank) {
        return anchorX - (minMainRank - rank) * (DEFAULT_NODE_W + COL_GAP);
      }
      const prevRank = [...orderedRanks].filter((r) => r < rank).sort((a, b) => b - a)[0];
      if (prevRank != null) {
        return (xByRank.get(prevRank) ?? anchorX) + (colWidthByRank.get(prevRank) ?? DEFAULT_NODE_W) + COL_GAP;
      }
      return anchorX;
    };

    const chainUnits: BottomLayoutUnit[] = [];
    const looseObservableItems: LayoutNode[] = [];
    const looseAnalysisItems: LayoutNode[] = [];

    for (const component of components) {
      if (component.length <= 1) {
        const item = component[0]!;
        if (isObservableNodeType(nodeType(item.node))) {
          looseObservableItems.push(item);
        } else {
          looseAnalysisItems.push(item);
        }
        continue;
      }
      chainUnits.push({
        kind: "chain",
        items: component,
        minRank: minRankForItems(component, bottomRanks),
        minY: minYForItems(component),
        minOrder: minOrderForItems(component),
      });
    }

    const units: BottomLayoutUnit[] = [...chainUnits];
    if (looseObservableItems.length) {
      const items = looseObservableItems.sort(compareStable);
      units.push({
        kind: "loose",
        lane: "observables",
        items,
        minRank: minRankForItems(items, bottomRanks),
        minY: minYForItems(items),
        minOrder: minOrderForItems(items),
      });
    }
    if (looseAnalysisItems.length) {
      const items = looseAnalysisItems.sort(compareStable);
      units.push({
        kind: "loose",
        lane: "analysis",
        items,
        minRank: minRankForItems(items, bottomRanks),
        minY: minYForItems(items),
        minOrder: minOrderForItems(items),
      });
    }
    units.sort(
      (a, b) =>
        a.minRank - b.minRank ||
        a.minY - b.minY ||
        a.minOrder - b.minOrder ||
        (a.kind === b.kind ? 0 : a.kind === "chain" ? -1 : 1),
    );

    const availableWidth = Math.max(DEFAULT_NODE_W * 2, mainRight - anchorX);

    const packRows = (items: LayoutNode[], startX: number, startY: number): number => {
      let x = startX;
      let y = startY;
      let rowHeight = 0;
      for (const item of items) {
        const wouldOverflow = x > startX && x + item.width > startX + availableWidth;
        if (wouldOverflow) {
          x = startX;
          y += rowHeight + BOTTOM_ROW_GAP;
          rowHeight = 0;
        }
        putPosition(nextPositions, item, x, y);
        bottomRight = Math.max(bottomRight, x + item.width);
        x += item.width + BOTTOM_COL_GAP;
        rowHeight = Math.max(rowHeight, item.height);
      }
      return y + rowHeight;
    };

    let y = mainBottom + BOTTOM_LANE_GAP;
    for (const unit of units) {
      if (unit.kind === "loose") {
        const startX = unit.lane === "observables" ? anchorX : xForBottomRank(unit.minRank);
        y = packRows(unit.items, startX, y);
        y += BOTTOM_ROW_GAP;
        continue;
      }

      const ranks = [...new Set(unit.items.map((item) => bottomRanks.get(item.node.id) ?? item.baseRank))].sort(
        (a, b) => a - b,
      );
      let groupHeight = 0;
      for (const rank of ranks) {
        const colItems = unit.items
          .filter((item) => (bottomRanks.get(item.node.id) ?? item.baseRank) === rank)
          .sort(compareStable);
        const x = xForBottomRank(rank);
        let colY = y;
        for (const item of colItems) {
          putPosition(nextPositions, item, x, colY);
          bottomRight = Math.max(bottomRight, x + item.width);
          groupHeight = Math.max(groupHeight, colY + item.height - y);
          colY += item.height + ROW_GAP;
        }
      }
      y += groupHeight + BOTTOM_ROW_GAP;
    }
    bottomBottom = Math.max(bottomBottom, y - BOTTOM_ROW_GAP);
  }

  if (notes.length) {
    const notesX = Math.max(mainRight, bottomRight) + NOTES_LANE_GAP;
    let y = anchorY;
    const sortedNotes = notes
      .map((item) => ({
        item,
        preferredY: relatedPlacedY(item.node.id, edges, nextPositions) ?? item.node.position.y,
      }))
      .sort(
        (a, b) =>
          a.preferredY - b.preferredY ||
          compareStable(a.item, b.item),
      );
    for (const { item, preferredY } of sortedNotes) {
      y = Math.max(y, preferredY);
      putPosition(nextPositions, item, notesX, y);
      y += item.height + ROW_GAP;
    }
    bottomBottom = Math.max(bottomBottom, y - ROW_GAP);
  }

  const laidOutNodeIds = [...nextPositions.keys()];
  let changed = false;
  const nextNodes = nodes.map((node) => {
    const pos = nextPositions.get(node.id);
    if (!pos) return node;
    if (Math.abs(node.position.x - pos.x) < 0.5 && Math.abs(node.position.y - pos.y) < 0.5) {
      return node;
    }
    changed = true;
    return { ...node, position: pos };
  });

  void bottomBottom;
  return { nodes: nextNodes, changed, laidOutNodeIds };
}
