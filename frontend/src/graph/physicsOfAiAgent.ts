import type { Edge, Node } from "@xyflow/react";
import { defaultObservableUserData } from "../components/nodes/observableUserDefaults";
import {
  buildRandomObservableDrafts,
  DEFAULT_RANDOM_GENERATION_PREFERENCES,
  type AlgebraObservableItem,
  type AxisReductionDraft,
  type ObservableFlattenMode,
  type ObservableSource,
  type ObservableTensorScope,
  type RepresentationEntry,
} from "../observables/observableAlgebra";
import { isObservableModelNodeType } from "../observables/modelNodeTypes";
import { appendResearchNode } from "./nodeInstanceTitle";
import { ensureTrainerAutoVizes } from "./trainerAutoVizSpawn";
import { serializeGraphForApi } from "./serializeGraphForApi";
import type { PlannedRandomTrainer } from "./selfDrivingGraph";

export const PHYSICS_OF_AI_OBSERVABLE_COUNT = 5;

export function findModelNodeIdFromPlan(plan: PlannedRandomTrainer): string | null {
  for (const step of plan.steps) {
    if (isObservableModelNodeType(step.node.type)) return step.node.id;
  }
  return null;
}

export async function fetchModelWeightSpecs(
  modelNodeId: string,
  nodes: Node[],
  edges: Edge[],
): Promise<Record<string, { shape: number[] }>> {
  const g = serializeGraphForApi(nodes, edges);
  const res = await fetch("/api/model_weight_specs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_node_id: modelNodeId,
      nodes: g.nodes,
      edges: g.edges,
    }),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  const j = (await res.json()) as { specs?: Record<string, { shape?: number[] }> };
  const raw = j.specs ?? {};
  const out: Record<string, { shape: number[] }> = {};
  for (const [name, spec] of Object.entries(raw)) {
    out[name] = {
      shape: Array.isArray(spec?.shape) ? spec.shape.map((x) => Number(x)) : [],
    };
  }
  return out;
}

async function postAlgebraObservable(
  modelNodeId: string,
  payload: {
    label: string;
    tensor_name: string;
    tensor_shape: number[];
    tensor_scope: ObservableTensorScope;
    flatten_mode: ObservableFlattenMode;
    observable_source: ObservableSource;
    representation_id?: string;
    layer_index?: number;
    layer_io?: string;
    reductions: AxisReductionDraft[];
  },
): Promise<AlgebraObservableItem> {
  const res = await fetch("/api/user-observables/algebra", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: payload.label,
      source_model_node_id: modelNodeId,
      tensor_name: payload.tensor_name,
      tensor_shape: payload.tensor_shape,
      tensor_scope: payload.tensor_scope,
      flatten_mode: payload.flatten_mode,
      observable_source: payload.observable_source,
      representation_id: payload.representation_id ?? "",
      layer_index: payload.layer_index ?? 0,
      layer_io: payload.layer_io ?? "",
      reductions: payload.reductions.map((r) => ({
        axis_index: r.axisIndex,
        axis_label: r.axisLabel,
        op: r.op,
      })),
    }),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  const j = (await res.json()) as { item?: AlgebraObservableItem };
  if (!j.item?.id) throw new Error("Observable create returned no item.");
  return j.item;
}

export async function fetchModelRepresentationSpecs(
  modelNodeId: string,
  nodes: Node[],
  edges: Edge[],
): Promise<RepresentationEntry[]> {
  const g = serializeGraphForApi(nodes, edges);
  const res = await fetch("/api/model_representation_specs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_node_id: modelNodeId,
      nodes: g.nodes,
      edges: g.edges,
    }),
  });
  if (!res.ok) return [];
  const j = (await res.json()) as { entries?: RepresentationEntry[] };
  return j.entries ?? [];
}

export async function createRandomUserObservablesForModel(
  modelNodeId: string,
  nodes: Node[],
  edges: Edge[],
  count: number,
  seed: number,
): Promise<AlgebraObservableItem[]> {
  const specs = await fetchModelWeightSpecs(modelNodeId, nodes, edges);
  const repEntries = await fetchModelRepresentationSpecs(modelNodeId, nodes, edges);
  const tensorNames = Object.keys(specs).sort();
  if (tensorNames.length === 0 && repEntries.length === 0) {
    throw new Error("No weight tensors or representations found for model.");
  }
  const drafts = buildRandomObservableDrafts(count, seed, tensorNames, specs, repEntries, {
    preferences: {
      ...DEFAULT_RANDOM_GENERATION_PREFERENCES,
      svEntropy: "none",
    },
  });
  if (drafts.length === 0) throw new Error("Could not sample random observable drafts.");
  const items: AlgebraObservableItem[] = [];
  for (const draft of drafts) {
    items.push(
      await postAlgebraObservable(modelNodeId, {
        label: draft.label,
        tensor_name: draft.tensorName,
        tensor_shape: draft.tensorShape,
        tensor_scope: draft.tensorScope,
        flatten_mode: draft.flattenMode,
        observable_source: draft.observableSource,
        representation_id: draft.representationId,
        layer_index: draft.layerIndex,
        layer_io: draft.layerIo,
        reductions: draft.reductions,
      }),
    );
  }
  return items;
}

function buildUserObsEdge(sourceId: string, trainerId: string): Edge {
  return {
    id: `e-poai-${sourceId}-${trainerId}`,
    source: sourceId,
    target: trainerId,
    sourceHandle: "observables",
    targetHandle: "observables",
    type: "research_default",
  };
}

/** Place ``observable_user`` nodes for saved algebra items and wire them to the trainer (+ auto viz). */
export function appendUserObservableNodesToTrainer(
  nodes: Node[],
  edges: Edge[],
  trainerId: string,
  modelNodeId: string,
  items: AlgebraObservableItem[],
): { nodes: Node[]; edges: Edge[] } {
  const trainer = nodes.find((n) => n.id === trainerId);
  const model = nodes.find((n) => n.id === modelNodeId);
  if (!trainer || !model || items.length === 0) return { nodes, edges };

  let outNodes = nodes;
  let outEdges = edges;
  const baseX = model.position.x;
  const baseY = model.position.y + 780;

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    const obsId = `observable_user-poai-${item.id.slice(0, 8)}`;
    if (outNodes.some((n) => n.id === obsId)) continue;
    const pos = {
      x: baseX + (i % 3) * 130,
      y: baseY + Math.floor(i / 3) * 72,
    };
    const obsNode = appendResearchNode(
      outNodes,
      "observable_user",
      pos,
      defaultObservableUserData({
        userObservableId: item.id,
        label: item.label,
      }) as Record<string, unknown>,
      obsId,
    );
    outNodes = [...outNodes, obsNode];
    outEdges = [...outEdges, buildUserObsEdge(obsId, trainerId)];
    const fin = ensureTrainerAutoVizes(outNodes, outEdges, trainerId);
    outNodes = fin.nodes;
    outEdges = fin.edges;
  }

  return { nodes: outNodes, edges: outEdges };
}
