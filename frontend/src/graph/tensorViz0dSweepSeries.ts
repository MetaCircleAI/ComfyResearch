import type { Dispatch, SetStateAction } from "react";
import type { Edge, Node } from "@xyflow/react";
import { hydrateResolved } from "./fetchActivationTensor";
import { resolveUpstreamTensor, type FlowEdge, type FlowNodeBare } from "./resolveUpstreamTensor";

/** ``tensor_viz_0d`` nodes reachable on **outgoing** edges from the trainer (subset of {@link tensorViz0dNodeIdsInTrainerGraphComponent}). */
export function tensorViz0dNodeIdsDownstreamOfTrainer(nodes: Node[], edges: Edge[], trainerNodeId: string): Set<string> {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const out = new Set<string>();
  const seen = new Set<string>();
  const q: string[] = [trainerNodeId];
  seen.add(trainerNodeId);
  while (q.length) {
    const id = q.shift()!;
    for (const e of edges) {
      if (e.source !== id) continue;
      const tid = e.target;
      if (seen.has(tid)) continue;
      seen.add(tid);
      const tgt = nodeById.get(tid);
      if (tgt?.type === "tensor_viz_0d") out.add(tid);
      q.push(tid);
    }
  }
  return out;
}

/**
 * ``tensor_viz_0d`` nodes in the same **undirected** connected component as the trainer.
 * Many canvases wire weight metrics **upstream** of the trainer (e.g. gap → 0D viz; trainer only
 * fans out checkpoint). Directed “downstream of trainer” misses those nodes, so sweep/blog never
 * persisted ``valueHistory`` while the UI still showed the correct live ``resolveUpstreamTensor`` scalar.
 */
export function tensorViz0dNodeIdsInTrainerGraphComponent(
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
): Set<string> {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  if (!nodeById.has(trainerNodeId)) return new Set();

  const adj = new Map<string, string[]>();
  for (const e of edges) {
    const a = e.source;
    const b = e.target;
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a)!.push(b);
    adj.get(b)!.push(a);
  }

  const out = new Set<string>();
  const seen = new Set<string>();
  const q: string[] = [trainerNodeId];
  seen.add(trainerNodeId);
  while (q.length) {
    const id = q.shift()!;
    const n = nodeById.get(id);
    if (n?.type === "tensor_viz_0d") out.add(id);
    for (const nb of adj.get(id) ?? []) {
      if (seen.has(nb)) continue;
      seen.add(nb);
      q.push(nb);
    }
  }
  return out;
}

/**
 * After each trainer stream payload, store **only the final step** scalar on each downstream
 * ``tensor_viz_0d`` (sweep / blog preview should not draw a curve vs. training step for 0D panels).
 */
export function patchTensorViz0dHistoriesFromTrainerStepTicks(
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
  stepTicks: number[],
): Node[] {
  const L = stepTicks.length;
  if (L === 0) return nodes;
  const tr = nodes.find((n) => n.id === trainerNodeId && (n.type === "trainer" || n.type === "crl_trainer"));
  if (!tr) return nodes;

  const bareN = nodes as FlowNodeBare[];
  const bareE = edges as FlowEdge[];
  const tv0dReach = tensorViz0dNodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId);
  const lastTick = Number(stepTicks[L - 1]);
  if (!Number.isFinite(lastTick)) return nodes;

  return nodes.map((n) => {
    if (n.type !== "tensor_viz_0d" || !tv0dReach.has(n.id)) return n;
    const resolved = resolveUpstreamTensor(bareN, bareE, n.id, "tensor");
    if (resolved.kind !== "ok" || resolved.rank !== 0 || !resolved.values.length) return n;
    const scalar = Number(resolved.values[0]!);
    if (!Number.isFinite(scalar)) return n;

    const d = { ...(n.data as Record<string, unknown>) };
    d.stepTicks = [lastTick];
    d.valueHistory = [scalar];
    return { ...n, data: d };
  });
}

/**
 * After train completes and React state reflects checkpoints + trainer payload, re-resolve each
 * ``tensor_viz_0d`` in the trainer graph component and write ``stepTicks`` / ``valueHistory``.
 * Required when the upstream tensor is ``lazy_activation`` / ``lazy_dataset`` (sync patch in
 * ``applyTrainerVizPayload`` cannot evaluate those without a network round-trip).
 */
export async function refreshTensorViz0dStoredScalarsFromLiveResolve(
  getNodes: () => Node[],
  getEdges: () => Edge[],
  setNodes: Dispatch<SetStateAction<Node[]>>,
  trainerNodeId: string,
  stepTicks: number[],
): Promise<void> {
  const nodes = getNodes().filter((n) => String(n.type) !== "graph_assist_failure_overlay");
  const edges = getEdges();
  const L = stepTicks.length;
  if (L === 0) return;
  const lastTick = Number(stepTicks[L - 1]!);
  if (!Number.isFinite(lastTick)) return;

  const bareN = nodes as FlowNodeBare[];
  const bareE = edges as FlowEdge[];
  const tvIds = tensorViz0dNodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId);
  const byId = new Map<string, { stepTicks: number[]; valueHistory: number[] }>();

  for (const id of tvIds) {
    let r = resolveUpstreamTensor(bareN, bareE, id, "tensor");
    if (r.kind === "lazy_activation" || r.kind === "lazy_dataset") {
      r = await hydrateResolved(r);
    }
    if (r.kind !== "ok" || r.rank !== 0 || !r.values.length) continue;
    const scalar = Number(r.values[0]!);
    if (!Number.isFinite(scalar)) continue;
    byId.set(id, { stepTicks: [lastTick], valueHistory: [scalar] });
  }
  if (byId.size === 0) return;

  setNodes((prev) =>
    prev.map((n) => {
      const p = byId.get(n.id);
      if (!p || n.type !== "tensor_viz_0d") return n;
      return {
        ...n,
        data: { ...((n.data as Record<string, unknown>) ?? {}), ...p },
      };
    }),
  );
}
