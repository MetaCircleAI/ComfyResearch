import type { Edge, Node } from "@xyflow/react";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  COMBINED_SUBGRAPH_IO_EDGE_TYPE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import { readNodeCanvasLevelMode } from "./nodeCanvasLevelMode";

const TENSOR_IN_BRIDGE_TARGET_HANDLES = new Set(["tensor", LAYER_STRIP_TARGET_HANDLE]);
const TENSOR_OUT_BRIDGE_SOURCE_HANDLES = new Set(["tensor", LAYER_STRIP_SOURCE_HANDLE, "model"]);

const ATOMIC_CHAIN_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
  "tensor_splitter",
  "reshape",
  "flatten",
  "einsum",
  "softmax",
  "causal_mask",
]);

/** Nodes whose primary numeric tensor leaves via a single ``tensor`` / ``tensor_out`` source (subgraph exits besides atomic layers). */
const TENSOR_SUBGRAPH_TAIL_TYPES = new Set([
  "tensor_add",
  "tensor_stack",
  "tensor_concat",
  "basic_calculator",
  "statistics",
  "statistics2",
  "effective_rank",
  "series_endpoint_gap",
  "smoothing_curve",
  "derivative_curve",
  "tensor_constant",
  "tensor_linspace",
  "dimension_permutator",
  "tensor_slicing",
  "elementwise_transform",
]);

function newEdgeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `e-${Math.random().toString(36).slice(2, 12)}`;
}

function stripTargetForNode(n: Node | undefined): string {
  if (!n || !ATOMIC_CHAIN_TYPES.has(String(n.type))) {
    return LAYER_STRIP_TARGET_HANDLE;
  }
  const mode = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
  return mode === "input-output" ? LAYER_STRIP_TARGET_HANDLE : "tensor";
}

function stripSourceForNode(n: Node | undefined): string {
  if (!n) return LAYER_STRIP_SOURCE_HANDLE;
  const t = String(n.type);
  if (ATOMIC_CHAIN_TYPES.has(t)) {
    const mode = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
    return mode === "input-output" ? LAYER_STRIP_SOURCE_HANDLE : "tensor";
  }
  if (
    t === "tensor_add" ||
    t === "tensor_stack" ||
    t === "tensor_concat" ||
    t === "basic_calculator" ||
    t === "statistics" ||
    t === "statistics2" ||
    t === "effective_rank" ||
    t === "series_endpoint_gap" ||
    t === "smoothing_curve" ||
    t === "derivative_curve" ||
    t === "tensor_constant" ||
    t === "tensor_linspace"
    || t === "elementwise_transform"
  ) {
    return "tensor";
  }
  if (t === "dimension_permutator") {
    return "tensor_out";
  }
  if (t === "tensor_slicing") {
    return "tensor";
  }
  return LAYER_STRIP_SOURCE_HANDLE;
}

function hasInternalTensorIn(idSet: Set<string>, edges: Edge[], nid: string): boolean {
  return edges.some((e) => {
    if (e.target !== nid || !idSet.has(e.source)) return false;
    const th = (e.targetHandle ?? "").trim();
    return th === LAYER_STRIP_TARGET_HANDLE || th === "tensor" || th === "tensor_in";
  });
}

function hasInternalTensorOut(idSet: Set<string>, edges: Edge[], nid: string): boolean {
  return edges.some((e) => {
    if (e.source !== nid || !idSet.has(e.target)) return false;
    const sh = (e.sourceHandle ?? "").trim();
    return sh === LAYER_STRIP_SOURCE_HANDLE || sh === "tensor" || sh === "tensor_out" || sh === "model";
  });
}

/** ``shellId`` plus every node whose ``parentId`` chain reaches ``shellId`` (RF subtree). */
function combinedModelSubtreeIds(allNodes: Node[], shellId: string): Set<string> {
  const sub = new Set<string>([shellId]);
  let added = true;
  while (added) {
    added = false;
    for (const n of allNodes) {
      if (sub.has(n.id)) continue;
      const pid = n.parentId != null && n.parentId !== "" ? String(n.parentId) : "";
      if (pid && sub.has(pid)) {
        sub.add(n.id);
        added = true;
      }
    }
  }
  return sub;
}

function isTensorFlowEdge(e: Edge): boolean {
  const sh = (e.sourceHandle ?? "").trim();
  const th = (e.targetHandle ?? "").trim();
  const tensorSources = new Set<string>([
    LAYER_STRIP_SOURCE_HANDLE,
    "tensor",
    "tensor_out",
    "model",
    "tensor_boundary",
  ]);
  const tensorTargets = new Set<string>([
    LAYER_STRIP_TARGET_HANDLE,
    "tensor",
    "tensor_in",
    COMBINED_MODEL_RETURN_TARGET_HANDLE,
  ]);
  return tensorSources.has(sh) && tensorTargets.has(th);
}

/**
 * Tensor enters this shell's RF subtree from another selected node without using the shell's
 * left strip ``tensor_in`` (e.g. previous block's last linear → this block's first linear).
 * Then this shell must not receive an extra ``tensor_boundary`` from a wrapping parent.
 */
function hasTensorEntryBypassingShellIn(
  idSet: Set<string>,
  edges: Edge[],
  allNodes: Node[],
  shellId: string,
): boolean {
  const sub = combinedModelSubtreeIds(allNodes, shellId);
  return edges.some((e) => {
    if (!idSet.has(e.source) || !idSet.has(e.target)) return false;
    if (!isTensorFlowEdge(e)) return false;
    if (sub.has(e.source)) return false;
    if (!sub.has(e.target)) return false;
    if (e.target === shellId) {
      const th = (e.targetHandle ?? "").trim();
      return !(th === LAYER_STRIP_TARGET_HANDLE || th === "tensor" || th === "tensor_in");
    }
    return true;
  });
}

/**
 * Tensor leaves this shell's subtree into another selected node without using the shell's
 * right strip ``tensor_out`` / ``model`` (e.g. last linear → next block's interior).
 */
function hasTensorExitBypassingShellOut(
  idSet: Set<string>,
  edges: Edge[],
  allNodes: Node[],
  shellId: string,
): boolean {
  const sub = combinedModelSubtreeIds(allNodes, shellId);
  return edges.some((e) => {
    if (!idSet.has(e.source) || !idSet.has(e.target)) return false;
    if (!isTensorFlowEdge(e)) return false;
    if (e.source === shellId) {
      const sh = (e.sourceHandle ?? "").trim();
      if (sh === LAYER_STRIP_SOURCE_HANDLE || sh === "tensor" || sh === "tensor_out" || sh === "model") {
        return false;
      }
    }
    return Boolean(sub.has(e.source) && !sub.has(e.target) && e.source !== shellId);
  });
}

function atomicChainNode(n: Node | undefined): boolean {
  return n != null && ATOMIC_CHAIN_TYPES.has(String(n.type));
}

function structuralEntryIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  const ids = [...idSet].filter((nid) => {
    const n = allNodes.find((x) => x.id === nid);
    return atomicChainNode(n) && !hasInternalTensorIn(idSet, edges, nid);
  });
  return [...new Set(ids)].sort();
}

function structuralExitIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  const ids = [...idSet].filter((nid) => {
    const n = allNodes.find((x) => x.id === nid);
    return atomicChainNode(n) && !hasInternalTensorOut(idSet, edges, nid);
  });
  return [...new Set(ids)].sort();
}

/** Nested ``combined_model``: shell ``tensor`` in with no inbound tensor from another selected node. */
function combinedModelShellEntryIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  const ids = [...idSet].filter((nid) => {
    const n = allNodes.find((x) => x.id === nid);
    if (String(n?.type) !== "combined_model") return false;
    return (
      !hasInternalTensorIn(idSet, edges, nid) && !hasTensorEntryBypassingShellIn(idSet, edges, allNodes, nid)
    );
  });
  return [...new Set(ids)].sort();
}

/** Nested ``combined_model``: shell ``tensor`` out with no outbound tensor into another selected node. */
function combinedModelShellExitIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  const ids = [...idSet].filter((nid) => {
    const n = allNodes.find((x) => x.id === nid);
    if (String(n?.type) !== "combined_model") return false;
    return (
      !hasInternalTensorOut(idSet, edges, nid) && !hasTensorExitBypassingShellOut(idSet, edges, allNodes, nid)
    );
  });
  return [...new Set(ids)].sort();
}

function mergedStructuralEntryIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  return [...new Set([...structuralEntryIds(idSet, edges, allNodes), ...combinedModelShellEntryIds(idSet, edges, allNodes)])].sort();
}

/** Tensor sink nodes (e.g. Tensor add) that only feed the shell's ``tensor_return``, not used in ``structuralExitIds`` (atomic-only). */
function subgraphTensorTailExitIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  const ids = [...idSet].filter((nid) => {
    const n = allNodes.find((x) => x.id === nid);
    if (!n || !TENSOR_SUBGRAPH_TAIL_TYPES.has(String(n.type))) return false;
    return !hasInternalTensorOut(idSet, edges, nid);
  });
  return [...new Set(ids)].sort();
}

function mergedStructuralExitIds(idSet: Set<string>, edges: Edge[], allNodes: Node[]): string[] {
  return [
    ...new Set([
      ...structuralExitIds(idSet, edges, allNodes),
      ...subgraphTensorTailExitIds(idSet, edges, allNodes),
      ...combinedModelShellExitIds(idSet, edges, allNodes),
    ]),
  ].sort();
}

/**
 * After Combine, wire the ``combined_model`` wrapper:
 * - external → left ``tensor_in``; ``tensor_boundary`` → inner chain heads (or structural heads);
 * - inner chain tails → ``tensor_return`` on the right; ``tensor_out`` → external bridges.
 */
export function computeCombinedModelBridgeRewire(
  idSet: Set<string>,
  allEdges: Edge[],
  combinedId: string,
  allNodes: Node[],
): Edge[] {
  const bridgeIn = allEdges.filter((e) => !idSet.has(e.source) && idSet.has(e.target));
  const bridgeOut = allEdges.filter((e) => idSet.has(e.source) && !idSet.has(e.target));

  const tensorBridgeIn = bridgeIn.filter((e) =>
    TENSOR_IN_BRIDGE_TARGET_HANDLES.has((e.targetHandle ?? "").trim()),
  );
  const tensorBridgeOut = bridgeOut.filter((e) =>
    TENSOR_OUT_BRIDGE_SOURCE_HANDLES.has((e.sourceHandle ?? "").trim()),
  );

  const out: Edge[] = [];

  if (tensorBridgeIn.length) {
    const targetHandles = new Map<string, string>();
    for (const e of tensorBridgeIn) {
      const th = (e.targetHandle ?? LAYER_STRIP_TARGET_HANDLE).trim() || LAYER_STRIP_TARGET_HANDLE;
      if (!targetHandles.has(e.target)) targetHandles.set(e.target, th);
    }
    for (const e of tensorBridgeIn) {
      out.push({
        ...e,
        id: newEdgeId(),
        target: combinedId,
        targetHandle: LAYER_STRIP_TARGET_HANDLE,
      });
    }
    for (const [tid, th] of [...targetHandles.entries()].sort(([a], [b]) => a.localeCompare(b))) {
      out.push({
        id: newEdgeId(),
        type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
        source: combinedId,
        target: tid,
        sourceHandle: "tensor_boundary",
        targetHandle: th,
      });
    }
  } else {
    for (const tid of mergedStructuralEntryIds(idSet, allEdges, allNodes)) {
      const n = allNodes.find((x) => x.id === tid);
      out.push({
        id: newEdgeId(),
        type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
        source: combinedId,
        target: tid,
        sourceHandle: "tensor_boundary",
        targetHandle: stripTargetForNode(n),
      });
    }
  }

  for (const e of tensorBridgeOut) {
    out.push({
      ...e,
      id: newEdgeId(),
      source: combinedId,
      sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
    });
  }

  const exitBridgeSources = new Set(tensorBridgeOut.map((e) => e.source));
  if (exitBridgeSources.size) {
    for (const sid of [...exitBridgeSources].sort()) {
      const n = allNodes.find((x) => x.id === sid);
      out.push({
        id: newEdgeId(),
        type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
        source: sid,
        target: combinedId,
        sourceHandle: stripSourceForNode(n),
        targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE,
      });
    }
  } else {
    for (const sid of mergedStructuralExitIds(idSet, allEdges, allNodes)) {
      const n = allNodes.find((x) => x.id === sid);
      out.push({
        id: newEdgeId(),
        type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
        source: sid,
        target: combinedId,
        sourceHandle: stripSourceForNode(n),
        targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE,
      });
    }
  }

  return out;
}

/**
 * Appends missing shell ↔ inner **combined_subgraph_io** edges for subgraph shells in **input-output**:
 * ``tensor_boundary`` → structural entry heads, and inner exits → ``tensor_return``.
 * Applies to ``combined_model`` and low-level ``mlp_model`` / ``attention_only_model`` shells.
 */
export function repairMissingCombinedModelReturnEdges(allNodes: Node[], allEdges: Edge[]): Edge[] {
  const working = [...allEdges];
  const added: Edge[] = [];

  for (const shell of allNodes) {
    const t = String(shell.type);
    const isCombined = t === "combined_model";
    const isMlpLow = t === "mlp_model" && readNodeCanvasLevelMode((shell.data ?? {}) as Record<string, unknown>) === "low";
    const isAttentionLow =
      t === "attention_only_model" && readNodeCanvasLevelMode((shell.data ?? {}) as Record<string, unknown>) === "low";
    if (!isCombined && !isMlpLow && !isAttentionLow) continue;
    if (readNodeCanvasIoMode((shell.data ?? {}) as Record<string, unknown>) !== "input-output") continue;

    const idSet = new Set(allNodes.filter((n) => n.parentId === shell.id).map((n) => n.id));
    if (idSet.size === 0) continue;

    const hasEntry = working.some(
      (e) =>
        e.source === shell.id && (e.sourceHandle ?? "").trim() === "tensor_boundary" && idSet.has(e.target),
    );
    if (!hasEntry) {
      const entryBatch: Edge[] = [];
      for (const tid of mergedStructuralEntryIds(idSet, working, allNodes)) {
        const n = allNodes.find((x) => x.id === tid);
        entryBatch.push({
          id: newEdgeId(),
          type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
          source: shell.id,
          target: tid,
          sourceHandle: "tensor_boundary",
          targetHandle: stripTargetForNode(n),
        });
      }
      for (const e of entryBatch) {
        const dup = working.some(
          (x) =>
            x.source === e.source &&
            x.target === e.target &&
            (x.sourceHandle ?? "").trim() === (e.sourceHandle ?? "").trim() &&
            (x.targetHandle ?? "").trim() === (e.targetHandle ?? "").trim(),
        );
        if (!dup) {
          working.push(e);
          added.push(e);
        }
      }
    }

    const hasReturn = working.some(
      (e) => e.target === shell.id && (e.targetHandle ?? "").trim() === COMBINED_MODEL_RETURN_TARGET_HANDLE,
    );
    if (hasReturn) continue;

    const bridgeOut = working.filter((e) => idSet.has(e.source) && !idSet.has(e.target));
    const tensorBridgeOut = bridgeOut.filter((e) =>
      TENSOR_OUT_BRIDGE_SOURCE_HANDLES.has((e.sourceHandle ?? "").trim()),
    );
    const exitBridgeSources = new Set(tensorBridgeOut.map((e) => e.source));
    const batch: Edge[] = [];
    if (exitBridgeSources.size) {
      for (const sid of [...exitBridgeSources].sort()) {
        const n = allNodes.find((x) => x.id === sid);
        batch.push({
          id: newEdgeId(),
          type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
          source: sid,
          target: shell.id,
          sourceHandle: stripSourceForNode(n),
          targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE,
        });
      }
    } else {
      for (const sid of mergedStructuralExitIds(idSet, working, allNodes)) {
        const n = allNodes.find((x) => x.id === sid);
        batch.push({
          id: newEdgeId(),
          type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
          source: sid,
          target: shell.id,
          sourceHandle: stripSourceForNode(n),
          targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE,
        });
      }
    }
    for (const e of batch) {
      const dup = working.some(
        (x) =>
          x.source === e.source &&
          x.target === e.target &&
          (x.sourceHandle ?? "").trim() === (e.sourceHandle ?? "").trim() &&
          (x.targetHandle ?? "").trim() === (e.targetHandle ?? "").trim(),
      );
      if (!dup) {
        working.push(e);
        added.push(e);
      }
    }
  }
  return added;
}

/** Nodes fully inside the selection set and edges whose both endpoints are in the set. */
export function extractSubgraphByNodeIds(
  ids: Set<string>,
  allNodes: Node[],
  allEdges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const nodes = allNodes.filter((n) => ids.has(n.id));
  const edges = allEdges.filter((e) => ids.has(e.source) && ids.has(e.target));
  return { nodes, edges };
}

/**
 * Include every strict descendant (via ``parentId``) of any initially selected node.
 * Without this, Combine keeps only edges with both ends in the id set; a selected
 * ``combined_model`` whose children were not in the selection loses shell↔inner I/O wires.
 * Clipboard copy (⌘C) uses the same expansion so paste includes the full subgraph.
 */
export function expandCombineSelectionNodeIds(selectedIds: Iterable<string>, allNodes: Node[]): Set<string> {
  const out = new Set<string>(selectedIds);
  let added = true;
  while (added) {
    added = false;
    for (const n of allNodes) {
      if (out.has(n.id)) continue;
      const pid = n.parentId != null && n.parentId !== "" ? String(n.parentId) : "";
      if (pid && out.has(pid)) {
        out.add(n.id);
        added = true;
      }
    }
  }
  return out;
}

export function centroidPosition(nodes: Node[]): { x: number; y: number } {
  if (!nodes.length) return { x: 0, y: 0 };
  let sx = 0;
  let sy = 0;
  for (const n of nodes) {
    sx += n.position.x;
    sy += n.position.y;
  }
  return { x: sx / nodes.length, y: sy / nodes.length };
}

/** Flow-space position of ``n`` using only ``parentId`` links inside ``clipIds``. */
export function clipAbsolutePosition(
  n: Node,
  clipById: Map<string, Node>,
  clipIds: Set<string>,
): { x: number; y: number } {
  let x = n.position.x;
  let y = n.position.y;
  let pid: string | undefined = n.parentId ?? undefined;
  while (pid && clipIds.has(pid)) {
    const p = clipById.get(pid);
    if (!p) break;
    x += p.position.x;
    y += p.position.y;
    pid = p.parentId ?? undefined;
  }
  return { x, y };
}

export function centroidClipAbsolute(nodes: Node[], clipById: Map<string, Node>, clipIds: Set<string>): {
  x: number;
  y: number;
} {
  if (!nodes.length) return { x: 0, y: 0 };
  let sx = 0;
  let sy = 0;
  for (const n of nodes) {
    const a = clipAbsolutePosition(n, clipById, clipIds);
    sx += a.x;
    sy += a.y;
  }
  return { x: sx / nodes.length, y: sy / nodes.length };
}
