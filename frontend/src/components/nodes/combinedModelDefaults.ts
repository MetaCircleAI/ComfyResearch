import type { Edge, Node } from "@xyflow/react";
import type { NodeCanvasIoMode } from "../../graph/nodeCanvasIoMode";
import type { NodeCanvasLevelMode } from "../../graph/nodeCanvasLevelMode";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "../../graph/layerStripHandles";

/** Internal: tensor I/O + subgraph edges removed while ``ioMode: model``; restored for ``input-output``. */
export const COMBINED_MODEL_STASHED_IO_EDGES_KEY = "__stashedTensorIoEdges";

export type CombinedModelNodeData = {
  /** Canvas I/O: right-only `tensor` vs paired left/right `tensor` (see node layout skill). */
  ioMode?: NodeCanvasIoMode;
  /** UI: low-level structure preview (reserved; currently same as high). */
  levelMode?: NodeCanvasLevelMode;
  /** User-visible title for this combined subgraph marker. */
  displayName: string;
  /** Graph library entry id (workflow for new saves; legacy template id) when added from the Nodes library. */
  templateId?: string;
  /** How many nodes were merged when this was created. */
  sourceNodeCount: number;
  /** Wrapper size when expanded (restore after collapse). */
  __expandedFrame?: { width: number; height: number };
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
  [COMBINED_MODEL_STASHED_IO_EDGES_KEY]?: Edge[];
};

export function defaultCombinedModelData(
  partial?: Partial<CombinedModelNodeData>,
): CombinedModelNodeData {
  return {
    displayName: "Combined model",
    sourceNodeCount: 0,
    ioMode: "model",
    ...partial,
  };
}

function edgeIsStashableForModelMode(e: Edge, nodeId: string): boolean {
  const th = (e.targetHandle ?? "").trim();
  const sh = (e.sourceHandle ?? "").trim();
  if (e.target === nodeId) {
    if (
      th === LAYER_STRIP_TARGET_HANDLE ||
      th === "tensor" ||
      th === COMBINED_MODEL_RETURN_TARGET_HANDLE
    ) {
      return true;
    }
  }
  if (e.source === nodeId) {
    if (
      sh === "model" ||
      sh === LAYER_STRIP_SOURCE_HANDLE ||
      sh === "tensor_out" ||
      sh === "tensor_boundary"
    ) {
      return true;
    }
  }
  return false;
}

/** Keep only serializable / structural fields so stashes survive clone and avoid function refs. */
function stashEdgeSnapshot(e: Edge): Edge {
  return {
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle,
    targetHandle: e.targetHandle,
    type: e.type,
    animated: e.animated,
    style: e.style,
    data: e.data,
    label: e.label,
    hidden: e.hidden,
    selected: e.selected,
    className: e.className,
    zIndex: e.zIndex,
  };
}

function remapCombinedModelEdgesForIoMode(edges: Edge[], nodeId: string): Edge[] {
  return edges.map((e) => {
    if (e.source === nodeId) {
      const sh = (e.sourceHandle ?? "").trim();
      if (sh === "tensor" || sh === "model") {
        return { ...e, sourceHandle: LAYER_STRIP_SOURCE_HANDLE };
      }
      return e;
    }
    if (e.target === nodeId) {
      const th = (e.targetHandle ?? "").trim();
      if (th === "tensor") {
        return { ...e, targetHandle: LAYER_STRIP_TARGET_HANDLE };
      }
      return e;
    }
    return e;
  });
}

/**
 * ``input-output`` → ``model``: remove paired / subgraph edges from the canvas but keep a copy on the
 * node so they can be restored. ``model`` → ``input-output``: remap the trainer wire and re-append
 * stashed edges, then clear the stash.
 */
export function applyCombinedModelIoModeChange(
  nodes: Node[],
  edges: Edge[],
  nodeId: string,
  next: NodeCanvasIoMode,
): { nodes: Node[]; edges: Edge[] } {
  const node = nodes.find((n) => n.id === nodeId && n.type === "combined_model");
  if (!node) return { nodes, edges };

  if (next === "model") {
    const stash: Edge[] = [];
    const nextEdges: Edge[] = [];
    for (const e of edges) {
      if (edgeIsStashableForModelMode(e, nodeId)) {
        stash.push(stashEdgeSnapshot(e));
      } else {
        nextEdges.push(e);
      }
    }
    const nextNodes = nodes.map((n) =>
      n.id === nodeId
        ? {
            ...n,
            data: {
              ...(n.data as Record<string, unknown>),
              ioMode: next,
              [COMBINED_MODEL_STASHED_IO_EDGES_KEY]: stash,
            },
          }
        : n,
    );
    return { nodes: nextNodes, edges: nextEdges };
  }

  const d = (node.data ?? {}) as Record<string, unknown>;
  const rawStash = d[COMBINED_MODEL_STASHED_IO_EDGES_KEY];
  const stashed = Array.isArray(rawStash) ? (rawStash as Edge[]) : [];

  let nextEdges = remapCombinedModelEdgesForIoMode(edges, nodeId);
  const existingIds = new Set(nextEdges.map((e) => String(e.id)));
  for (const se of stashed) {
    let id = String(se.id ?? "");
    if (!id || existingIds.has(id)) {
      id = `e-${Math.random().toString(36).slice(2, 12)}`;
    }
    nextEdges = [...nextEdges, { ...se, id }];
    existingIds.add(id);
  }

  const nextNodes = nodes.map((n) => {
    if (n.id !== nodeId) return n;
    const nextData = { ...(n.data as Record<string, unknown>) };
    delete nextData[COMBINED_MODEL_STASHED_IO_EDGES_KEY];
    return { ...n, data: { ...nextData, ioMode: next } };
  });

  return { nodes: nextNodes, edges: nextEdges };
}
