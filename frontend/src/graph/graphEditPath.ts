import type { GraphDocument, GraphEdge, GraphNode } from "../types/graph";

/**
 * Minimum edit path between two graphs when nodes are matched by **id** (and
 * edges by endpoint + handles). Each step applies exactly one atomic edit;
 * consecutive snapshots differ by that single edit.
 *
 * **Viewport** (pan/zoom) is not counted as an edit and does not appear in the
 * step list; the final snapshot still matches the target viewport for export.
 *
 * This is a practical “setup distance” for duplicated canvases or aligned
 * exports. If ids differ between graphs, the path degenerates to
 * remove-all / add-all (still valid, but not semantically minimal matching).
 */

export type GraphAtomicEdit =
  | { type: "remove_edge"; edgeId: string }
  | { type: "remove_node"; nodeId: string }
  | { type: "add_node"; node: GraphNode }
  | { type: "patch_node"; nodeId: string; node: GraphNode }
  | { type: "add_edge"; edge: GraphEdge }
  | { type: "set_viewport"; viewport: GraphDocument["viewport"] };

export type GraphEditPathResult = {
  /** Number of atomic edits (length of `edits`). */
  distance: number;
  edits: GraphAtomicEdit[];
  /** snapshots[0] === from, snapshots[k] === after k edits, last === to. */
  snapshots: GraphDocument[];
  /** One-line human summary per edit. */
  summaries: string[];
};

function cloneDoc(doc: GraphDocument): GraphDocument {
  return JSON.parse(JSON.stringify(doc)) as GraphDocument;
}

function cloneNode(n: GraphNode): GraphNode {
  return JSON.parse(JSON.stringify(n)) as GraphNode;
}

function cloneEdge(e: GraphEdge): GraphEdge {
  return JSON.parse(JSON.stringify(e)) as GraphEdge;
}

function edgeTopologyKey(e: GraphEdge): string {
  const sh = e.sourceHandle ?? "";
  const th = e.targetHandle ?? "";
  return `${e.source}\0${sh}\0${e.target}\0${th}`;
}

function stableDataKey(data: Record<string, unknown>): string {
  const keys = Object.keys(data).sort();
  return JSON.stringify(keys.map((k) => [k, data[k]]));
}

function nodeSignature(n: GraphNode): string {
  return JSON.stringify({
    t: n.type,
    p: n.position,
    d: stableDataKey((n.data ?? {}) as Record<string, unknown>),
  });
}

function viewportEqual(
  a: GraphDocument["viewport"],
  b: GraphDocument["viewport"],
): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function fmtValue(v: unknown): string {
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (typeof v === "number" && Number.isFinite(v)) {
    const s = String(v);
    return s;
  }
  if (typeof v === "boolean") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/** Human-readable diff for node.data (and type/position when they change). */
export function describeNodePatch(before: GraphNode, after: GraphNode): string {
  const parts: string[] = [];
  if (before.type !== after.type) {
    parts.push(`type: ${before.type} → ${after.type}`);
  }
  const pb = before.position;
  const pa = after.position;
  if (pb.x !== pa.x || pb.y !== pa.y) {
    parts.push(`position: (${pb.x}, ${pb.y}) → (${pa.x}, ${pa.y})`);
  }
  const db = (before.data ?? {}) as Record<string, unknown>;
  const da = (after.data ?? {}) as Record<string, unknown>;
  const keys = new Set([...Object.keys(db), ...Object.keys(da)]);
  for (const k of [...keys].sort()) {
    const vb = db[k];
    const va = da[k];
    const same =
      vb === va ||
      (typeof vb === "object" &&
        vb !== null &&
        typeof va === "object" &&
        va !== null &&
        JSON.stringify(vb) === JSON.stringify(va));
    if (same) continue;
    parts.push(`${k}: ${fmtValue(vb)} → ${fmtValue(va)}`);
  }
  const head = `Update node ${after.id} (${after.type})`;
  if (parts.length === 0) return head;
  return `${head}: ${parts.join("; ")}`;
}

function summarizeEdit(e: GraphAtomicEdit): string {
  switch (e.type) {
    case "remove_edge":
      return `Remove edge ${e.edgeId}`;
    case "remove_node":
      return `Remove node ${e.nodeId}`;
    case "add_node":
      return `Add node ${e.node.id} (${e.node.type})`;
    case "patch_node":
      return `Update node ${e.nodeId} (${e.node.type})`;
    case "add_edge":
      return `Add edge ${e.edge.id} (${e.edge.source} → ${e.edge.target})`;
    case "set_viewport":
      return "Update viewport";
  }
}

export function applyAtomicEdit(doc: GraphDocument, edit: GraphAtomicEdit): GraphDocument {
  const next = cloneDoc(doc);
  switch (edit.type) {
    case "remove_edge":
      next.edges = next.edges.filter((x) => x.id !== edit.edgeId);
      break;
    case "remove_node": {
      next.nodes = next.nodes.filter((n) => n.id !== edit.nodeId);
      next.edges = next.edges.filter(
        (e) => e.source !== edit.nodeId && e.target !== edit.nodeId,
      );
      break;
    }
    case "add_node":
      next.nodes = [...next.nodes, cloneNode(edit.node)];
      break;
    case "patch_node":
      next.nodes = next.nodes.map((n) => (n.id === edit.nodeId ? cloneNode(edit.node) : n));
      break;
    case "add_edge":
      next.edges = [...next.edges, cloneEdge(edit.edge)];
      break;
    case "set_viewport":
      next.viewport = edit.viewport;
      break;
  }
  return next;
}

/**
 * Builds an ordered list of atomic edits from `from` to `to` and materializes
 * each intermediate `GraphDocument`. Order: remove edges → remove nodes →
 * add nodes → patch nodes → add edges. Viewport is applied to the final
 * snapshot only and does not increase distance.
 */
export function buildIdPreservingEditPath(from: GraphDocument, to: GraphDocument): GraphEditPathResult {
  const fromNodes = new Map(from.nodes.map((n) => [n.id, n]));
  const toNodes = new Map(to.nodes.map((n) => [n.id, n]));

  const fromEdgeKeys = new Map(from.edges.map((e) => [edgeTopologyKey(e), e]));
  const toEdgeKeys = new Map(to.edges.map((e) => [edgeTopologyKey(e), e]));

  const edits: GraphAtomicEdit[] = [];

  for (const e of from.edges) {
    if (!toEdgeKeys.has(edgeTopologyKey(e))) {
      edits.push({ type: "remove_edge", edgeId: e.id });
    }
  }

  for (const id of fromNodes.keys()) {
    if (!toNodes.has(id)) {
      edits.push({ type: "remove_node", nodeId: id });
    }
  }

  for (const [, node] of toNodes) {
    if (!fromNodes.has(node.id)) {
      edits.push({ type: "add_node", node: cloneNode(node) });
    }
  }

  for (const [id, nTo] of toNodes) {
    const nFrom = fromNodes.get(id);
    if (nFrom && nodeSignature(nFrom) !== nodeSignature(nTo)) {
      edits.push({
        type: "patch_node",
        nodeId: id,
        node: cloneNode(nTo),
      });
    }
  }

  for (const e of to.edges) {
    if (!fromEdgeKeys.has(edgeTopologyKey(e))) {
      edits.push({ type: "add_edge", edge: cloneEdge(e) });
    }
  }

  let cur = cloneDoc(from);
  const snapshots: GraphDocument[] = [cloneDoc(cur)];
  for (const ed of edits) {
    cur = applyAtomicEdit(cur, ed);
    snapshots.push(cloneDoc(cur));
  }

  // Pan/zoom is not part of edit distance; align final snapshot to target for export.
  if (!viewportEqual(cur.viewport, to.viewport)) {
    cur = { ...cur, viewport: to.viewport ?? null };
    snapshots[snapshots.length - 1] = cloneDoc(cur);
  }

  const summaries = edits.map((ed) => {
    if (ed.type === "patch_node") {
      const prev = fromNodes.get(ed.nodeId);
      if (prev) return describeNodePatch(prev, ed.node);
    }
    return summarizeEdit(ed);
  });

  return {
    distance: edits.length,
    edits,
    snapshots,
    summaries,
  };
}
