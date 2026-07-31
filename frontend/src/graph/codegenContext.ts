import type { Edge, Node as RFNode } from "@xyflow/react";

/** Minimal edge shape for resolving wiring in the browser (no server call). */
export type CodegenEdge = Pick<Edge, "source" | "target" | "sourceHandle" | "targetHandle">;

export type CodegenContext = {
  nodes: RFNode[];
  edges: CodegenEdge[];
};

function sanitizePyIdentSegment(s: string): string {
  const t = s.replace(/[^a-zA-Z0-9_]+/g, "_").replace(/^_+/, "").replace(/_+$/, "");
  if (!t) return "node";
  if (/^[0-9]/.test(t)) return `n_${t}`;
  return t;
}

/**
 * Stable slug for generated ``fn_<slug>_*\`\` names: derived from ``node.type`` only, with ``_1``, ``_2``, …
 * when several nodes share the same type (order by ``id`` for reproducibility). Graph node ids are not used.
 */
export function pySlugForNode(nodeId: string, allNodes: RFNode[]): string {
  const node = allNodes.find((n) => n.id === nodeId);
  const type = String(node?.type ?? "node");
  const base = sanitizePyIdentSegment(type);
  const sameType = allNodes.filter((n) => String(n.type ?? "") === type).sort((a, b) => a.id.localeCompare(b.id));
  if (sameType.length <= 1) return base;
  const rankIdx = sameType.findIndex((n) => n.id === nodeId);
  if (rankIdx < 0) return `${base}_ref`;
  return `${base}_${rankIdx + 1}`;
}
