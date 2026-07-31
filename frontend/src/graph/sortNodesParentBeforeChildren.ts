import type { Node } from "@xyflow/react";

/** Parents must appear before children in the React Flow ``nodes`` array. */
export function sortNodesParentBeforeChildren(nodes: Node[]): Node[] {
  const seen = new Set<string>();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const out: Node[] = [];
  function walk(id: string) {
    if (seen.has(id)) return;
    const n = byId.get(id);
    if (!n) return;
    const p = n.parentId ? String(n.parentId) : "";
    if (p && byId.has(p) && !seen.has(p)) walk(p);
    seen.add(id);
    out.push(n);
  }
  for (const n of nodes) walk(n.id);
  return out;
}
