import type { Edge, Node } from "@xyflow/react";

/** Shapes accepted by FastAPI `list[Node]` / `list[Edge]` in user-observables and describe-path. */
export function serializeGraphForApi(
  nodes: Node[],
  edges: Edge[],
): { nodes: unknown[]; edges: unknown[] } {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type as string,
      position: n.position,
      data: (n.data as Record<string, unknown>) ?? {},
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    })),
  };
}
