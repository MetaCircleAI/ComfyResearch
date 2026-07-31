import type { Edge, Node } from "@xyflow/react";

const SAMPLER_TYPE = "deterministic_diffusion_sampler";
const CHECKPOINT_TYPE = "model_checkpoint";

function activeCheckpoint(data: Record<string, unknown>): string {
  return String(
    data.checkpointSource === "file"
      ? data.checkpoint_b64 ?? ""
      : data.memoryCheckpoint_b64 ?? data.checkpoint_b64 ?? "",
  );
}

function snapshotNode(node: Node): Record<string, unknown> {
  const data = { ...((node.data as Record<string, unknown>) ?? {}) };
  // A trainer checkpoint is kept in two canvas fields for file/memory UX. The
  // sampling API needs one active copy only; duplicate base64 made requests
  // several times larger than the model itself.
  if (node.type === CHECKPOINT_TYPE) {
    data.checkpoint_b64 = activeCheckpoint(data);
    data.checkpointSource = "file";
    delete data.memoryCheckpoint_b64;
  } else {
    // Trainers retain the same snapshot for their "load from memory" control,
    // but the sampler resolves weights through the checkpoint node only.
    delete data.checkpoint_b64;
    delete data.memoryCheckpoint_b64;
  }
  return {
    id: node.id,
    type: node.type as string,
    position: node.position,
    data,
  };
}

/**
 * Emit only the graph required by one interactive diffusion operation.
 * Metrics stop at sampler outputs, so they never resend model checkpoints.
 */
export function diffusionReproRequestGraph(
  nodes: Node[],
  edges: Edge[],
  targetNodeId: string,
  kind: "sampler" | "observable",
): { nodes: Record<string, unknown>[]; edges: Record<string, unknown>[] } {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const selected = new Set<string>();

  const visit = (nodeId: string) => {
    if (selected.has(nodeId)) return;
    const node = byId.get(nodeId);
    if (!node) return;
    selected.add(nodeId);
    if (kind === "observable" && node.type === SAMPLER_TYPE) return;
    for (const edge of edges) {
      if (edge.target === nodeId) visit(edge.source);
    }
  };
  visit(targetNodeId);

  return {
    nodes: nodes.filter((node) => selected.has(node.id)).map(snapshotNode),
    edges: edges
      .filter((edge) => selected.has(edge.source) && selected.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle ?? null,
        targetHandle: edge.targetHandle ?? null,
      })),
  };
}
