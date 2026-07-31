import type { Edge, Node } from "@xyflow/react";
import { nodeRegistryTypesWithFamily } from "./nodeRegistrySpec";

/** Trainer connected to this training_visualization via loss_results → tensor_list. */
export function trainerIdForTrainingVisualization(edges: Edge[], vizId: string): string | null {
  const e = edges.find(
    (x) =>
      x.target === vizId &&
      (x.targetHandle ?? null) === "tensor_list" &&
      (x.sourceHandle ?? null) === "loss_results",
  );
  return e?.source ?? null;
}

const PRIMARY_TRAINER_LOSS_TYPES = new Set<string>(nodeRegistryTypesWithFamily("trainer_primary_loss"));

function lossSourceNodeType(nodes: Node[], edges: Edge[], trainerId: string): string | null {
  for (const e of edges) {
    if (e.target !== trainerId || (e.targetHandle ?? null) !== "loss") continue;
    const n = nodes.find((x) => x.id === e.source);
    if (n && PRIMARY_TRAINER_LOSS_TYPES.has(String(n.type))) {
      return (n.type as string) ?? null;
    }
  }
  return null;
}

/** Perplexity (exp(loss)) is only meaningful for cross-entropy; MSE should stay on raw loss. */
export function trainingVisualizationSupportsPerplexityYAxis(nodes: Node[], edges: Edge[], vizId: string): boolean {
  const tid = trainerIdForTrainingVisualization(edges, vizId);
  if (!tid) return false;
  return lossSourceNodeType(nodes, edges, tid) === "cross_entropy_loss";
}
