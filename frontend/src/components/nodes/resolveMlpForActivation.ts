import type { Edge, Node } from "@xyflow/react";
import type { MlpModelNodeData } from "./mlpModelDefaults";
import type { ResidualLnModelNodeData } from "./residualLnModelDefaults";

export type ResolvedActivationModel =
  | { nodeId: string; modelType: "mlp_model"; data: MlpModelNodeData }
  | { nodeId: string; modelType: "gated_mlp_model"; data: MlpModelNodeData }
  | { nodeId: string; modelType: "moe_mlp_model"; data: MlpModelNodeData }
  | { nodeId: string; modelType: "residual_ln_model"; data: ResidualLnModelNodeData };

function resolveModelFromTrainer(trainerId: string, nodes: Node[], edges: Edge[]): ResolvedActivationModel | null {
  const mlpEdge = edges.find((x) => x.target === trainerId && x.targetHandle === "model");
  if (!mlpEdge) return null;
  const mlp = nodes.find((n) => n.id === mlpEdge.source);
  if (mlp?.type === "mlp_model") {
    return { nodeId: mlp.id, modelType: "mlp_model", data: (mlp.data ?? {}) as MlpModelNodeData };
  }
  if (mlp?.type === "gated_mlp_model") {
    return { nodeId: mlp.id, modelType: "gated_mlp_model", data: (mlp.data ?? {}) as MlpModelNodeData };
  }
  if (mlp?.type === "moe_mlp_model") {
    return { nodeId: mlp.id, modelType: "moe_mlp_model", data: (mlp.data ?? {}) as MlpModelNodeData };
  }
  if (mlp?.type === "residual_ln_model") {
    return {
      nodeId: mlp.id,
      modelType: "residual_ln_model",
      data: (mlp.data ?? {}) as ResidualLnModelNodeData,
    };
  }
  if (mlp?.type === "model_checkpoint") {
    return resolveModelUpstreamOfModelOutput(mlp.id, nodes, edges);
  }
  return null;
}

/** Follow `model` edges from an Activation node to a supported model. */
export function resolveModelForActivation(
  activationNodeId: string,
  nodes: Node[],
  edges: Edge[],
): ResolvedActivationModel | null {
  const modelEdges = edges.filter(
    (e) => e.target === activationNodeId && e.targetHandle === "model",
  );
  for (const e of modelEdges) {
    const src = nodes.find((n) => n.id === e.source);
    if (!src) continue;
    if (src.type === "mlp_model") {
      return { nodeId: src.id, modelType: "mlp_model", data: (src.data ?? {}) as MlpModelNodeData };
    }
    if (src.type === "gated_mlp_model") {
      return { nodeId: src.id, modelType: "gated_mlp_model", data: (src.data ?? {}) as MlpModelNodeData };
    }
    if (src.type === "moe_mlp_model") {
      return { nodeId: src.id, modelType: "moe_mlp_model", data: (src.data ?? {}) as MlpModelNodeData };
    }
    if (src.type === "residual_ln_model") {
      return {
        nodeId: src.id,
        modelType: "residual_ln_model",
        data: (src.data ?? {}) as ResidualLnModelNodeData,
      };
    }
    if (src.type === "trainer") {
      if (e.sourceHandle != null && e.sourceHandle !== "checkpoint") continue;
      const r = resolveModelFromTrainer(src.id, nodes, edges);
      if (r) return r;
    }
    if (src.type === "model_checkpoint") {
      const r = resolveModelUpstreamOfModelOutput(src.id, nodes, edges);
      if (r) return r;
    }
  }
  return null;
}

function resolveModelUpstreamOfModelOutput(
  nodeId: string,
  nodes: Node[],
  edges: Edge[],
): ResolvedActivationModel | null {
  const inc = edges.filter(
    (e) => e.target === nodeId && e.targetHandle === "model_checkpoint",
  );
  for (const e of inc) {
    const tr = nodes.find((n) => n.id === e.source);
    if (tr?.type !== "trainer") continue;
    if (e.sourceHandle != null && e.sourceHandle !== "checkpoint") continue;
    const r = resolveModelFromTrainer(tr.id, nodes, edges);
    if (r) return r;
  }
  return null;
}
