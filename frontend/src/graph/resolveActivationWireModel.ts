import type { Edge, Node } from "@xyflow/react";
import type { MlpModelNodeData } from "../components/nodes/mlpModelDefaults";
import type { ResidualLnModelNodeData } from "../components/nodes/residualLnModelDefaults";

const SEQUENTIAL_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
]);

export type ActivationWireResolvedModel =
  | { kind: "mlp"; node: Node; data: MlpModelNodeData }
  | { kind: "residual_ln"; node: Node; data: ResidualLnModelNodeData }
  | { kind: "combined_model"; node: Node }
  | { kind: "sequential_tip"; tipNode: Node };

function modelCheckpointUpstream(checkpointId: string, nodes: Node[], edges: Edge[]): Node | null {
  const inc = edges.filter((e) => e.target === checkpointId && e.targetHandle === "model_checkpoint");
  for (const e of inc) {
    const tr = nodes.find((n) => n.id === e.source);
    if (tr?.type !== "trainer") continue;
    if (e.sourceHandle != null && e.sourceHandle !== "checkpoint") continue;
    return resolveTrainerModelOutput(tr.id, nodes, edges);
  }
  return null;
}

function resolveTrainerModelOutput(trainerId: string, nodes: Node[], edges: Edge[]): Node | null {
  const mlpEdge = edges.find((x) => x.target === trainerId && x.targetHandle === "model");
  if (!mlpEdge) return null;
  const src = nodes.find((n) => n.id === mlpEdge.source);
  if (!src) return null;
  if (
    src.type === "mlp_model" ||
    src.type === "gated_mlp_model" ||
    src.type === "moe_mlp_model" ||
    src.type === "residual_ln_model" ||
    src.type === "combined_model"
  ) {
    return src;
  }
  if (SEQUENTIAL_TYPES.has(String(src.type))) {
    return src;
  }
  if (src.type === "model_checkpoint") {
    return modelCheckpointUpstream(src.id, nodes, edges);
  }
  return null;
}

/**
 * Model root used for the read-only activation wire picker (extends activation model resolution
 * with ``combined_model`` and atomic sequential tips).
 */
export function resolveActivationWireModel(
  activationNodeId: string,
  nodes: Node[],
  edges: Edge[],
): ActivationWireResolvedModel | null {
  const modelEdges = edges.filter((e) => e.target === activationNodeId && e.targetHandle === "model");
  for (const e of modelEdges) {
    const src = nodes.find((n) => n.id === e.source);
    if (!src) continue;
    if (src.type === "mlp_model" || src.type === "gated_mlp_model" || src.type === "moe_mlp_model") {
      return { kind: "mlp", node: src, data: (src.data ?? {}) as MlpModelNodeData };
    }
    if (src.type === "residual_ln_model") {
      return { kind: "residual_ln", node: src, data: (src.data ?? {}) as ResidualLnModelNodeData };
    }
    if (src.type === "combined_model") {
      return { kind: "combined_model", node: src };
    }
    if (SEQUENTIAL_TYPES.has(String(src.type))) {
      return { kind: "sequential_tip", tipNode: src };
    }
    if (src.type === "trainer") {
      if (e.sourceHandle != null && e.sourceHandle !== "checkpoint") continue;
      const inner = resolveTrainerModelOutput(src.id, nodes, edges);
      if (!inner) continue;
      if (inner.type === "mlp_model" || inner.type === "gated_mlp_model" || inner.type === "moe_mlp_model") {
        return { kind: "mlp", node: inner, data: (inner.data ?? {}) as MlpModelNodeData };
      }
      if (inner.type === "residual_ln_model") {
        return { kind: "residual_ln", node: inner, data: (inner.data ?? {}) as ResidualLnModelNodeData };
      }
      if (inner.type === "combined_model") {
        return { kind: "combined_model", node: inner };
      }
      if (SEQUENTIAL_TYPES.has(String(inner.type))) {
        return { kind: "sequential_tip", tipNode: inner };
      }
    }
    if (src.type === "model_checkpoint") {
      const inner = modelCheckpointUpstream(src.id, nodes, edges);
      if (!inner) continue;
      if (inner.type === "mlp_model" || inner.type === "gated_mlp_model" || inner.type === "moe_mlp_model") {
        return { kind: "mlp", node: inner, data: (inner.data ?? {}) as MlpModelNodeData };
      }
      if (inner.type === "residual_ln_model") {
        return { kind: "residual_ln", node: inner, data: (inner.data ?? {}) as ResidualLnModelNodeData };
      }
      if (inner.type === "combined_model") {
        return { kind: "combined_model", node: inner };
      }
      if (SEQUENTIAL_TYPES.has(String(inner.type))) {
        return { kind: "sequential_tip", tipNode: inner };
      }
    }
  }
  return null;
}

export function supportsActivationWirePicker(resolved: ActivationWireResolvedModel | null): boolean {
  if (!resolved) return false;
  return resolved.kind === "mlp" || resolved.kind === "combined_model" || resolved.kind === "sequential_tip";
}
