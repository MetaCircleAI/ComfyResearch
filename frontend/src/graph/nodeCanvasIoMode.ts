import type { Edge } from "@xyflow/react";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";

/** Canvas-level I/O mode: boxed model handle vs paired tensor in/out (where a node implements both). */
export type NodeCanvasIoMode = "model" | "input-output";

export function readNodeCanvasIoMode(data: Record<string, unknown> | undefined): NodeCanvasIoMode {
  const v = data?.ioMode;
  return v === "input-output" ? "input-output" : "model";
}

/** `mlp_model` / `residual_ln_model` / `attention_only_model`: right `model` only vs paired `tensor`. */
export type NodeCanvasIoPruneKind = "full_model" | "atomic_layer";

/** Remove edges that reference handles hidden after an `ioMode` switch (see node layout skill). */
export function pruneEdgesForNodeCanvasIoMode(
  edges: Edge[],
  nodeId: string,
  next: NodeCanvasIoMode,
  kind: NodeCanvasIoPruneKind,
): Edge[] {
  const stripTarget = (h: string | null | undefined) => {
    const v = (h ?? "").trim();
    return v === LAYER_STRIP_TARGET_HANDLE || v === "tensor";
  };
  const stripSource = (h: string | null | undefined) => {
    const v = (h ?? "").trim();
    return v === LAYER_STRIP_SOURCE_HANDLE || v === "tensor";
  };

  if (kind === "full_model") {
    if (next === "model") {
      return edges.filter((e) => {
        if (e.target === nodeId && stripTarget(e.targetHandle)) return false;
        if (e.source === nodeId && stripSource(e.sourceHandle)) return false;
        const th = (e.targetHandle ?? "").trim();
        const sh = (e.sourceHandle ?? "").trim();
        if (e.target === nodeId && th === COMBINED_MODEL_RETURN_TARGET_HANDLE) return false;
        if (e.source === nodeId && sh === "tensor_boundary") return false;
        return true;
      });
    }
    return edges.filter((e) => !(e.source === nodeId && e.sourceHandle === "model"));
  }
  if (next === "model") {
    return edges.filter((e) => !(e.target === nodeId && stripTarget(e.targetHandle)));
  }
  return edges;
}
