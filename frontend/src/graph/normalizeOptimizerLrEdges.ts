import type { Edge, Node } from "@xyflow/react";

const OPTIMIZER_NODE_TYPES = new Set([
  "adam_optimizer",
  "adamw_optimizer",
  "sgd_optimizer",
  "signsgd_optimizer",
  "muon_optimizer",
  "shampoo_optimizer",
  "soap_optimizer",
]);

/** Map UI-only ``optimizer_lr_schedule`` to persisted ``lr_schedule`` / ``mup_lr_schedule`` by source type. */
export function normalizeOptimizerLrScheduleEdgeTargets(edges: Edge[], nodes: Node[]): Edge[] {
  return edges.map((e) => {
    const th = e.targetHandle ?? "";
    if (th !== "optimizer_lr_schedule" || !e.target || !e.source) return e;
    const targetNode = nodes.find((n) => n.id === e.target);
    const sourceNode = nodes.find((n) => n.id === e.source);
    if (!targetNode || !sourceNode || !OPTIMIZER_NODE_TYPES.has(String(targetNode.type))) return e;
    const sh = e.sourceHandle ?? "";
    if (sourceNode.type === "lr_schedule" && sh === "lr_schedule") {
      return { ...e, targetHandle: "lr_schedule" };
    }
    if (sourceNode.type === "cyclic_lr_schedule" && sh === "lr_schedule") {
      return { ...e, targetHandle: "lr_schedule" };
    }
    if (sourceNode.type === "mup_lr_schedule" && sh === "mup_lr_schedule") {
      return { ...e, targetHandle: "mup_lr_schedule" };
    }
    return e;
  });
}
