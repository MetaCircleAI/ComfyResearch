import type { Edge, Node } from "@xyflow/react";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";
import { trainerIdForTrainingVisualization } from "./trainingVizPerplexitySupport";

export type TrainingVizLossView = "loss" | "reg" | "loss_plus_reg";

const WEIGHT_REG_LOSS_TYPES = new Set(["l1_reg", "l2_reg"]);

export type WeightRegKind = "l1_reg" | "l2_reg";

export function weightRegKindLabel(kind: WeightRegKind): string {
  if (kind === "l1_reg") return "L1 reg";
  if (kind === "l2_reg") return "L2 reg";
  return "reg";
}

export function trainerWeightRegKindsOnLoss(nodes: Node[], edges: Edge[], trainerId: string): WeightRegKind[] {
  const kinds: WeightRegKind[] = [];
  const seen = new Set<WeightRegKind>();
  for (const e of edges) {
    if (e.target !== trainerId || (e.targetHandle ?? null) !== "loss") continue;
    const src = nodes.find((n) => n.id === e.source);
    const t = String(src?.type ?? "");
    if (t !== "l1_reg" && t !== "l2_reg") continue;
    const kind = t as WeightRegKind;
    if (seen.has(kind)) continue;
    seen.add(kind);
    kinds.push(kind);
  }
  kinds.sort((a, b) => a.localeCompare(b));
  return kinds;
}

export function weightRegCombinedLabel(kinds: WeightRegKind[]): string {
  if (kinds.length === 0) return "reg";
  return kinds.map(weightRegKindLabel).join(" + ");
}

export function trainingVizWeightRegLabels(
  nodes: Node[],
  edges: Edge[],
  vizId: string,
): { regLabel: string; lossPlusRegLabel: string } {
  const tid = trainerIdForTrainingVisualization(edges, vizId);
  const kinds = tid ? trainerWeightRegKindsOnLoss(nodes, edges, tid) : [];
  const regLabel = weightRegCombinedLabel(kinds);
  return { regLabel, lossPlusRegLabel: `loss + ${regLabel}` };
}

export function trainerHasWeightRegOnLoss(nodes: Node[], edges: Edge[], trainerId: string): boolean {
  for (const e of edges) {
    if (e.target !== trainerId || (e.targetHandle ?? null) !== "loss") continue;
    const src = nodes.find((n) => n.id === e.source);
    if (src && WEIGHT_REG_LOSS_TYPES.has(String(src.type))) return true;
  }
  return false;
}

export function trainingVisualizationHasWeightRegLoss(nodes: Node[], edges: Edge[], vizId: string): boolean {
  const tid = trainerIdForTrainingVisualization(edges, vizId);
  if (!tid) return false;
  if (trainerHasWeightRegOnLoss(nodes, edges, tid)) return true;
  const viz = nodes.find((n) => n.id === vizId);
  const d = (viz?.data ?? {}) as TrainingVisualizationNodeData;
  const reg = d.regLossHistory ?? [];
  return reg.length >= 2;
}

export function resolveTrainingVizTrainSeries(
  d: Pick<TrainingVisualizationNodeData, "lossHistory" | "regLossHistory" | "lossView">,
): number[] {
  const primary = d.lossHistory ?? [];
  const reg = d.regLossHistory ?? [];
  const view = d.lossView ?? "loss";
  if (view === "reg") {
    return reg.length === primary.length ? reg : [];
  }
  if (view === "loss_plus_reg" && reg.length === primary.length && reg.length > 0) {
    return primary.map((v, i) => v + (reg[i] ?? 0));
  }
  return primary;
}

/** Test curve is only defined for primary task loss (not weight reg). */
export function trainingVizShowsTestCurve(
  d: Pick<TrainingVisualizationNodeData, "lossView" | "testLossHistory" | "stepTicks">,
): boolean {
  const view = d.lossView ?? "loss";
  if (view === "reg") return false;
  const test = d.testLossHistory ?? [];
  const steps = d.stepTicks ?? [];
  return test.length >= 2 && test.length === steps.length;
}
