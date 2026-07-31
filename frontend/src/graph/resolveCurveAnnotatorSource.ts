import type { Edge, Node } from "@xyflow/react";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";
import { getObservableVizVariant, type ObservableVizVariant } from "./observableVizVariant";

/** Observable scalar line panels: stepTicks + valueHistory, shared controls. */
export type ObservableScalarVizData = {
  stepTicks?: number[];
  valueHistory?: number[];
  testValueHistory?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  showSeries?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
  lastSweepSummary?: string;
};

const SCALAR_ANNOTATOR_VARIANTS = new Set<ObservableVizVariant>([
  "weight_l2",
  "weight_l1",
  "capacity",
  "accuracy",
  "relu_nonlinear",
  "user",
]);

export type CurveAnnotatorResolved =
  | {
      kind: "training";
      sourceId: string;
      data: TrainingVisualizationNodeData;
    }
  | {
      kind: "observable_scalar";
      sourceId: string;
      variant: ObservableVizVariant;
      data: ObservableScalarVizData;
    }
  | {
      kind: "unsupported";
      sourceId: string;
      reason: string;
    }
  | { kind: "disconnected" };

export function resolveCurveAnnotatorSource(
  nodes: Node[],
  edges: Edge[],
  annotatorId: string,
): CurveAnnotatorResolved {
  const inc = edges.find((e) => e.target === annotatorId && (e.targetHandle ?? "") === "from_viz");
  if (!inc) return { kind: "disconnected" };

  const src = nodes.find((n) => n.id === inc.source);
  if (!src) return { kind: "disconnected" };

  if (inc.sourceHandle !== "annotator") {
    return { kind: "disconnected" };
  }

  if (src.type === "training_visualization") {
    return {
      kind: "training",
      sourceId: src.id,
      data: (src.data ?? {}) as TrainingVisualizationNodeData,
    };
  }

  if (src.type === "observable_viz") {
    const variant = getObservableVizVariant(src);
    if (!variant) {
      return { kind: "unsupported", sourceId: src.id, reason: "Unknown observable viz variant." };
    }
    if (variant === "embedding_trajectory") {
      return {
        kind: "unsupported",
        sourceId: src.id,
        reason: "Embedding trajectory is not a 1D scalar line chart.",
      };
    }
    if (!SCALAR_ANNOTATOR_VARIANTS.has(variant)) {
      return { kind: "unsupported", sourceId: src.id, reason: "This observable panel is not supported yet." };
    }
    return {
      kind: "observable_scalar",
      sourceId: src.id,
      variant,
      data: (src.data ?? {}) as ObservableScalarVizData,
    };
  }

  return { kind: "disconnected" };
}
