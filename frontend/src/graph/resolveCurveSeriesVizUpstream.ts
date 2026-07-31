import type { Edge, Node } from "@xyflow/react";
import type { CurveSeriesTableNodeData, CurveSeriesTableRow } from "../components/nodes/curveSeriesDefaults";

export type ResolvedCurveSeriesVizUpstream = {
  rows: CurveSeriesTableRow[];
  selectedSeriesIds: string[] | null;
  paramKeyOrder: string[] | null;
};

export function resolveCurveSeriesVizUpstream(
  nodes: Node[],
  edges: Edge[],
  vizNodeId: string,
): ResolvedCurveSeriesVizUpstream | null {
  const edge = edges.find((e) => e.target === vizNodeId && (e.targetHandle ?? "") === "curves");
  if (!edge?.source) return null;
  const src = nodes.find((n) => n.id === edge.source);
  if (!src || src.type !== "curve_series_table") return null;
  const d = (src.data ?? {}) as Partial<CurveSeriesTableNodeData>;
  const rows = Array.isArray(d.rows) ? d.rows : [];
  const selectedSeriesIds =
    d.selectedSeriesIds !== undefined ? (d.selectedSeriesIds as string[] | null) : null;
  const paramKeyOrder = Array.isArray(d.paramKeyOrder) ? d.paramKeyOrder : null;
  return { rows, selectedSeriesIds, paramKeyOrder };
}
