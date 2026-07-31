import type { Node } from "@xyflow/react";
import { resolveUpstreamTensor, type FlowEdge } from "./resolveUpstreamTensor";
import type { ResolvedObservableVizCompare } from "./observableVizCompareResolution";

/** Resolves a rank-one tensor visualization as an index/value series for Metric compare. */
export function resolveTensorViz1dCompare(nodes: Node[], edges: FlowEdge[], node: Node): ResolvedObservableVizCompare | null {
  if (node.type !== "tensor_viz_1d") return null;
  const tensor = resolveUpstreamTensor(nodes, edges, node.id, "tensor");
  if (tensor.kind !== "ok" || tensor.rank !== 1) return null;
  const points = tensor.values
    .map((y, x) => ({ x, xDisplay: String(x), y, rowId: `${node.id}-${x}` }))
    .filter((point) => Number.isFinite(point.y));
  return {
    nodeId: node.id,
    title: "1D tensor viz",
    plotSeries: points.length >= 2 ? [{ id: node.id, label: "tensor values", color: "var(--cr-chart-2)", points }] : [],
    xKey: "index",
    yAxisLabel: "value",
    canLogX: points.length > 0 && points.every((point) => point.x > 0),
    canLogY: points.length > 0 && points.every((point) => point.y > 0),
    logScaleX: false,
    logScaleY: false,
  };
}
