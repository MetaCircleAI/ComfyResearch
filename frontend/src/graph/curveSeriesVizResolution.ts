import type { Edge, Node } from "@xyflow/react";
import { effectiveCurveSeriesRows, sortCurveSeriesRowsForDisplay, type CurveSeriesTableRow } from "../components/nodes/curveSeriesDefaults";
import { buildCurveOverlayPlotSeries, curveSeriesCanLogX, curveSeriesCanLogY, type CurveSeriesPlotXMode } from "./curveSeriesPlot";
import { dualAxisWarranted, inferSeriesYAxis, type PlotSeries } from "./sweepVizPlot";
import { resolveCurveSeriesVizUpstream } from "./resolveCurveSeriesVizUpstream";
import { resolveObservableVizCompare, type ResolvedObservableVizCompare } from "./observableVizCompareResolution";
import { resolveTensorViz1dCompare } from "./tensorViz1dCompareResolution";

export type CurveSeriesVizSettings = {
  logScaleX: boolean;
  logScaleY: boolean;
  dualAxis: boolean;
  plotXMode: CurveSeriesPlotXMode;
  plotXKey: string;
};

export const defaultCurveSeriesVizSettings = (): CurveSeriesVizSettings => ({
  logScaleX: false,
  logScaleY: false,
  dualAxis: true,
  plotXMode: "progress",
  plotXKey: "step",
});

export type ResolvedCurveSeriesViz = {
  nodeId: string;
  title: string;
  rows: CurveSeriesTableRow[];
  effectiveRows: CurveSeriesTableRow[];
  plotSeries: PlotSeries[];
  settings: CurveSeriesVizSettings;
  xKey: string;
  yAxisLabel: string;
  yAxisLabelRight: string;
  canLogX: boolean;
  canLogY: boolean;
  connected: boolean;
};

export function resolveCurveSeriesViz(
  nodes: Node[],
  edges: Edge[],
  vizNodeId: string,
): ResolvedCurveSeriesViz | null {
  const node = nodes.find((candidate) => candidate.id === vizNodeId);
  if (!node || node.type !== "curve_series_viz") return null;
  const upstream = resolveCurveSeriesVizUpstream(nodes, edges, vizNodeId);
  const raw = (node.data ?? {}) as Partial<CurveSeriesVizSettings & { title?: unknown; label?: unknown }>;
  const defaults = defaultCurveSeriesVizSettings();
  const settings: CurveSeriesVizSettings = {
    logScaleX: raw.logScaleX ?? defaults.logScaleX,
    logScaleY: raw.logScaleY ?? defaults.logScaleY,
    dualAxis: raw.dualAxis ?? defaults.dualAxis,
    plotXMode: raw.plotXMode === "step" ? "step" : "progress",
    plotXKey: typeof raw.plotXKey === "string" ? raw.plotXKey : defaults.plotXKey,
  };
  const rows = upstream?.rows ?? [];
  const displayRows = sortCurveSeriesRowsForDisplay(rows, upstream?.paramKeyOrder ?? null);
  const effectiveRows = effectiveCurveSeriesRows(displayRows, upstream?.selectedSeriesIds ?? null);
  const plotSeries = buildCurveOverlayPlotSeries(displayRows, upstream?.selectedSeriesIds ?? null, settings.plotXMode);
  const metrics = new Set(effectiveRows.map((row) => row.metricId).filter(Boolean));
  const metric = metrics.size === 1 ? [...metrics][0]! : "";
  const dualAxis = settings.dualAxis && dualAxisWarranted(plotSeries);
  const logYRows = dualAxis
    ? effectiveRows.filter((row) => inferSeriesYAxis(row.label ?? "", row.metricId) === "left")
    : effectiveRows;
  const rightRows = effectiveRows.filter((row) => inferSeriesYAxis(row.label ?? "", row.metricId) === "right");
  const rightLabels = [...new Set(rightRows.map((row) => row.label).filter(Boolean))];
  const yAxisLabelRight = rightLabels.length === 1
    ? rightLabels[0]!
    : rightRows.length > 0 && rightRows.every((row) => (row.metricId ?? "").includes("acc"))
      ? "accuracy"
      : rightRows.length > 0 ? "value" : "accuracy";
  const title = typeof raw.title === "string" ? raw.title : typeof raw.label === "string" ? raw.label : "Curve series viz";
  return {
    nodeId: node.id,
    title,
    rows,
    effectiveRows,
    plotSeries,
    settings,
    xKey: settings.plotXMode === "progress" ? "progress %" : settings.plotXKey,
    yAxisLabel: metric.includes("acc") ? "accuracy" : metric.includes("loss") ? "loss" : "value",
    yAxisLabelRight,
    canLogX: curveSeriesCanLogX(effectiveRows),
    canLogY: curveSeriesCanLogY(logYRows),
    connected: upstream !== null,
  };
}

export function resolvedCurveVizUsesDualAxis(resolved: ResolvedCurveSeriesViz): boolean {
  return resolved.settings.dualAxis && dualAxisWarranted(resolved.plotSeries);
}

export function resolveMetricCompareSource(
  nodes: Node[],
  edges: Edge[],
  compareNodeId: string,
  targetHandle: "left" | "right",
): ResolvedCurveSeriesViz | ResolvedObservableVizCompare | null {
  const edge = edges.find((candidate) => candidate.target === compareNodeId && candidate.targetHandle === targetHandle);
  if (!edge || edge.sourceHandle !== "compare") return null;
  const source = nodes.find((node) => node.id === edge.source);
  if (!source) return null;
  if (source.type === "curve_series_viz") return resolveCurveSeriesViz(nodes, edges, edge.source);
  if (source.type === "observable_viz") return resolveObservableVizCompare(source);
  return resolveTensorViz1dCompare(nodes, edges, source);
}
