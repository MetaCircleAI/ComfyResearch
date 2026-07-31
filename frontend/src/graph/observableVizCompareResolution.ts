import type { Node } from "@xyflow/react";
import type { ObservableVizUserNodeData } from "../components/nodes/observableVizUserDefaults";
import type { PlotSeries } from "./sweepVizPlot";

export type ResolvedObservableVizCompare = {
  nodeId: string;
  title: string;
  plotSeries: PlotSeries[];
  xKey: string;
  yAxisLabel: string;
  canLogX: boolean;
  canLogY: boolean;
  logScaleX: boolean;
  logScaleY: boolean;
};

function finiteSeries(steps: unknown, values: unknown, id: string, label: string, color: string, metricId: string): PlotSeries | null {
  if (!Array.isArray(steps) || !Array.isArray(values) || steps.length !== values.length || steps.length < 2) return null;
  const points = steps.map((x, index) => ({ x: Number(x), y: Number(values[index]), rowId: id })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)).map((point) => ({ ...point, xDisplay: String(point.x) }));
  return points.length >= 2 ? { id, label, color, metricId, points } : null;
}

export function resolveObservableVizCompare(node: Node): ResolvedObservableVizCompare | null {
  if (node.type !== "observable_viz") return null;
  const data = (node.data ?? {}) as ObservableVizUserNodeData;
  const title = data.observableName?.trim() || "Observable viz";
  const steps = data.stepTicks;
  const plotSeries: PlotSeries[] = [];
  const multi = Array.isArray(data.valueHistories) ? data.valueHistories : [];
  const colors = ["var(--cr-chart-1)", "var(--cr-chart-2)", "var(--cr-chart-3)", "var(--cr-chart-4)", "var(--cr-chart-9)", "var(--cr-chart-6)", "var(--cr-chart-7)", "var(--cr-chart-5)"];
  if (multi.length > 0) {
    multi.forEach((values, index) => {
      if (data.multiSeriesVisible?.[index] === false) return;
      const label = data.seriesLabels?.[index] || `series ${index + 1}`;
      const series = finiteSeries(steps, values, `${node.id}-${index}`, label, colors[index % colors.length]!, title);
      if (series) plotSeries.push(series);
    });
  } else {
    if (data.showSeries !== false && data.showTrainCurve !== false) {
      const series = finiteSeries(steps, data.valueHistory, `${node.id}-train`, "train", "var(--cr-chart-1)", title);
      if (series) plotSeries.push(series);
    }
    if (data.showTestCurve !== false) {
      const series = finiteSeries(steps, data.testValueHistory, `${node.id}-test`, "test", "var(--cr-chart-2)", title);
      if (series) plotSeries.push({ ...series, strokeDasharray: "4 3" });
    }
  }
  const allPoints = plotSeries.flatMap((series) => series.points);
  return {
    nodeId: node.id,
    title,
    plotSeries,
    xKey: "step",
    yAxisLabel: data.vizYAxisLabel?.trim() || (data.vizVariant === "accuracy" ? "accuracy" : "value"),
    canLogX: allPoints.length > 0 && allPoints.every((point) => point.x > 0),
    canLogY: allPoints.length > 0 && allPoints.every((point) => point.y > 0),
    logScaleX: !!data.logScaleX,
    logScaleY: !!data.logScaleY,
  };
}
