import { normalizeCurveSeriesProgress } from "./curveSeriesPlot";
import type { ResolvedCurveSeriesViz } from "./curveSeriesVizResolution";
import type { ResolvedObservableVizCompare } from "./observableVizCompareResolution";
import { decoratePlotSeries, type PlotSeries } from "./sweepVizPlot";

export type MetricCompareOverlay = {
  series: PlotSeries[];
  xKey: string;
  canLogX: boolean;
  canLogY: boolean;
};

export type MetricCompareSource = ResolvedCurveSeriesViz | ResolvedObservableVizCompare;

function decorateSourceSeries(source: MetricCompareSource, slot: "left" | "right", normalizeX: boolean): PlotSeries[] {
  return source.plotSeries.map((series, index) => {
    const normalized = normalizeCurveSeriesProgress(series.points.map((point) => point.x));
    return {
      ...series,
      id: `${slot}-${series.id}`,
      label: `Source ${slot === "left" ? "A" : "B"} — ${source.title}: ${series.label}`,
      color: slot === "right" ? ["var(--cr-chart-8)", "var(--cr-chart-5)", "var(--cr-chart-4)", "var(--cr-chart-7)"][index % 4]! : series.color,
      strokeDasharray: slot === "right" ? "6 3" : series.strokeDasharray,
      points: normalizeX ? series.points.map((point, pointIndex) => ({ ...point, x: normalized[pointIndex]! })) : series.points,
    };
  });
}

export function buildMetricCompareOverlay(
  left: MetricCompareSource | null,
  right: MetricCompareSource | null,
): MetricCompareOverlay {
  const available = [left, right].filter((source): source is MetricCompareSource => source !== null);
  const isCurve = (source: MetricCompareSource): source is ResolvedCurveSeriesViz => "settings" in source;
  const sameXMode = available.length < 2 || available.every((source) =>
    isCurve(source) && isCurve(available[0]!)
      ? source.settings.plotXMode === available[0]!.settings.plotXMode
      : source.xKey === available[0]!.xKey,
  );
  const normalizeX = !sameXMode;
  return {
    series: decoratePlotSeries(available.flatMap((source) => decorateSourceSeries(source, source === left ? "left" : "right", normalizeX))),
    xKey: normalizeX ? "progress %" : available[0]?.xKey ?? "step",
    canLogX: !normalizeX && available.length > 0 && available.every((source) => source.canLogX),
    canLogY: available.length > 0 && available.every((source) => source.canLogY),
  };
}
