import { SweepVizLinePlot } from "./SweepVizLinePlot";
import type { ResolvedCurveSeriesViz } from "../../graph/curveSeriesVizResolution";
import { resolvedCurveVizUsesDualAxis } from "../../graph/curveSeriesVizResolution";

type CurveSeriesPlotBodyProps = {
  resolved: ResolvedCurveSeriesViz;
  chartId: string;
};

/** Read-only chart body shared by the source visualizer and comparison panels. */
export function CurveSeriesPlotBody({ resolved, chartId }: CurveSeriesPlotBodyProps) {
  if (!resolved.connected) return <p className="cr-tensor-viz__hint">Connect a curve series table to plot.</p>;
  if (resolved.rows.length === 0) return <p className="cr-tensor-viz__hint">Upstream table has no series yet.</p>;
  if (resolved.plotSeries.length === 0) {
    return <p className="cr-sweep-viz__plot-muted">No series to plot (check upstream selection).</p>;
  }
  const dualAxis = resolvedCurveVizUsesDualAxis(resolved);
  return (
    <SweepVizLinePlot
      chartId={chartId}
      series={resolved.plotSeries}
      xKey={resolved.xKey}
      xIsNumeric
      yAxisLabel={dualAxis ? "loss" : resolved.yAxisLabel}
      yAxisLabelRight={resolved.yAxisLabelRight}
      legendSummary={`${resolved.plotSeries.length} series`}
      logScaleX={resolved.settings.logScaleX && resolved.canLogX}
      logScaleY={resolved.settings.logScaleY && resolved.canLogY}
      dualAxis={dualAxis}
      showMarkers={false}
    />
  );
}
