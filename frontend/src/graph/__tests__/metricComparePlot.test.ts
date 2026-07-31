import { describe, expect, it } from "vitest";
import { buildMetricCompareOverlay } from "../metricComparePlot";
import type { ResolvedCurveSeriesViz } from "../curveSeriesVizResolution";
import { dualAxisWarranted } from "../sweepVizPlot";
import type { ResolvedObservableVizCompare } from "../observableVizCompareResolution";

function source(title: string, plotXMode: "step" | "progress", metricId: string): ResolvedCurveSeriesViz {
  return {
    nodeId: title,
    title,
    rows: [],
    effectiveRows: [{ id: title, label: metricId, x: [1, 2], y: [0.2, 0.8], params: {}, rawSweep: "", metricId }],
    plotSeries: [{ id: metricId, label: metricId, metricId, color: "#fff", points: [{ x: 1, xDisplay: "1", y: 0.2, rowId: title }, { x: 2, xDisplay: "2", y: 0.8, rowId: title }]}],
    settings: { logScaleX: false, logScaleY: false, dualAxis: true, plotXMode, plotXKey: "step" },
    xKey: plotXMode === "progress" ? "progress %" : "step",
    yAxisLabel: "value",
    canLogX: true,
    canLogY: true,
    connected: true,
  };
}

describe("metric comparison overlay", () => {
  it("keeps a common x mode and distinguishes source series", () => {
    const overlay = buildMetricCompareOverlay(source("A", "step", "train_loss"), source("B", "step", "train_loss"));
    expect(overlay.xKey).toBe("step");
    expect(overlay.series.map((series) => series.id)).toEqual(["left-train_loss", "right-train_loss"]);
    expect(overlay.series[1]!.label).toBe("Source B — B: train_loss");
    expect(overlay.series[1]!.strokeDasharray).toBe("6 3");
  });

  it("normalizes differing x modes and handles a missing source", () => {
    const overlay = buildMetricCompareOverlay(source("A", "step", "train_loss"), source("B", "progress", "train_acc"));
    expect(overlay.xKey).toBe("progress %");
    expect(overlay.canLogX).toBe(false);
    expect(overlay.series[0]!.points.map((point) => point.x)).toEqual([50, 100]);
    expect(buildMetricCompareOverlay(null, null).series).toEqual([]);
  });

  it("uses two axes only for mixed metric classes", () => {
    const lossOnly = buildMetricCompareOverlay(source("A", "step", "train_loss"), source("B", "step", "test_loss"));
    const mixed = buildMetricCompareOverlay(source("A", "step", "train_loss"), source("B", "step", "train_acc"));
    expect(dualAxisWarranted(lossOnly.series)).toBe(false);
    expect(dualAxisWarranted(mixed.series)).toBe(true);
  });

  it("merges observable 1D series with a curve visualization", () => {
    const observable: ResolvedObservableVizCompare = {
      nodeId: "attention", title: "Attention relation score", xKey: "step", yAxisLabel: "attention score",
      canLogX: true, canLogY: true, logScaleX: false, logScaleY: false,
      plotSeries: [{ id: "score", label: "score", color: "#fff", points: [{ x: 1, xDisplay: "1", y: 0.2, rowId: "score" }, { x: 2, xDisplay: "2", y: 0.4, rowId: "score" }]}],
    };
    const overlay = buildMetricCompareOverlay(source("Loss", "step", "train_loss"), observable);
    expect(overlay.series.map((series) => series.label)).toEqual(["Source A — Loss: train_loss", "Source B — Attention relation score: score"]);
  });
});
