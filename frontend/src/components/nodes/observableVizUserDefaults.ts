export type ObservableVizUserNodeData = {
  /** Chooses the unified observable viz renderer. */
  vizVariant?: import("../../graph/observableVizVariant").ObservableVizVariant;
  pairedObservableId?: string;
  pairedTrainerId?: string;
  /** Shown as “{observableName} viz” in the node header. */
  observableName?: string;
  /** Short y-axis label (e.g. trainer scalar observables); defaults to “Value”. */
  vizYAxisLabel?: string;
  lastSweepSummary?: string;
  valueHistory?: number[];
  /** Per-series rows (e.g. sink attention mass **All layers**); aligned with ``stepTicks``. */
  valueHistories?: number[][];
  /** Legend strings for ``valueHistories`` rows (same order / length). */
  seriesLabels?: string[];
  /** Test split (e.g. Accuracy observable), aligned with ``valueHistory`` / ``stepTicks``. */
  testValueHistory?: number[];
  stepTicks?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  /** When ``valueHistories`` has multiple rows: per-row visibility (Hessian-style toggles). */
  multiSeriesVisible?: boolean[];
  showSeries?: boolean;
  /** When ``testValueHistory`` is present, mirror training viz train/test toggles. */
  showTrainCurve?: boolean;
  showTestCurve?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
};

export function defaultObservableVizUserData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
  observableName?: string,
  vizYAxisLabel?: string,
): ObservableVizUserNodeData {
  const name = (observableName ?? "").trim() || "User observable";
  return {
    pairedObservableId,
    pairedTrainerId,
    observableName: name,
    ...(vizYAxisLabel != null && String(vizYAxisLabel).trim() ? { vizYAxisLabel: String(vizYAxisLabel).trim() } : {}),
    logScaleX: false,
    logScaleY: false,
    showSeries: true,
    showTrainCurve: true,
    showTestCurve: true,
  };
}
