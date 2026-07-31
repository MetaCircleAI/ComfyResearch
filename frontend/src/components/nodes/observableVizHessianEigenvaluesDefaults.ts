export type ObservableVizHessianEigenvaluesNodeData = {
  pairedObservableId?: string;
  pairedTrainerId?: string;
  lastSweepSummary?: string;
  /** When embedded in ``observable_viz``: gradient_norm | activation_stats | hessian_eigenvalues | … */
  vizVariant?: string;
  /** Legend labels when streaming multi-series (gradient norms, activation mean/std, …). */
  seriesLabels?: string[];
  /** Copied from paired observable at spawn (for legend labels). */
  topK?: number;
  order?: "descending" | "ascending";
  /** One series per eigenvalue rank; each length matches `stepTicks`. */
  valueHistories?: number[][];
  /** Single-series payloads from the train stream (mirrored into ``valueHistories`` in applyTrainerVizPayload). */
  valueHistory?: number[];
  stepTicks?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  /** Per-rank visibility for plotted curves (length matches ranks shown in controls). */
  eigenSeriesVisible?: boolean[];
  /** @deprecated Use `eigenSeriesVisible` per rank; if false, all ranks were hidden. */
  showSeries?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
  /** Manual horizontal reference line at this λ (e.g. 2/η for edge-of-stability). Unset = hidden. */
  sharpnessThreshold?: number;
  /** Per-parameter L2 viz: selected ``tensorFamilyFromParameterLabel`` bucket (gradient_norm / weight_l2). */
  l2TensorFamily?: string;
};

export function defaultObservableVizHessianEigenvaluesData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
  topK = 5,
  order: "descending" | "ascending" = "descending",
): ObservableVizHessianEigenvaluesNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    topK,
    order,
    logScaleX: false,
    logScaleY: false,
  };
}
