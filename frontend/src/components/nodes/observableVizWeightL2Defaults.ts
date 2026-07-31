export type ObservableVizWeightL2NodeData = {
  /** Observable node this viz is paired with (set when auto-spawned). */
  pairedObservableId?: string;
  pairedTrainerId?: string;
  /** Swept-parameter label for the series shown (set by Trainer after each run). */
  lastSweepSummary?: string;
  valueHistory?: number[];
  /** Multi-series (``observable_viz`` + ``vizVariant`` ``weight_l2`` shares the Hessian-style chart). */
  valueHistories?: number[][];
  seriesLabels?: string[];
  stepTicks?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  showSeries?: boolean;
  eigenSeriesVisible?: boolean[];
  vizVariant?: "weight_l2";
  topK?: number;
  order?: "descending" | "ascending";
  zoomXMin?: number;
  zoomXMax?: number;
};

export function defaultObservableVizWeightL2Data(
  pairedObservableId?: string,
  pairedTrainerId?: string,
): ObservableVizWeightL2NodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    logScaleX: false,
    logScaleY: false,
    showSeries: true,
  };
}
