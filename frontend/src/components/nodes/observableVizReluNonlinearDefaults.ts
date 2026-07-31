export type ObservableVizReluNonlinearNodeData = {
  /** Observable node this viz is paired with (set when auto-spawned). */
  pairedObservableId?: string;
  pairedTrainerId?: string;
  /** Swept-parameter label for the series shown (set by Trainer after each run). */
  lastSweepSummary?: string;
  valueHistory?: number[];
  stepTicks?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  showSeries?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
};

export function defaultObservableVizReluNonlinearData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
): ObservableVizReluNonlinearNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    logScaleX: false,
    logScaleY: false,
    showSeries: true,
  };
}
