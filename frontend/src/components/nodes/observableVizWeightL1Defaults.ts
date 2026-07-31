export type ObservableVizWeightL1NodeData = {
  /** Observable node this viz is paired with (set when auto-spawned). */
  pairedObservableId?: string;
  pairedTrainerId?: string;
  lastSweepSummary?: string;
  valueHistory?: number[];
  stepTicks?: number[];
  logScaleX?: boolean;
  logScaleY?: boolean;
  showSeries?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
};

export function defaultObservableVizWeightL1Data(
  pairedObservableId?: string,
  pairedTrainerId?: string,
): ObservableVizWeightL1NodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    logScaleX: false,
    logScaleY: false,
    showSeries: true,
  };
}
