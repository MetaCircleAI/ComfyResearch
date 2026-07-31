export type ObservableVizKanRegNodeData = {
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

export function defaultObservableVizKanRegData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
): ObservableVizKanRegNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    logScaleX: false,
    logScaleY: false,
    showSeries: true,
  };
}
