/** Viz data for neuron trajectory 2D: scatter of final positions + trajectory paths per neuron. */
export type ObservableVizNeuronTrajectory2dNodeData = {
  pairedObservableId?: string;
  pairedTrainerId?: string;
  /** Cached history: step → neuron → [x, y, ...]. */
  embeddingHistory?: number[][][];
  stepTicks?: number[];
  dimX?: number;
  dimY?: number;
  showTrails?: boolean;
  showPoints?: boolean;
};

export function defaultObservableVizNeuronTrajectory2dData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
): ObservableVizNeuronTrajectory2dNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    dimX: 0,
    dimY: 1,
    showTrails: true,
    showPoints: true,
  };
}
