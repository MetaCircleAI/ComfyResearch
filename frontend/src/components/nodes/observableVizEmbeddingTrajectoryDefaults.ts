export type ObservableVizEmbeddingTrajectoryNodeData = {
  pairedObservableId?: string;
  pairedTrainerId?: string;
  observableName?: string;
  lastSweepSummary?: string;
  embeddingHistory?: number[][][]; // step -> token -> dim
  stepTicks?: number[];
  dimX?: number;
  dimY?: number;
  showTrails?: boolean;
  showPoints?: boolean;
};

export function defaultObservableVizEmbeddingTrajectoryData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
  observableName?: string,
): ObservableVizEmbeddingTrajectoryNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    observableName: (observableName ?? "Embedding trajectory").trim() || "Embedding trajectory",
    showTrails: true,
    showPoints: true,
  };
}
