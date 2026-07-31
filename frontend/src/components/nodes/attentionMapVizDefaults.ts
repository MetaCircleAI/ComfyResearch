export type AttentionMapVizNodeData = {
  pairedObservableId?: string;
  pairedTrainerId?: string;
  observableName?: string;
  lastSweepSummary?: string;
  attentionMapFrames?: AttentionMapFrame[];
  /** Absent means follow the newest logged frame. */
  selectedFrameStep?: number;
  selectedLayer?: number;
  selectedBatch?: number;
  selectedHead?: number;
};

export type AttentionMapSlice = {
  layer: number;
  batch: number;
  head: number;
  map: number[][];
  token_ids: number[] | null;
  source_shape: [number, number];
  row_start: number;
  col_start: number;
};

export type AttentionMapFrame = { step: number; slices: AttentionMapSlice[] };

export function defaultAttentionMapVizData(
  pairedObservableId?: string,
  pairedTrainerId?: string,
  observableName?: string,
): AttentionMapVizNodeData {
  return {
    pairedObservableId,
    pairedTrainerId,
    observableName: (observableName ?? "Attention map").trim() || "Attention map",
  };
}
