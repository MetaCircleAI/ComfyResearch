import type { ListOr1 } from "./multiValueUtils";

/** pykan ``plot(..., metric=...)`` attribution mode. */
export type KanPlotMetricId = "backward" | "forward_n" | "forward_u";

export type DatasetSampleSplitId = "train" | "test";

export type VisualizeKanNodeData = {
  /** Cached PNG from last successful ``/api/kan_plot`` (standard base64, no data URL prefix). */
  plotPngBase64: string;
  lastPlotError?: string | null;
  /** When the dataset socket is wired, which split’s size and sampling law to use for the plot. */
  datasetSampleSplit?: DatasetSampleSplitId;
  /** Number of random samples for a forward pass before plotting. */
  sampleCount: ListOr1<number>;
  /** Diagram scale passed to pykan ``plot(scale=...)``. */
  plotScale: ListOr1<number>;
  plotMetric: KanPlotMetricId;
};

export const KAN_PLOT_METRIC_OPTIONS: { id: KanPlotMetricId; label: string }[] = [
  { id: "backward", label: "backward (default)" },
  { id: "forward_n", label: "forward_n" },
  { id: "forward_u", label: "forward_u" },
];

export function defaultVisualizeKanData(): VisualizeKanNodeData {
  return {
    plotPngBase64: "",
    lastPlotError: null,
    datasetSampleSplit: "train",
    sampleCount: 256,
    plotScale: 0.35,
    plotMetric: "backward",
  };
}
