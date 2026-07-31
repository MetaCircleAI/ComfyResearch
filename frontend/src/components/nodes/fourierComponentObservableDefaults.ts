export type FourierComponentMetric = "relative_projection_mse" | "amplitude_ratio";

export type FourierComponentObservableNodeData = {
  frequency: number;
  metric: FourierComponentMetric;
  inputAxis: number;
  outputIndex: number;
};

export function defaultFourierComponentObservableData(): FourierComponentObservableNodeData {
  return { frequency: 1, metric: "relative_projection_mse", inputAxis: 0, outputIndex: 0 };
}
