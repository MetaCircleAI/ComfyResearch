export type TrainingVisualizationNodeData = {
  lossHistory?: number[];
  testLossHistory?: number[];
  /** Weight regularization (L1/L2 on loss socket); train-only, no test split. */
  regLossHistory?: number[];
  stepTicks?: number[];
  /** Swept-parameter label for the curves shown (e.g. `dataset.seed=1, model.width=32`). */
  lastSweepSummary?: string;
  /** Unused in UI (plot is client SVG); kept for API compatibility. */
  plotPngBase64?: string;
  logScaleX?: boolean;
  logScaleY?: boolean;
  showTrainCurve?: boolean;
  showTestCurve?: boolean;
  zoomXMin?: number;
  zoomXMax?: number;
  /** Raw histories are CE or MSE loss; perplexity (= exp(loss)) applies only to cross-entropy in the UI. */
  yPlotMetric?: "loss" | "perplexity";
  /** When L1/L2 reg is wired to trainer loss: primary loss, reg only, or sum (test = primary only). */
  lossView?: "loss" | "reg" | "loss_plus_reg";
};

export function defaultTrainingVisualizationData(): TrainingVisualizationNodeData {
  return {
    logScaleX: false,
    logScaleY: false,
    showTrainCurve: true,
    showTestCurve: true,
    yPlotMetric: "loss",
    lossView: "loss",
  };
}
