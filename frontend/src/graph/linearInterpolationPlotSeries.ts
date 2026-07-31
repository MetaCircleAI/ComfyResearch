import type { PlotSeries } from "./sweepVizPlot";

export const LMC_TRAIN_COLOR = "var(--cr-accent-loss)";
export const LMC_TEST_COLOR = "var(--cr-accent-model)";

function finiteValues(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === "number" && Number.isFinite(item))
    : [];
}

function series(alpha: number[], values: number[], id: string, label: string, color: string): PlotSeries | null {
  const count = Math.min(alpha.length, values.length);
  if (count < 2) return null;
  return {
    id,
    label,
    color,
    points: Array.from({ length: count }, (_, index) => ({
      x: alpha[index]!,
      xDisplay: alpha[index]!.toFixed(2),
      y: values[index]!,
      rowId: `${id}-${index}`,
    })),
  };
}

/** Build only the explicitly enabled endpoint series; train and test never share a fallback. */
export function linearInterpolationPlotSeries(
  alphaValue: unknown,
  trainValue: unknown,
  testValue: unknown,
  showTrain: boolean,
  showTest: boolean,
  metric: "loss" | "accuracy",
): PlotSeries[] {
  const alpha = finiteValues(alphaValue);
  const items = [
    showTrain ? series(alpha, finiteValues(trainValue), `train-${metric}`, "train", LMC_TRAIN_COLOR) : null,
    showTest ? series(alpha, finiteValues(testValue), `test-${metric}`, "test", LMC_TEST_COLOR) : null,
  ];
  return items.filter((item): item is PlotSeries => item !== null);
}

export function interpolationAlphaCount(value: unknown): number {
  return finiteValues(value).length;
}
