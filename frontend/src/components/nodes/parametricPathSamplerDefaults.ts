export type ParametricPathSamplerNodeData = {
  alphaMin: number;
  alphaMax: number;
  alphaSteps: number;
  metric: "loss" | "accuracy";
  split: "train" | "test";
  computeDevice: string;
  remoteGpu?: boolean;
  alphaSeries: number[];
  valueSeries: number[];
  runSummary: string | null;
  runError: string | null;
  seriesLabel: string;
};

export function defaultParametricPathSamplerData(): ParametricPathSamplerNodeData {
  return {
    alphaMin: -1,
    alphaMax: 2,
    alphaSteps: 50,
    metric: "loss",
    split: "test",
    computeDevice: "auto",
    remoteGpu: false,
    alphaSeries: [],
    valueSeries: [],
    runSummary: null,
    runError: null,
    seriesLabel: "parametric path",
  };
}
