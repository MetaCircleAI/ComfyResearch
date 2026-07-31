export const KAN_REG_METRIC_IDS = [
  "edge_forward_spline_n",
  "edge_forward_spline_u",
  "edge_forward_sum",
  "edge_backward",
  "node_backward",
] as const;

export type KanRegMetricId = (typeof KAN_REG_METRIC_IDS)[number];

export type KanRegNodeData = {
  regMetric?: KanRegMetricId | string;
  lamb?: number | number[];
  lambL1?: number | number[];
  lambEntropy?: number | number[];
  lambCoef?: number | number[];
  lambCoefDiff?: number | number[];
};

export function defaultKanRegData(): Required<
  Pick<KanRegNodeData, "regMetric" | "lamb" | "lambL1" | "lambEntropy" | "lambCoef" | "lambCoefDiff">
> {
  return {
    regMetric: "edge_forward_spline_n",
    lamb: 0.01,
    lambL1: 1,
    lambEntropy: 2,
    lambCoef: 0,
    lambCoefDiff: 0,
  };
}
