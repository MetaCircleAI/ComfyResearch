/** LaTeX fully defines y(x) (including numeric constants like 10). */

import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode, InputDistributionId, OutputDistributionId } from "./linearDatasetDefaults";

export type SymbolicFuncDatasetNodeData = {
  equationLatex: string;
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  inputDistribution: ListOr1<InputDistributionId>;
  evaluationPrecision?: "input" | "float64";
  outputDistribution: ListOr1<OutputDistributionId>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  noiseLevel: ListOr1<number>;
  seed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  /** Python ``def`` / spec identifier for the view/edit-code modal. */
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

/** Default LaTeX for y(x) on new symbolic function dataset nodes. */
export const DEFAULT_SYMBOLIC_FUNC_EQUATION_LATEX = String.raw`\exp(\sin(\pi x_1) + x_2^2)`;

export function defaultSymbolicFuncDatasetData(): SymbolicFuncDatasetNodeData {
  return {
    equationLatex: DEFAULT_SYMBOLIC_FUNC_EQUATION_LATEX,
    inputDim: 2,
    outputDim: 1,
    inputDistribution: "standard_normal",
    evaluationPrecision: "input",
    outputDistribution: "deterministic",
    trainSize: 500,
    testSize: 0,
    noiseLevel: 0,
    seed: 0,
    samplingMode: "fixed",
  };
}
