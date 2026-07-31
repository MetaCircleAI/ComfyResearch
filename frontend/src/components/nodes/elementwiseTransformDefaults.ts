import type { CollectedActivationTensor } from "./activationDefaults";

export type ElementwiseTransformNodeData = {
  ruleLatex: string;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

export function defaultElementwiseTransformData(
  partial?: Partial<ElementwiseTransformNodeData>,
): ElementwiseTransformNodeData {
  return {
    ruleLatex: partial?.ruleLatex ?? "x^2",
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
