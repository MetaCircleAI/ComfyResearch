import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";

export type SmoothingCurveNodeData = {
  sigma: number;
  logScaleX: boolean;
  logScaleY: boolean;
  outputTensor: WeightTensorPayload | null;
  lastError: string | null;
};

export function defaultSmoothingCurveData(
  partial?: Partial<SmoothingCurveNodeData>,
): SmoothingCurveNodeData {
  const sigmaRaw = Number(partial?.sigma);
  const sigma = Number.isFinite(sigmaRaw) ? sigmaRaw : 1;
  return {
    sigma: Math.max(0.05, sigma),
    logScaleX: partial?.logScaleX ?? false,
    logScaleY: partial?.logScaleY ?? false,
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
