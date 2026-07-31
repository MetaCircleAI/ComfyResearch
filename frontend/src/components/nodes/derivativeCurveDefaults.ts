import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";

export type DerivativeOrder = "1" | "2" | "3" | "4" | "5";

export type DerivativeCurveNodeData = {
  order: DerivativeOrder;
  logScaleX: boolean;
  logScaleY: boolean;
  outputTensor: WeightTensorPayload | null;
  lastError: string | null;
};

export function defaultDerivativeCurveData(
  partial?: Partial<DerivativeCurveNodeData>,
): DerivativeCurveNodeData {
  const order = (partial?.order ?? "1") as DerivativeOrder;
  const validOrder: DerivativeOrder = ["1", "2", "3", "4", "5"].includes(order) ? order : "1";
  return {
    order: validOrder,
    logScaleX: partial?.logScaleX ?? false,
    logScaleY: partial?.logScaleY ?? false,
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
