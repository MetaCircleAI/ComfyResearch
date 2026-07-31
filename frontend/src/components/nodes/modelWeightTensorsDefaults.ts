export type WeightTensorPayload = { shape: number[]; values: number[] };

export type ModelWeightTensorsNodeData = {
  weightTensorPayloads: Record<string, WeightTensorPayload>;
  scanMessage: string | null;
  scanSummary: string | null;
};

export function defaultModelWeightTensorsData(
  partial?: Partial<ModelWeightTensorsNodeData>,
): ModelWeightTensorsNodeData {
  return {
    weightTensorPayloads: partial?.weightTensorPayloads ?? {},
    scanMessage: partial?.scanMessage ?? null,
    scanSummary: partial?.scanSummary ?? null,
  };
}
