import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";

export type SeriesEndpointGapNodeData = {
  outputTensor: WeightTensorPayload | null;
  lastError: string | null;
};

export function defaultSeriesEndpointGapData(
  partial?: Partial<SeriesEndpointGapNodeData>,
): SeriesEndpointGapNodeData {
  return {
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
