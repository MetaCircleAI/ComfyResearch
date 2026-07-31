import type { ListOr1 } from "./multiValueUtils";

export type DatasetMixerBNodeData = {
  interpolationLambda: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultDatasetMixerBData(): DatasetMixerBNodeData {
  return {
    interpolationLambda: 0.5,
  };
}

export function normalizeDatasetMixerBData(raw: Record<string, unknown>): DatasetMixerBNodeData {
  const defs = defaultDatasetMixerBData();
  const r = raw as Partial<DatasetMixerBNodeData>;
  return {
    ...defs,
    ...r,
    interpolationLambda: r.interpolationLambda ?? defs.interpolationLambda,
    specCodeName: r.specCodeName,
    paramOrder: r.paramOrder,
    extras: r.extras,
  };
}
