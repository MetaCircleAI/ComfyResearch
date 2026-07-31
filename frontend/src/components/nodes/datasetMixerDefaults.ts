import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type DatasetMixerNodeData = {
  trainTotalSamples: ListOr1<number>;
  testTotalSamples: ListOr1<number>;
  proportionA: ListOr1<number>;
  initSeed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

/** Legacy field from first mixer version; merged into train only. */
type LegacyDatasetMixerPartial = Partial<DatasetMixerNodeData> & {
  totalSamples?: ListOr1<number>;
};

export function defaultDatasetMixerData(): DatasetMixerNodeData {
  return {
    trainTotalSamples: 800,
    testTotalSamples: 0,
    proportionA: 0.5,
    initSeed: 0,
    samplingMode: "fixed",
  };
}

/** Normalize saved graphs that still use ``totalSamples`` only. */
export function normalizeDatasetMixerData(raw: Record<string, unknown>): DatasetMixerNodeData {
  const defs = defaultDatasetMixerData();
  const r = raw as LegacyDatasetMixerPartial;
  const { totalSamples: legacyTotal, ...rest } = r;
  const merged: DatasetMixerNodeData = {
    ...defs,
    ...rest,
    trainTotalSamples: r.trainTotalSamples ?? defs.trainTotalSamples,
    testTotalSamples: r.testTotalSamples ?? defs.testTotalSamples,
    proportionA: r.proportionA ?? defs.proportionA,
    initSeed: r.initSeed ?? defs.initSeed,
    samplingMode: r.samplingMode ?? defs.samplingMode,
    specCodeName: r.specCodeName,
    paramOrder: r.paramOrder,
    extras: r.extras,
  };
  if (r.trainTotalSamples === undefined && legacyTotal !== undefined) {
    merged.trainTotalSamples = legacyTotal;
    merged.testTotalSamples = 0;
  }
  return merged;
}
