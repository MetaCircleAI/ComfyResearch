import type { DatasetSamplingMode } from "./linearDatasetDefaults";

/** Synthetic regression pairs (x, y) with y = f_teacher(x) from wired train/test input tensors. */
export type TeacherDatasetNodeData = {
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export const defaultTeacherDatasetData = (): TeacherDatasetNodeData => ({ samplingMode: "fixed" });
