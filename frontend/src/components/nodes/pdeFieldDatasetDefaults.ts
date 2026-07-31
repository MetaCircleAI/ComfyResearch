import type { ListOr1 } from "./multiValueUtils";
import type { DatasetSamplingMode } from "./linearDatasetDefaults";

export type PdeFieldDatasetKind = "diffusion_pde_dataset" | "reaction_diffusion_dataset" | "advection_dataset";

export const PDE_FIELD_DATASET_KINDS: readonly PdeFieldDatasetKind[] = [
  "diffusion_pde_dataset",
  "reaction_diffusion_dataset",
  "advection_dataset",
] as const;

export type PdeFieldDatasetNodeData = {
  contextFrames: ListOr1<number>;
  channels: ListOr1<number>;
  gridSize: ListOr1<number>;
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  warmupSteps: ListOr1<number>;
  dt: ListOr1<number>;
  diffusionCoeff: ListOr1<number>;
  reactionRate: ListOr1<number>;
  velocityX: ListOr1<number>;
  velocityY: ListOr1<number>;
  icScale: ListOr1<number>;
  initSeed: ListOr1<number>;
  samplingMode?: DatasetSamplingMode;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export function defaultPdeFieldDatasetData(kind: PdeFieldDatasetKind): PdeFieldDatasetNodeData {
  const base: PdeFieldDatasetNodeData = {
    contextFrames: 4,
    channels: 1,
    gridSize: 16,
    trainSize: 512,
    testSize: 128,
    warmupSteps: 40,
    dt: 0.05,
    diffusionCoeff: 0.2,
    reactionRate: 1.0,
    velocityX: 0.5,
    velocityY: 0.2,
    icScale: 0.5,
    initSeed: 0,
    samplingMode: "fixed",
  };
  if (kind === "diffusion_pde_dataset") {
    return base;
  }
  if (kind === "reaction_diffusion_dataset") {
    return base;
  }
  return base;
}
