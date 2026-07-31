import type { ListOr1 } from "./multiValueUtils";

export type ResnetVariant = "resnet18" | "resnet34" | "self_defined";

export type ResnetModelNodeData = {
  variant: ListOr1<ResnetVariant>;
  /** Stem / stage-1 base width (self_defined only; presets use 32). */
  baseChannels: ListOr1<number>;
  blocksStage1: ListOr1<number>;
  blocksStage2: ListOr1<number>;
  blocksStage3: ListOr1<number>;
  blocksStage4: ListOr1<number>;
  /** Odd 3–11; residual and stem convs (self_defined). */
  kernelSize: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
};

export function defaultResnetModelData(): ResnetModelNodeData {
  return {
    variant: "resnet18",
    baseChannels: 32,
    blocksStage1: 2,
    blocksStage2: 2,
    blocksStage3: 2,
    blocksStage4: 2,
    kernelSize: 3,
    seed: 0,
    specCodeName: "resnetModelSpec",
  };
}
