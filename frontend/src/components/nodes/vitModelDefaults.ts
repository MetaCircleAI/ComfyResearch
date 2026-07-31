import type { ListOr1 } from "./multiValueUtils";

export type VitVariant = "tiny" | "small";

export type VitModelNodeData = {
  variant: ListOr1<VitVariant>;
  patchSize: ListOr1<number>;
  hiddenDim: ListOr1<number>;
  depth: ListOr1<number>;
  numHeads: ListOr1<number>;
  seed: ListOr1<number>;
  specCodeName?: string;
};

export function defaultVitModelData(): VitModelNodeData {
  return {
    variant: "tiny",
    patchSize: 4,
    hiddenDim: 128,
    depth: 3,
    numHeads: 4,
    seed: 0,
    specCodeName: "vitModelSpec",
  };
}
