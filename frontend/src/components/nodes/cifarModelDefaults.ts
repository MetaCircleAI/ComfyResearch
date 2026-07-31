import type { ListOr1 } from "./multiValueUtils";

export type Vgg11CifarModelNodeData = {
  seed: ListOr1<number>;
  specCodeName?: string;
};

export function defaultVgg11CifarModelData(): Vgg11CifarModelNodeData {
  return { seed: 0, specCodeName: "vgg11CifarModelSpec" };
}

export type SmallInceptionCifarModelNodeData = Vgg11CifarModelNodeData;

export function defaultSmallInceptionCifarModelData(): SmallInceptionCifarModelNodeData {
  return { seed: 0, specCodeName: "smallInceptionCifarModelSpec" };
}
