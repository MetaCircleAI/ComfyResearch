import type { ListOr1 } from "./multiValueUtils";

export type KeskarArchitecture = "c1" | "c2";

export type KeskarCnnModelNodeData = {
  architecture: ListOr1<KeskarArchitecture>;
  seed: ListOr1<number>;
  specCodeName?: string;
};

export function defaultKeskarCnnModelData(): KeskarCnnModelNodeData {
  return {
    architecture: "c1",
    seed: 0,
    specCodeName: "keskarCnnModelSpec",
  };
}
