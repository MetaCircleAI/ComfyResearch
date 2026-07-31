import type { ListOr1 } from "./multiValueUtils";

export type MuonOptimizerNodeData = {
  learningRate: ListOr1<number>;
  momentum: ListOr1<number>;
};

export function defaultMuonOptimizerData(): MuonOptimizerNodeData {
  return {
    learningRate: 0.003,
    momentum: 0.95,
  };
}
