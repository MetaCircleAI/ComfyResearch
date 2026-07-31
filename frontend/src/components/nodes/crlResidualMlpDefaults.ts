import type { MlpActivationId } from "./mlpModelDefaults";

export type CrlResidualMlpNodeData = {
  stateDim: number;
  actionDim: number;
  goalDim: number;
  actorWidth: number;
  criticWidth: number;
  actorDepth: number;
  criticDepth: number;
  embedDim: number;
  /** Same ids as MLP node (relu, gelu, silu, …). */
  activation: MlpActivationId;
  seed: number;
};

export function defaultCrlResidualMlpData(): CrlResidualMlpNodeData {
  return {
    stateDim: 4,
    actionDim: 2,
    goalDim: 2,
    actorWidth: 128,
    criticWidth: 128,
    actorDepth: 4,
    criticDepth: 4,
    embedDim: 64,
    activation: "silu",
    seed: 0,
  };
}
