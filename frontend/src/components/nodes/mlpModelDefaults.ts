/** MLP: input → (depth × width, activation) → output. */

import type { NodeCanvasLevelMode } from "../../graph/nodeCanvasLevelMode";
import type { ListOr1 } from "./multiValueUtils";

export type MlpActivationId =
  | "relu"
  | "gelu"
  | "tanh"
  | "sigmoid"
  | "leaky_relu"
  | "silu"
  | "identity";

export type MlpModelNodeData = {
  /** UI: collapsed vs expanded internal layer wiring (default high). */
  levelMode?: NodeCanvasLevelMode;
  inputDim: ListOr1<number>;
  outputDim: ListOr1<number>;
  depth: ListOr1<number>;
  width: ListOr1<number>;
  activation: ListOr1<MlpActivationId>;
  /** Multiply output-layer weights by α after init: small α → rich/feature-learning, α=1 no-op (Chizat 2019). */
  outputScale?: ListOr1<number>;
  /** PyTorch RNG seed for weight initialization (dataset sampling still uses the dataset node seed). */
  seed: ListOr1<number>;
  /** Python ``def`` / spec identifier. */
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, string | number | boolean>;
};

export const MLP_ACTIVATION_OPTIONS: { id: MlpActivationId; label: string }[] = [
  { id: "relu", label: "ReLU" },
  { id: "gelu", label: "GELU" },
  { id: "tanh", label: "Tanh" },
  { id: "sigmoid", label: "Sigmoid" },
  { id: "leaky_relu", label: "Leaky ReLU" },
  { id: "silu", label: "SiLU (Swish)" },
  { id: "identity", label: "Identity (linear)" },
];

export function defaultMlpModelData(): MlpModelNodeData {
  return {
    inputDim: 10,
    outputDim: 1,
    depth: 2,
    width: 64,
    activation: "relu",
    outputScale: 1,
    seed: 0,
  };
}
