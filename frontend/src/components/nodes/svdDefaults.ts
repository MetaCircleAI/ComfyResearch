import type { CollectedActivationTensor } from "./activationDefaults";

export type SvdNodeData = {
  representationId: string;
  /** When true, subtract column-wise mean before SVD (PCA-style centering). Default false. */
  removeMean: boolean;
  uTensor: CollectedActivationTensor | null;
  sTensor: CollectedActivationTensor | null;
  /** NumPy convention: Vh (k × n_features), rows are right singular vectors. */
  vTensor: CollectedActivationTensor | null;
  svdSummary: string | null;
};

export function defaultSvdData(): SvdNodeData {
  return {
    representationId: "",
    removeMean: false,
    uTensor: null,
    sTensor: null,
    vTensor: null,
    svdSummary: null,
  };
}
