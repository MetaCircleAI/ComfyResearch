import { defaultLinearLayerData, type LinearLayerNodeData } from "./linearLayerDefaults";

export type Ai4ScienceToyAtomicKind =
  | "pairwise_rbf_layer"
  | "equivariant_message_layer"
  | "energy_readout_layer"
  | "relative_pose_encoder_layer"
  | "distance_contact_layer";


export function defaultAi4ScienceToyAtomicData(kind: Ai4ScienceToyAtomicKind): LinearLayerNodeData {
  const base = defaultLinearLayerData();
  if (kind === "pairwise_rbf_layer") {
    return { ...base, inFeatures: 18, outFeatures: 48, bias: 1, seed: 7 };
  }
  if (kind === "equivariant_message_layer") {
    return { ...base, inFeatures: 48, outFeatures: 48, bias: 1, seed: 7 };
  }
  if (kind === "energy_readout_layer") {
    return { ...base, inFeatures: 48, outFeatures: 4, bias: 1, seed: 7 };
  }
  if (kind === "relative_pose_encoder_layer") {
    return { ...base, inFeatures: 24, outFeatures: 64, bias: 1, seed: 11 };
  }
  return { ...base, inFeatures: 64, outFeatures: 6, bias: 1, seed: 11 };
}
