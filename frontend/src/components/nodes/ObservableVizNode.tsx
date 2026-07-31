import type { NodeProps } from "@xyflow/react";
import type { ObservableVizVariant } from "../../graph/observableVizVariant";
import { ObservableVizEmbeddingTrajectoryNode } from "./ObservableVizEmbeddingTrajectoryNode";
import { ObservableVizReluNonlinearNode } from "./ObservableVizReluNonlinearNode";
import { ObservableVizUserNode } from "./ObservableVizUserNode";
import { ObservableVizWeightL1Node } from "./ObservableVizWeightL1Node";
import { ObservableVizKanRegNode } from "./ObservableVizKanRegNode";
import { ObservableVizHessianEigenvaluesNode } from "./ObservableVizHessianEigenvaluesNode";
import { NeuronTrajectory2dVizNode } from "./NeuronTrajectory2dVizNode";
import { ObservableVizAccuracyNode } from "./ObservableVizAccuracyNode";
import { ObservableVizAttentionMapNode } from "./ObservableVizAttentionMapNode";
import { InformationPlaneVizNode } from "./InformationPlaneVizNode";
import { LayerSpectralNormVizNode } from "./LayerSpectralNormVizNode";

/**
 * Single node kind for all metric / user / embedding observable mirrors (except training viz).
 * `data.vizVariant` selects the panel implementation.
 */
export function ObservableVizNode(props: NodeProps) {
  const v = (props.data as { vizVariant?: ObservableVizVariant } | undefined)?.vizVariant ?? "user";
  switch (v) {
    case "weight_l1":
      return <ObservableVizWeightL1Node {...props} />;
    case "capacity":
      return <ObservableVizUserNode {...props} />;
    case "accuracy":
      return <ObservableVizAccuracyNode {...props} />;
    case "relu_nonlinear":
      return <ObservableVizReluNonlinearNode {...props} />;
    case "kan_reg":
      return <ObservableVizKanRegNode {...props} />;
    case "hessian_eigenvalues":
      return <ObservableVizHessianEigenvaluesNode {...props} />;
    case "gradient_norm":
    case "activation_stats":
    case "weight_l2":
    case "weight_product_sv":
      return <ObservableVizHessianEigenvaluesNode {...props} />;
    case "embedding_trajectory":
      return <ObservableVizEmbeddingTrajectoryNode {...props} />;
    case "neuron_trajectory_2d":
      return <NeuronTrajectory2dVizNode {...props} />;
    case "information_plane":
      return <InformationPlaneVizNode {...props} />;
    case "layer_spectral_norm":
      return <LayerSpectralNormVizNode {...props} />;
    case "attention_map":
      return <ObservableVizAttentionMapNode {...props} />;
    case "user":
    default:
      return <ObservableVizUserNode {...props} />;
  }
}
