import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  OBSERVABLE_ENCODER_LAYER_OPTIONS,
  readObservableEncoderLayerMode,
  type ObservableEncoderLayerMode,
} from "../../graph/observableEncoderLayerMode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

export function EmbeddingFeatureDriftObservableNode({ id, data, selected }: NodeProps) {
  const layerMode = readObservableEncoderLayerMode(data as object | undefined);
  const { setNodes } = useReactFlow();
  const setLayerMode = (next: ObservableEncoderLayerMode) =>
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id
          ? { ...node, data: { ...(node.data as object), observableEncoderLayers: next } }
          : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--observable-embedding-feature-drift${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_embedding_feature_drift"
        title={readInstanceTitle(data, "Embedding feature drift")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <DiscreteMultiSelect
          label="Layers"
          options={OBSERVABLE_ENCODER_LAYER_OPTIONS}
          value={layerMode}
          onCommit={(next) =>
            setLayerMode((Array.isArray(next) ? next[0] : next) as ObservableEncoderLayerMode)
          }
          ariaLabel="Embedding feature drift layer aggregation"
          singleSelect
        />
        <p className="cr-observable-hint">
          <strong>Global</strong>: 1 − cosine between consecutive flattened <strong>embedding</strong> matrices.{" "}
          <strong>All layers</strong>: same metric on flattened 2D weights concatenated per encoder block{" "}
          <code className="cr-code-inline">.layers.i.</code>.
        </p>
      </div>
    </div>
  );
}
