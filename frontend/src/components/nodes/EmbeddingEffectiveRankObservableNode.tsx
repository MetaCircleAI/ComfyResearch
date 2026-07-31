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

export function EmbeddingEffectiveRankObservableNode({ id, data, selected }: NodeProps) {
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
      className={`cr-node cr-node--observable-embedding-effective-rank${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_embedding_effective_rank"
        title={readInstanceTitle(data, "Embedding effective rank")}
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
          ariaLabel="Embedding effective rank layer aggregation"
          singleSelect
        />
        <p className="cr-observable-hint">
          <strong>Global</strong>: effective rank of the token embedding matrix (supported token models).{" "}
          <strong>All layers</strong>: max effective rank among 2D weights in each encoder block{" "}
          <code className="cr-code-inline">.layers.i.</code> (multi-curve viz).
        </p>
      </div>
    </div>
  );
}
