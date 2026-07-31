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

export function AttentionEntropyMeanObservableNode({ id, data, selected }: NodeProps) {
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
      className={`cr-node cr-node--observable-attn-entropy${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_attention_entropy_mean"
        title={readInstanceTitle(data, "Attention entropy (mean)")}
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
          ariaLabel="Attention entropy layer aggregation"
          singleSelect
        />
        <p className="cr-observable-hint">
          Mean Shannon entropy over key dimension of softmax self-attention (high ≈ diffuse, low ≈ peaked / possible
          sink). <strong>Global</strong> averages over encoder layers; <strong>All layers</strong> logs one series per
          layer (multi-curve viz). Same models as sink mass; otherwise NaN.
        </p>
      </div>
    </div>
  );
}
