import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  readSinkAttentionMassLayerMode,
  SINK_ATTENTION_MASS_LAYER_OPTIONS,
  type SinkAttentionMassLayerMode,
} from "../../graph/sinkAttentionMassLayerMode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

export function SinkAttentionMassObservableNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as { sinkTokenIndex?: unknown };
  const sinkTokenIndex = Number.isFinite(Number(raw.sinkTokenIndex)) ? Math.max(0, Math.floor(Number(raw.sinkTokenIndex))) : 0;
  const layerMode = readSinkAttentionMassLayerMode(data as object | undefined);
  const { setNodes } = useReactFlow();

  const setLayerMode = (next: SinkAttentionMassLayerMode) =>
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id
          ? { ...node, data: { ...(node.data as object), sinkAttentionMassLayers: next } }
          : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--observable-sink-attn${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_sink_attention_mass"
        title={readInstanceTitle(data, "Sink attention mass")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <DiscreteMultiSelect
          label="Layers"
          options={SINK_ATTENTION_MASS_LAYER_OPTIONS}
          value={layerMode}
          onCommit={(next) =>
            setLayerMode((Array.isArray(next) ? next[0] : next) as SinkAttentionMassLayerMode)
          }
          ariaLabel="Sink attention mass layer aggregation"
          singleSelect
        />
        <ComfyIntField
          label="sink token index"
          value={sinkTokenIndex}
          min={0}
          max={4096}
          title="Key position index j in A[query, j] (often 0 for BOS / first token)."
          onCommit={(n) =>
            setNodes((nds) =>
              nds.map((node) =>
                node.id === id ? { ...node, data: { ...(node.data as object), sinkTokenIndex: Math.max(0, n) } } : node,
              ),
            )
          }
          ariaLabel="Sink token index"
        />
        <p className="cr-observable-hint">
          Mean attention mass to the sink key across batch, heads, and queries (per layer).{" "}
          <strong>Global</strong> averages that mean over encoder layers; <strong>All layers</strong> logs one series
          per layer (viz shows multiple curves). Supported models: <strong>attention_only_model</strong> (single layer),{" "}
          <strong>transformer_token_model</strong>, <strong>transformer_multi_token_model</strong>,{" "}
          <strong>numeric_transformer_model</strong>; otherwise NaN.
        </p>
      </div>
    </div>
  );
}
