import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  OBSERVABLE_ENCODER_LAYER_OPTIONS,
  readObservableEncoderLayerMode,
  type ObservableEncoderLayerMode,
} from "../../graph/observableEncoderLayerMode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

export function AttentionHeadSinkMaxObservableNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as { sinkTokenIndex?: unknown };
  const sinkTokenIndex = Number.isFinite(Number(raw.sinkTokenIndex)) ? Math.max(0, Math.floor(Number(raw.sinkTokenIndex))) : 0;
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
      className={`cr-node cr-node--observable-head-sink${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_attention_head_sink_max"
        title={readInstanceTitle(data, "Attention head sink max")}
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
          ariaLabel="Attention head sink layer aggregation"
          singleSelect
        />
        <ComfyIntField
          label="sink token index"
          value={sinkTokenIndex}
          min={0}
          max={4096}
          title="Key index for sink mass; max over heads of mean attention to that key."
          onCommit={(n) =>
            setNodes((nds) =>
              nds.map((node) =>
                node.id === id ? { ...node, data: { ...(node.data as object), sinkTokenIndex: Math.max(0, n) } } : node,
              ),
            )
          }
          ariaLabel="Sink token index for head sink"
        />
        <p className="cr-observable-hint">
          max<sub>h</sub> E<sub>i</sub>[A<sub>i,sink,h</sub>] — detects specialized sink heads.{" "}
          <strong>Global</strong> averages over encoder layers; <strong>All layers</strong> logs one series per layer.
          Same models as sink mass (<strong>attention_only_model</strong>, transformer token / multi-token,{" "}
          <strong>numeric_transformer_model</strong>); otherwise NaN.
        </p>
      </div>
    </div>
  );
}
