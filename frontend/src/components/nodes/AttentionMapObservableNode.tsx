import { useReactFlow, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntListField } from "./comfyMultiFields";
import { intChoices, packIntList } from "./multiValueUtils";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

const MAX_SLICES_PER_FRAME = 20;

export function AttentionMapObservableNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as Record<string, unknown>;
  const layers = intChoices(raw.attentionLayerIndices, 0);
  const batches = intChoices(raw.attentionBatchIndices, 0);
  const heads = intChoices(raw.attentionHeadIndices, 0);
  const tupleCount = layers.length * batches.length * heads.length;
  const { setNodes } = useReactFlow();
  const patch = (key: string, values: number[]) =>
    setNodes((nodes) =>
      nodes.map((node) =>
        node.id === id ? { ...node, data: { ...(node.data as object), [key]: packIntList(values) } } : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--generic-observable${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_attention_map"
        title={readInstanceTitle(data, "Attention map")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <ComfyIntListField
          label="Layer indices"
          values={layers}
          min={0}
          onCommit={(values) => patch("attentionLayerIndices", values)}
          ariaLabel="Attention map layer indices"
        />
        <ComfyIntListField
          label="Batch indices"
          values={batches}
          min={0}
          onCommit={(values) => patch("attentionBatchIndices", values)}
          ariaLabel="Attention map batch indices"
        />
        <ComfyIntListField
          label="Head indices"
          values={heads}
          min={0}
          onCommit={(values) => patch("attentionHeadIndices", values)}
          ariaLabel="Attention map head indices"
        />
        <p className="cr-observable-hint">
          {layers.length} layers x {batches.length} batches x {heads.length} = {tupleCount} / {MAX_SLICES_PER_FRAME} slices per log tick.
          {tupleCount > MAX_SLICES_PER_FRAME ? " Reduce the selections to 20 tuples or fewer." : ""}
        </p>
        <p className="cr-observable-hint">Retains the newest 50 log frames. Maps are capped to the final 25 x 25 query/key window.</p>
      </div>
    </div>
  );
}
