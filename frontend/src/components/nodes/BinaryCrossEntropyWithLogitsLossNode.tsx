import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatListField } from "./comfyMultiFields";
import { floatChoices, packFloatList } from "./multiValueUtils";
import { SourceSocketRow } from "./SourceSocketRow";

export function BinaryCrossEntropyWithLogitsLossNode({ id, data, selected }: NodeProps) {
  const current = { lossScale: 1, ...(data as { lossScale?: number | number[] }) };
  const { setNodes } = useReactFlow();
  const update = (lossScale: number | number[]) => {
    setNodes((nodes: Node[]) =>
      nodes.map((node) =>
        node.id === id ? { ...node, data: { ...current, lossScale } } : node,
      ),
    );
  };

  return (
    <div
      className={`cr-node cr-node--mse-loss${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">
        {readInstanceTitle(data, "Binary cross-entropy (logits)")}
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="loss scale"
          values={floatChoices(current.lossScale, 1)}
          positiveOnly
          title="Multiplies binary cross-entropy in training and logs."
          ariaLabel="Binary cross-entropy loss scale"
          onCommit={(values) => update(packFloatList(values))}
        />
        <p className="cr-observable-hint">
          Use with exactly one output logit and binary class labels 0/1. The sigmoid is
          applied inside the numerically stable loss.
        </p>
      </div>
    </div>
  );
}
