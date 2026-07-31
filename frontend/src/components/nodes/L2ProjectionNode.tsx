import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField } from "./comfyMultiFields";
import { SourceSocketRow } from "./SourceSocketRow";
import { floatChoices, packFloatList } from "./multiValueUtils";

export type L2ProjectionNodeData = {
  targetNorm: number | string;
};

function defaultL2ProjectionData(): L2ProjectionNodeData {
  return { targetNorm: 1 };
}

function patchData(
  id: string,
  prev: L2ProjectionNodeData,
  patch: Partial<L2ProjectionNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function L2ProjectionNode({ id, data, selected }: NodeProps) {
  const defs = defaultL2ProjectionData();
  const d = { ...defs, ...(data as Partial<L2ProjectionNodeData>) } as L2ProjectionNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<L2ProjectionNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--mse-loss cr-node--l2-projection${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">L2 Projection</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="target norm"
          values={floatChoices(d.targetNorm, 1)}
          title="After each optimizer step, scale trainable weights so global L2 norm equals this value"
          ariaLabel="L2 projection target norm"
          onCommit={(vals) => update({ targetNorm: packFloatList(vals) })}
        />
        <p className="cr-observable-hint">
          Projects weights onto an L2 shell of radius <strong>target norm</strong>. Wire with primary loss on
          the trainer loss socket (does not add to loss).
        </p>
      </div>
    </div>
  );
}
