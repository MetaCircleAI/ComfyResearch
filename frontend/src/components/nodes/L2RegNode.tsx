import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField } from "./comfyMultiFields";
import { SourceSocketRow } from "./SourceSocketRow";
import { floatChoices, packFloatList } from "./multiValueUtils";
import { defaultWeightRegData, type WeightRegNodeData } from "./weightRegDefaults";

function patchData(
  id: string,
  prev: WeightRegNodeData,
  patch: Partial<WeightRegNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function L2RegNode({ id, data, selected }: NodeProps) {
  const defs = defaultWeightRegData();
  const d = { ...defs, ...(data as Partial<WeightRegNodeData>) } as WeightRegNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<WeightRegNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--mse-loss cr-node--l2-reg${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">L2 Reg</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="loss scale"
          values={floatChoices(d.lossScale, 1)}
          title="Multiplies Σw² over trainable weights — 0 disables; negative subtracts from loss — comma-separated for multiple runs"
          ariaLabel="L2 reg loss scale"
          onCommit={(vals) => update({ lossScale: packFloatList(vals) })}
        />
        <p className="cr-observable-hint">
          Adds <strong>loss scale × Σw²</strong> to the primary loss. Wire together with MSE or cross-entropy on
          the trainer loss socket.
        </p>
      </div>
    </div>
  );
}
