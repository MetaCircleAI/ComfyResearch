import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField } from "./comfyMultiFields";
import { SourceSocketRow } from "./SourceSocketRow";
import { floatChoices, packFloatList } from "./multiValueUtils";
import { defaultDiffusionMseLossData, type DiffusionMseLossNodeData } from "./diffusionMseLossDefaults";

function patchData(
  id: string,
  prev: DiffusionMseLossNodeData,
  patch: Partial<DiffusionMseLossNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function DiffusionMseLossNode({ id, data, selected }: NodeProps) {
  const defs = defaultDiffusionMseLossData();
  const d = { ...defs, ...(data as Partial<DiffusionMseLossNodeData>) } as DiffusionMseLossNodeData;
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<DiffusionMseLossNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--mse-loss${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">Diffusion MSE loss</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="loss scale"
          values={floatChoices(d.lossScale, 1)}
          positiveOnly
          ariaLabel="Diffusion loss scale"
          onCommit={(vals) => update({ lossScale: packFloatList(vals) })}
        />
      </div>
    </div>
  );
}
