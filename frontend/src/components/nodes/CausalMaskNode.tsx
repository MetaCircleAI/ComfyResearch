import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ComfyIntListField } from "./comfyMultiFields";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultCausalMaskNodeData, type CausalMaskNodeData } from "./causalMaskDefaults";
import { intChoices, packIntList } from "./multiValueUtils";

function patchData(
  id: string,
  prev: CausalMaskNodeData,
  patch: Partial<CausalMaskNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function CausalMaskNode({ id, data, selected }: NodeProps) {
  const defs = defaultCausalMaskNodeData();
  const d = { ...defs, ...(data as Partial<CausalMaskNodeData>) } as CausalMaskNodeData;
  const { setNodes } = useReactFlow();

  return (
    <div
      className={`cr-node cr-node--causal-mask${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">
            {readInstanceTitle(data as Record<string, unknown>, "Causal mask")}
          </div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <AtomicLayerIoStrip />
        <ComfyIntListField
          label="diagonal offset"
          values={intChoices(d.diagonalOffset, 1)}
          ariaLabel="Causal mask diagonal offset"
          onCommit={(vals) => patchData(id, d, { diagonalOffset: packIntList(vals) }, setNodes)}
        />
        <p className="cr-node__hint">Mask future positions before softmax (upper triangle by default).</p>
      </div>
    </div>
  );
}
