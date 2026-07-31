import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ComfyIntListField } from "./comfyMultiFields";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultSoftmaxNodeData, type SoftmaxNodeData } from "./softmaxDefaults";
import { intChoices, packIntList } from "./multiValueUtils";

function patchData(
  id: string,
  prev: SoftmaxNodeData,
  patch: Partial<SoftmaxNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function SoftmaxNode({ id, data, selected }: NodeProps) {
  const defs = defaultSoftmaxNodeData();
  const d = { ...defs, ...(data as Partial<SoftmaxNodeData>) } as SoftmaxNodeData;
  const { setNodes } = useReactFlow();

  return (
    <div
      className={`cr-node cr-node--softmax${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Softmax")}</div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <AtomicLayerIoStrip />
        <ComfyIntListField
          label="dimension"
          values={intChoices(d.dimension, -1)}
          ariaLabel="Softmax dimension"
          onCommit={(vals) => patchData(id, d, { dimension: packIntList(vals) }, setNodes)}
        />
        <p className="cr-node__hint">Apply softmax along the selected dimension.</p>
      </div>
    </div>
  );
}
