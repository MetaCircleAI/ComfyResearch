import { Handle, Position, useNodeId, useReactFlow, useUpdateNodeInternals, type Node, type NodeProps } from "@xyflow/react";
import { useLayoutEffect, useMemo } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { LAYER_STRIP_TARGET_HANDLE } from "../../graph/layerStripHandles";
import { ComfyIntListField } from "./comfyMultiFields";
import { intChoices, packIntList } from "./multiValueUtils";
import { defaultTensorSplitterNodeData, type TensorSplitterNodeData } from "./tensorSplitterDefaults";

function patchData(
  id: string,
  prev: TensorSplitterNodeData,
  patch: Partial<TensorSplitterNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function TensorSplitterNode({ id, data, selected }: NodeProps) {
  const defs = defaultTensorSplitterNodeData();
  const d = { ...defs, ...(data as Partial<TensorSplitterNodeData>) } as TensorSplitterNodeData;
  const numParts = Math.max(2, intChoices(d.numParts, 3)[0]);
  const outputs = useMemo(() => Array.from({ length: numParts }, (_v, i) => `tensor_${i}`), [numParts]);
  const nodeId = useNodeId();
  const { setNodes } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals, numParts]);

  return (
    <div
      className={`cr-node cr-node--tensor-splitter${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor splitter")}</div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Tensor splitter inputs and outputs">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id={LAYER_STRIP_TARGET_HANDLE}
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor</span>
            </div>
          </div>
          {outputs.map((outId) => (
            <div key={outId} className="cr-trainer-io-row cr-trainer-io-row--source-out">
              <div className="cr-trainer-io-row__rightwrap">
                <span className="cr-trainer-output-label">tensor</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={outId}
                  className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
                />
              </div>
            </div>
          ))}
        </div>
        <ComfyIntListField
          label="split dimension"
          values={intChoices(d.splitDimension, -1)}
          ariaLabel="Splitter dimension"
          onCommit={(vals) => patchData(id, d, { splitDimension: packIntList(vals) }, setNodes)}
        />
        <ComfyIntListField
          label="parts"
          values={intChoices(d.numParts, 3)}
          min={2}
          ariaLabel="Number of split parts"
          onCommit={(vals) => patchData(id, d, { numParts: packIntList(vals) }, setNodes)}
        />
        <p className="cr-node__hint">Split one tensor into multiple tensor branches.</p>
      </div>
    </div>
  );
}
