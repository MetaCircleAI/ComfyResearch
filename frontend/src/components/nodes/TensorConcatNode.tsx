import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useMemo } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  clampTensorInputCount,
  defaultTensorConcatData,
  type TensorMultiInputNodeData,
} from "./tensorMultiInputDefaults";

function patchData(
  id: string,
  prev: TensorMultiInputNodeData,
  patch: Partial<TensorMultiInputNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function TensorConcatNode({ id, data, selected }: NodeProps) {
  const defs = defaultTensorConcatData();
  const d = { ...defs, ...(data as Partial<TensorMultiInputNodeData>) } as TensorMultiInputNodeData;
  const inputCount = clampTensorInputCount(d.inputCount);
  const concatDimension = Number.isFinite(Number(d.concatDimension)) ? Math.floor(Number(d.concatDimension)) : 0;
  const { setNodes } = useReactFlow();
  const handles = useMemo(
    () => Array.from({ length: inputCount }, (_, i) => ({ id: `tensor_${i + 1}`, label: `tensor ${i + 1}` })),
    [inputCount],
  );

  return (
    <div
      className={`cr-node cr-node--tensor-concat${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor concat")}</span>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Tensor concat inputs and output">
          {handles.map((h, idx) => (
            <div key={h.id} className="cr-trainer-io-row">
              <div className="cr-trainer-io-row__leftwrap">
                <Handle
                  type="target"
                  position={Position.Left}
                  id={h.id}
                  className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
                />
                <span className="cr-trainer-socket-label">{h.label}</span>
              </div>
              {idx === handles.length - 1 ? (
                <div className="cr-trainer-io-row__rightwrap">
                  <span className="cr-trainer-output-label">tensor</span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id="tensor"
                    className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
                  />
                </div>
              ) : (
                <div className="cr-trainer-io-row__rightwrap" />
              )}
            </div>
          ))}
        </div>

        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">input tensors</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="number"
                min={2}
                step={1}
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={inputCount}
                aria-label="Tensor concat input tensors"
                onChange={(e) => patchData(id, d, { inputCount: clampTensorInputCount(e.target.value) }, setNodes)}
              />
            </div>
          </div>
        </div>
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">concat dimension</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="number"
                step={1}
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={concatDimension}
                aria-label="Tensor concat dimension"
                onChange={(e) =>
                  patchData(
                    id,
                    d,
                    {
                      concatDimension: Number.isFinite(Number(e.target.value))
                        ? Math.floor(Number(e.target.value))
                        : 0,
                    },
                    setNodes,
                  )
                }
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
