import { Handle, Position, useNodeId, useReactFlow, useUpdateNodeInternals, type Node, type NodeProps } from "@xyflow/react";
import { useEffect, useLayoutEffect, useState } from "react";
import { LAYER_STRIP_SOURCE_HANDLE } from "../../graph/layerStripHandles";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultEinsumNodeData, type EinsumNodeData } from "./einsumDefaults";

function patchData(
  id: string,
  prev: EinsumNodeData,
  patch: Partial<EinsumNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function EinsumNode({ id, data, selected }: NodeProps) {
  const defs = defaultEinsumNodeData();
  const d = { ...defs, ...(data as Partial<EinsumNodeData>) } as EinsumNodeData;
  const equation = Array.isArray(d.equation) ? d.equation[0] : d.equation;
  const [equationDraft, setEquationDraft] = useState(equation);
  const { setNodes } = useReactFlow();
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals]);
  useEffect(() => {
    setEquationDraft(equation);
  }, [equation]);

  return (
    <div
      className={`cr-node cr-node--einsum${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)", minWidth: "340px" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Einsum")}</div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Einsum inputs and output">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_1"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor 1</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_2"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor 2</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id={LAYER_STRIP_SOURCE_HANDLE}
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
              />
            </div>
          </div>
        </div>
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">einsum rule</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="text"
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={equationDraft}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                aria-label="Einsum rule"
                onChange={(e) => setEquationDraft(e.target.value)}
                onBlur={() =>
                  patchData(
                    id,
                    d,
                    { equation: equationDraft.trim() || equation },
                    setNodes,
                  )
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.currentTarget.blur();
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
