import { type Node, type NodeProps, useReactFlow } from "@xyflow/react";
import { useEffect, useState } from "react";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultReshapeNodeData, type ReshapeNodeData } from "./reshapeDefaults";

function patchData(
  id: string,
  prev: ReshapeNodeData,
  patch: Partial<ReshapeNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function ReshapeNode({ id, data, selected }: NodeProps) {
  const defs = defaultReshapeNodeData();
  const d = { ...defs, ...(data as Partial<ReshapeNodeData>) } as ReshapeNodeData;
  const reshapeRule = Array.isArray(d.reshapeRule) ? d.reshapeRule[0] : d.reshapeRule;
  const hint = Array.isArray(d.shapeHint) ? d.shapeHint[0] : d.shapeHint;
  const [ruleDraft, setRuleDraft] = useState(reshapeRule);
  const { setNodes } = useReactFlow();
  useEffect(() => {
    setRuleDraft(reshapeRule);
  }, [reshapeRule]);

  return (
    <div
      className={`cr-node cr-node--reshape${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Reshape")}</div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <AtomicLayerIoStrip />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">reshape rule</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="text"
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={ruleDraft}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                aria-label="Reshape rule"
                onChange={(e) => setRuleDraft(e.target.value)}
                onBlur={() => patchData(id, d, { reshapeRule: ruleDraft.trim() || reshapeRule }, setNodes)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.currentTarget.blur();
                }}
              />
            </div>
          </div>
        </div>
        <p className="cr-node__hint">Transform tensor view for multi-head attention.</p>
        <p className="cr-node__hint cr-node__hint--extras">shape: {hint}</p>
      </div>
    </div>
  );
}
