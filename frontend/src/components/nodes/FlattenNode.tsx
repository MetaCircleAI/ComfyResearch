import { type Node, type NodeProps, useReactFlow } from "@xyflow/react";
import { useEffect, useState } from "react";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultFlattenNodeData, readFlattenExceptDim, type FlattenNodeData } from "./flattenDefaults";

function patchData(
  id: string,
  prev: FlattenNodeData,
  patch: Partial<FlattenNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)));
}

export function FlattenNode({ id, data, selected }: NodeProps) {
  const defs = defaultFlattenNodeData();
  const d = { ...defs, ...(data as Partial<FlattenNodeData>) } as FlattenNodeData;
  const exceptDim = readFlattenExceptDim(d.exceptDim);
  const [draft, setDraft] = useState(exceptDim === null ? "" : String(exceptDim));
  const { setNodes } = useReactFlow();
  useEffect(() => {
    setDraft(exceptDim === null ? "" : String(exceptDim));
  }, [exceptDim]);

  const commitDraft = () => {
    const t = draft.trim().toLowerCase();
    if (t === "" || t === "null") {
      patchData(id, d, { exceptDim: null }, setNodes);
      return;
    }
    const n = Number.parseInt(draft.trim(), 10);
    patchData(id, d, { exceptDim: Number.isFinite(n) ? n : null }, setNodes);
  };

  return (
    <div
      className={`cr-node cr-node--flatten${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Flatten")}</div>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <AtomicLayerIoStrip />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">flatten except for dimension</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="text"
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={draft}
                placeholder="null"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                aria-label="Flatten except for dimension (empty or null = flatten all)"
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commitDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.currentTarget.blur();
                }}
              />
            </div>
          </div>
        </div>
        <p className="cr-node__hint">
          Empty or null: one axis with the full product. Integer: keep that axis; merge the others (0-based, negatives
          allowed).
        </p>
      </div>
    </div>
  );
}
