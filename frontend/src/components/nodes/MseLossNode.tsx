import { useEffect, useState } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { defaultMseLossData, type MseLossMaskModeId, type MseLossNodeData } from "./mseLossDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";

function patchData(
  id: string,
  prev: MseLossNodeData,
  patch: Partial<MseLossNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

const LOSS_MASK_MODE_OPTIONS: { id: MseLossMaskModeId; label: string }[] = [
  { id: "all", label: "all context slots" },
  { id: "last_context", label: "last context slot only" },
  { id: "custom", label: "custom (comma weights)" },
];

export function MseLossNode({ id, data, selected }: NodeProps) {
  const defs = defaultMseLossData();
  const d = { ...defs, ...(data as Partial<MseLossNodeData>) } as MseLossNodeData;
  const { setNodes } = useReactFlow();
  const [maskDraft, setMaskDraft] = useState(d.lossMaskCustom ?? "");
  useEffect(() => {
    setMaskDraft(d.lossMaskCustom ?? "");
  }, [d.lossMaskCustom]);

  const update = (patch: Partial<MseLossNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--mse-loss${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">MSE Loss</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="loss scale"
          values={floatChoices(d.lossScale, 1)}
          positiveOnly
          title="Multiplies MSE in training and logs — comma-separated for multiple runs"
          ariaLabel="Loss scale"
          onCommit={(vals) => update({ lossScale: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="mask context length (T)"
          values={intChoices(d.lossMaskContextLength, 1)}
          min={1}
          title="Group flat model output as T blocks of (dim / T) consecutive features (time-major). Use 2 with uniform motion to mask along x1,x2 steps."
          ariaLabel="MSE mask context length"
          onCommit={(vals) => update({ lossMaskContextLength: packIntList(vals) })}
        />
        <DiscreteMultiSelect
          label="mask along context"
          options={LOSS_MASK_MODE_OPTIONS}
          value={d.lossMaskMode}
          ariaLabel="MSE loss mask along context axis"
          singleSelect
          onCommit={(lossMaskMode) => update({ lossMaskMode })}
        />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">custom mask weights</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="text"
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={maskDraft}
                placeholder="T comma weights if T>1, else one weight per flat output dim"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                disabled={d.lossMaskMode !== "custom"}
                aria-label="Comma-separated MSE mask weights"
                onChange={(e) => setMaskDraft(e.target.value)}
                onBlur={() => update({ lossMaskCustom: maskDraft })}
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
