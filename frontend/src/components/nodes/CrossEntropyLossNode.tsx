import { useEffect, useState } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import {
  defaultCrossEntropyLossData,
  type CrossEntropyLossNodeData,
} from "./crossEntropyLossDefaults";
import type { MseLossMaskModeId } from "./mseLossDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";

function patchData(
  id: string,
  prev: CrossEntropyLossNodeData,
  patch: Partial<CrossEntropyLossNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
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

export function CrossEntropyLossNode({ id, data, selected }: NodeProps) {
  const defs = defaultCrossEntropyLossData();
  const d = { ...defs, ...(data as Partial<CrossEntropyLossNodeData>) } as CrossEntropyLossNodeData;
  const { setNodes } = useReactFlow();
  const [maskDraft, setMaskDraft] = useState(d.lossMaskCustom ?? "");
  useEffect(() => {
    setMaskDraft(d.lossMaskCustom ?? "");
  }, [d.lossMaskCustom]);

  const update = (patch: Partial<CrossEntropyLossNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--mse-loss${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">Cross-entropy loss</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="loss" label="loss" />
        <ComfyFloatListField
          label="loss scale"
          values={floatChoices(d.lossScale, 1)}
          positiveOnly
          title="Multiplies CE in training and logs — comma-separated for multiple runs"
          ariaLabel="Loss scale"
          onCommit={(vals) => update({ lossScale: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="label smoothing"
          values={floatChoices(d.labelSmoothing, 0)}
          positiveOnly={false}
          title="PyTorch label smoothing for CE (0 = off). Clamped to [0, 1] in the trainer."
          ariaLabel="Label smoothing"
          onCommit={(vals) => update({ labelSmoothing: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="mask context length (T)"
          values={intChoices(d.lossMaskContextLength, 1)}
          min={1}
          title="Split flat logits [batch, T×V] into T slots of V-way CE each (same label y per slot). Use with memorization A: set dataset outputDim=V and model outputDim=T×V when masking."
          ariaLabel="Cross-entropy mask context length"
          onCommit={(vals) => update({ lossMaskContextLength: packIntList(vals) })}
        />
        <DiscreteMultiSelect
          label="mask along context"
          options={LOSS_MASK_MODE_OPTIONS}
          value={d.lossMaskMode}
          ariaLabel="Cross-entropy loss mask along context axis"
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
                placeholder="T comma-separated slot weights (when T>1 and mode is custom)"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                disabled={d.lossMaskMode !== "custom"}
                aria-label="Comma-separated cross-entropy slot mask weights"
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
