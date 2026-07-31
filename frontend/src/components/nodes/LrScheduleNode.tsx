import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import { defaultLrScheduleData, type LrScheduleKindId, type LrScheduleNodeData } from "./lrScheduleDefaults";

const LR_SCHEDULE_OPTIONS: { id: LrScheduleKindId; label: string }[] = [
  { id: "constant", label: "constant LR" },
  { id: "cosine", label: "warmup + cosine decay" },
  {
    id: "stable_stable_decay",
    label: "stable–stable–decay",
  },
  { id: "exponential_epoch", label: "exponential decay by epoch" },
];

function patchData(
  id: string,
  prev: LrScheduleNodeData,
  patch: Partial<LrScheduleNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function LrScheduleNode({ id, data, selected }: NodeProps) {
  const defs = defaultLrScheduleData();
  const d = { ...defs, ...(data as Partial<LrScheduleNodeData>) } as LrScheduleNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<LrScheduleNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--lr-schedule${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-optimizer)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "LR schedule")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="lr_schedule" label="lr schedule" />
        <ComfyIntField
          label="LR warmup steps"
          value={d.lrWarmupSteps}
          min={0}
          title="Linear LR warmup from 0→base over this many steps (then constant LR or cosine decay per schedule)."
          onCommit={(lrWarmupSteps) => update({ lrWarmupSteps })}
          ariaLabel="Learning rate warmup steps"
        />
        <DiscreteMultiSelect<LrScheduleKindId>
          label="LR schedule"
          options={LR_SCHEDULE_OPTIONS}
          singleSelect
          value={d.lrSchedule}
          onCommit={(lrSchedule) => update({ lrSchedule })}
          ariaLabel="Learning rate schedule"
        />
        <ComfyFloatField
          label="cosine LR min fraction"
          value={d.cosineLrMinFraction}
          min={0}
          title="Cosine tail floor: after warmup, cosine schedules decay down to this fraction of base LR per group; stable–stable–decay uses it for the final third only."
          onCommit={(cosineLrMinFraction) =>
            update({ cosineLrMinFraction: Math.min(1, Math.max(0, cosineLrMinFraction)) })
          }
          ariaLabel="Cosine minimum LR fraction"
        />
        {d.lrSchedule === "exponential_epoch" && (
          <>
            <ComfyFloatField
              label="exponential decay factor"
              value={d.exponentialDecayFactor}
              min={0}
              max={1}
              title="Multiply the base learning rate by this factor after each decay interval."
              onCommit={(exponentialDecayFactor) => update({ exponentialDecayFactor: Math.min(1, Math.max(0, exponentialDecayFactor)) })}
              ariaLabel="Exponential learning rate decay factor"
            />
            <ComfyIntField
              label="decay every epochs"
              value={d.exponentialDecayEpochs}
              min={1}
              title="Apply exponential decay after this many complete training epochs."
              onCommit={(exponentialDecayEpochs) => update({ exponentialDecayEpochs: Math.max(1, exponentialDecayEpochs) })}
              ariaLabel="Exponential learning rate decay interval in epochs"
            />
          </>
        )}
        <p className="cr-observable-hint">
          Time-dependent LR only (warmup, cosine, etc.). Wire to the optimizer&apos;s <strong>lr schedule</strong>{' '}
          socket (Adam, SGD, or Muon). You can also wire a <strong>mup_lr_schedule</strong> node to that same socket for
          μP multipliers (Adam + supported models). If no lr_schedule is attached, training uses constant LR (no
          warmup).
        </p>
      </div>
    </div>
  );
}
