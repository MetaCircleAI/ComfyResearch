import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField } from "./comfyNumberFields";
import { SourceSocketRow } from "./SourceSocketRow";
import { defaultMupLrScheduleData, type MupLrScheduleNodeData } from "./mupLrScheduleDefaults";

function patchData(
  id: string,
  prev: MupLrScheduleNodeData,
  patch: Partial<MupLrScheduleNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function MupLrScheduleNode({ id, data, selected }: NodeProps) {
  const defs = defaultMupLrScheduleData();
  const d = { ...defs, ...(data as Partial<MupLrScheduleNodeData>) } as MupLrScheduleNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<MupLrScheduleNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--mup-lr-schedule${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-optimizer)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "MuP LR schedule")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="mup_lr_schedule" label="μP LR schedule" />
        <ComfyFloatField
          label={String.raw`$\mu$P embed LR mult`}
          value={d.mupEmbedLrMult}
          min={0}
          title="Adam LR multiplier for embedding parameters (relative to optimizer base LR)."
          onCommit={(mupEmbedLrMult) => update({ mupEmbedLrMult: Math.max(0, mupEmbedLrMult) })}
          ariaLabel="MuP embedding LR multiplier"
        />
        <ComfyFloatField
          label={String.raw`$\mu$P hidden LR mult`}
          value={d.mupHiddenLrMult}
          min={0}
          title="Adam LR multiplier for hidden layers."
          onCommit={(mupHiddenLrMult) => update({ mupHiddenLrMult: Math.max(0, mupHiddenLrMult) })}
          ariaLabel="MuP hidden LR multiplier"
        />
        <ComfyFloatField
          label={String.raw`$\mu$P output LR mult`}
          value={d.mupOutputLrMult}
          min={0}
          title="Adam LR multiplier for lm_head / output weights."
          onCommit={(mupOutputLrMult) => update({ mupOutputLrMult: Math.max(0, mupOutputLrMult) })}
          ariaLabel="MuP output LR multiplier"
        />
        <p className="cr-observable-hint">
          Layer-dependent Adam LR grouping — wire to the optimizer&apos;s <strong>lr schedule</strong> socket (same
          port as the time-dependent schedule). Optionally add an <strong>lr_schedule</strong> node to the same socket
          for warmup / cosine / etc.
        </p>
      </div>
    </div>
  );
}
