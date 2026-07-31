import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import {
  defaultCyclicLrScheduleData,
  type CyclicLrScheduleNodeData,
  type CyclicScheduleModeId,
} from "./cyclicLrScheduleDefaults";

const SCHEDULE_MODE_OPTIONS: { id: CyclicScheduleModeId; label: string }[] = [
  { id: "discrete_epoch", label: "discrete per epoch (paper)" },
  { id: "triangular_step", label: "triangular per step" },
];

function patchData(
  id: string,
  prev: CyclicLrScheduleNodeData,
  patch: Partial<CyclicLrScheduleNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function CyclicLrScheduleNode({ id, data, selected }: NodeProps) {
  const defs = defaultCyclicLrScheduleData();
  const d = { ...defs, ...(data as Partial<CyclicLrScheduleNodeData>) } as CyclicLrScheduleNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<CyclicLrScheduleNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--cyclic-lr-schedule${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-optimizer)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "Cyclic LR")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="lr_schedule" label="lr schedule" />
        <DiscreteMultiSelect<CyclicScheduleModeId>
          label="schedule mode"
          options={SCHEDULE_MODE_OPTIONS}
          singleSelect
          value={d.scheduleMode ?? "discrete_epoch"}
          onCommit={(scheduleMode) => update({ scheduleMode })}
          ariaLabel="Cyclic schedule mode"
        />
        <ComfyFloatField
          label="LR min"
          value={d.lrMin}
          min={0}
          onCommit={(lrMin) => update({ lrMin })}
          ariaLabel="Cyclic LR minimum"
        />
        <ComfyFloatField
          label="LR max"
          value={d.lrMax}
          min={0}
          onCommit={(lrMax) => update({ lrMax })}
          ariaLabel="Cyclic LR maximum"
        />
        <ComfyIntField
          label="cycle length (epochs)"
          value={d.cycleLengthEpochs}
          min={1}
          title="Triangular cycle period in epochs (converted to steps at train time using ref batch)."
          onCommit={(cycleLengthEpochs) => update({ cycleLengthEpochs })}
          ariaLabel="Cyclic LR cycle length epochs"
        />
        <ComfyIntField
          label="ref batch (epoch→steps)"
          value={d.refBatchSize}
          min={1}
          onCommit={(refBatchSize) => update({ refBatchSize: Math.max(1, refBatchSize) })}
          ariaLabel="Reference batch size for epoch conversion"
        />
        <ComfyIntField
          label="cycle length (steps)"
          value={d.cycleLengthSteps}
          min={0}
          title="Optional override; 0 = use epochs × ceil(trainSize / ref batch)."
          onCommit={(cycleLengthSteps) => update({ cycleLengthSteps: Math.max(0, cycleLengthSteps) })}
          ariaLabel="Cyclic LR cycle length steps override"
        />
        <p className="cr-observable-hint">
          Jastrzębski Fig 1: use <strong>discrete per epoch</strong> (η constant within each epoch, switches at epoch
          boundaries). Wire to optimizer <strong>lr schedule</strong> socket.
        </p>
      </div>
    </div>
  );
}
