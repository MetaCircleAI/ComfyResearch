import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import {
  defaultCyclicBatchScheduleData,
  type CyclicBatchScheduleNodeData,
  type CyclicScheduleModeId,
} from "./cyclicBatchScheduleDefaults";

const SCHEDULE_MODE_OPTIONS: { id: CyclicScheduleModeId; label: string }[] = [
  { id: "discrete_epoch", label: "discrete per epoch (paper)" },
  { id: "triangular_step", label: "triangular per step" },
];

function patchData(
  id: string,
  prev: CyclicBatchScheduleNodeData,
  patch: Partial<CyclicBatchScheduleNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function CyclicBatchScheduleNode({ id, data, selected }: NodeProps) {
  const defs = defaultCyclicBatchScheduleData();
  const d = { ...defs, ...(data as Partial<CyclicBatchScheduleNodeData>) } as CyclicBatchScheduleNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<CyclicBatchScheduleNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--cyclic-batch-schedule${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-optimizer)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "Cyclic batch")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="batch_schedule" label="batch schedule" />
        <DiscreteMultiSelect<CyclicScheduleModeId>
          label="schedule mode"
          options={SCHEDULE_MODE_OPTIONS}
          singleSelect
          value={d.scheduleMode ?? "discrete_epoch"}
          onCommit={(scheduleMode) => update({ scheduleMode })}
          ariaLabel="Cyclic batch schedule mode"
        />
        <ComfyIntField
          label="batch min"
          value={d.batchMin}
          min={1}
          onCommit={(batchMin) => update({ batchMin: Math.max(1, batchMin) })}
          ariaLabel="Cyclic batch minimum"
        />
        <ComfyIntField
          label="batch max"
          value={d.batchMax}
          min={1}
          onCommit={(batchMax) => update({ batchMax: Math.max(1, batchMax) })}
          ariaLabel="Cyclic batch maximum"
        />
        <ComfyIntField
          label="cycle length (epochs)"
          value={d.cycleLengthEpochs}
          min={1}
          onCommit={(cycleLengthEpochs) => update({ cycleLengthEpochs })}
          ariaLabel="Cyclic batch cycle length epochs"
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
          ariaLabel="Cyclic batch cycle length steps override"
        />
        <p className="cr-observable-hint">
          Jastrzębski Fig 1: use <strong>discrete per epoch</strong> (B constant within each epoch). Wire to trainer{" "}
          <strong>batch schedule</strong> socket.
        </p>
      </div>
    </div>
  );
}
