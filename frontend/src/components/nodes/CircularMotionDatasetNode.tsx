import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_CIRCULAR_MOTION_DATASET_PARAM_ORDER,
  DEFAULT_CIRCULAR_MOTION_DATASET_SPEC_NAME,
  generateCircularMotionDatasetSpecCode,
} from "../../graph/specCode/circularMotionDatasetSpecCode";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultCircularMotionDatasetData,
  type CircularMotionDatasetNodeData,
} from "./circularMotionDatasetDefaults";

function replaceNodeData(
  id: string,
  data: CircularMotionDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function CircularMotionDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultCircularMotionDatasetData();
  const d = { ...defs, ...(data as Partial<CircularMotionDatasetNodeData>) } as CircularMotionDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_CIRCULAR_MOTION_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_CIRCULAR_MOTION_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateCircularMotionDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<CircularMotionDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="circular_motion_dataset"
        nodeId={id}
        graphNodeType="circular_motion_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Circular motion dataset")}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <DatasetSourceSockets />
        <DiscreteMultiSelect<DatasetSamplingMode>
          label="train data sampling"
          options={DATASET_SAMPLING_MODE_OPTIONS}
          value={d.samplingMode ?? "fixed"}
          onCommit={(v) =>
            update({
              samplingMode: (typeof v === "string" ? v : v[0] ?? "fixed") as DatasetSamplingMode,
            })
          }
          ariaLabel="Train data sampling mode for the trainer"
          singleSelect
        />
        <ComfyIntListField
          label="token bins $n$"
          ariaLabel="token bins n"
          values={intChoices(d.vocabSize, 128)}
          min={2}
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="context length $l$ (timesteps)"
          ariaLabel="context length L timesteps"
          values={intChoices(d.contextLength, 20)}
          min={1}
          title="Each timestep emits two token ids (x, y); context tensor shape [batch, L, 2]."
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
        />
        <ComfyFloatListField
          label="radius min"
          ariaLabel="radius min"
          values={floatChoices(d.radiusMin, 0.15)}
          min={0}
          positiveOnly={false}
          onCommit={(vals) => update({ radiusMin: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="radius max"
          ariaLabel="radius max"
          values={floatChoices(d.radiusMax, 0.35)}
          min={0}
          positiveOnly={false}
          onCommit={(vals) => update({ radiusMax: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="angular velocity ω"
          ariaLabel="angular velocity ω"
          values={floatChoices(d.angularVelocity, 0.5)}
          positiveOnly={false}
          onCommit={(vals) => update({ angularVelocity: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="train size"
          ariaLabel="train size"
          values={intChoices(d.trainSize, 4000)}
          min={1}
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          ariaLabel="test size"
          values={intChoices(d.testSize, 1000)}
          min={0}
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="seed"
          ariaLabel="seed"
          values={intChoices(d.seed, 0)}
          min={0}
          title="Init seed for radius/phase sampling."
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
