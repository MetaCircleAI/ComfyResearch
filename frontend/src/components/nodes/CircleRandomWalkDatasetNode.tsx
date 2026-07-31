import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_CIRCLE_RANDOM_WALK_DATASET_PARAM_ORDER,
  DEFAULT_CIRCLE_RANDOM_WALK_DATASET_SPEC_NAME,
  generateCircleRandomWalkDatasetSpecCode,
} from "../../graph/specCode/circleRandomWalkDatasetSpecCode";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultCircleRandomWalkDatasetData,
  type CircleRandomWalkDatasetNodeData,
} from "./circleRandomWalkDatasetDefaults";

function replaceNodeData(
  id: string,
  data: CircleRandomWalkDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function CircleRandomWalkDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultCircleRandomWalkDatasetData();
  const d = { ...defs, ...(data as Partial<CircleRandomWalkDatasetNodeData>) } as CircleRandomWalkDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_CIRCLE_RANDOM_WALK_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_CIRCLE_RANDOM_WALK_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateCircleRandomWalkDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<CircleRandomWalkDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="circle_random_walk_dataset"
        nodeId={id}
        graphNodeType="circle_random_walk_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Circle random walk dataset")}
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
          label="vocab size $v$"
          ariaLabel="vocab size v"
          values={intChoices(d.vocabSize, 10)}
          min={2}
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="context length $l$"
          ariaLabel="context length l"
          values={intChoices(d.contextLength, 1)}
          min={1}
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
        />
        <ComfyFloatListField
          label="step prob right"
          ariaLabel="step prob right"
          values={floatChoices(d.rightStepProb, 0.5)}
          min={0}
          max={1}
          positiveOnly={false}
          onCommit={(vals) => update({ rightStepProb: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="train size"
          ariaLabel="train size"
          values={intChoices(d.trainSize, 800)}
          min={1}
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          ariaLabel="test size"
          values={intChoices(d.testSize, 200)}
          min={0}
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="seed"
          ariaLabel="seed"
          values={intChoices(d.seed, 0)}
          min={0}
          title="Init seed for walk sampling."
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
