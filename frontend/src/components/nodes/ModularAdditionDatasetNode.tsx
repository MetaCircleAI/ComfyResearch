import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_MODULAR_ADDITION_DATASET_PARAM_ORDER,
  DEFAULT_MODULAR_ADDITION_DATASET_SPEC_NAME,
  generateModularAdditionDatasetSpecCode,
} from "../../graph/specCode/modularAdditionDatasetSpecCode";
import {
  defaultModularAdditionDatasetData,
  type ModularAdditionDatasetNodeData,
} from "./modularAdditionDatasetDefaults";

function replaceNodeData(
  id: string,
  data: ModularAdditionDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function ModularAdditionDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultModularAdditionDatasetData();
  const d = { ...defs, ...(data as Partial<ModularAdditionDatasetNodeData>) } as ModularAdditionDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_MODULAR_ADDITION_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_MODULAR_ADDITION_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateModularAdditionDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<ModularAdditionDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="modular_addition_dataset"
        nodeId={id}
        graphNodeType="modular_addition_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Modular addition dataset")}
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
          label="modulus $p$"
          ariaLabel="modulus p"
          values={intChoices(d.modulus, 59)}
          min={2}
          onCommit={(vals) => update({ modulus: packIntList(vals) })}
        />
        <ComfyFloatListField
          label="train fraction"
          ariaLabel="train fraction"
          values={floatChoices(d.trainFraction, 0.3)}
          min={0}
          max={1}
          positiveOnly={false}
          onCommit={(vals) => update({ trainFraction: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="seed"
          ariaLabel="seed"
          values={intChoices(d.seed, 0)}
          min={0}
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
