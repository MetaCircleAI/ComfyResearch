import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_UNIGRAM_DATASET_PARAM_ORDER,
  DEFAULT_UNIGRAM_DATASET_SPEC_NAME,
  generateUnigramDatasetSpecCode,
} from "../../graph/specCode/unigramDatasetSpecCode";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultUnigramDatasetData,
  type UnigramDatasetNodeData,
  UNIGRAM_OUTPUT_DISTRIBUTION_OPTIONS,
} from "./unigramDatasetDefaults";

function replaceNodeData(
  id: string,
  data: UnigramDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function UnigramDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultUnigramDatasetData();
  const d = { ...defs, ...(data as Partial<UnigramDatasetNodeData>) } as UnigramDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_UNIGRAM_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_UNIGRAM_DATASET_SPEC_NAME;
  const generatedCode = useMemo(() => generateUnigramDatasetSpecCode(d, order, specName), [d, order, specName]);
  const update = useCallback(
    (patch: Partial<UnigramDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="unigram_dataset"
        nodeId={id}
        graphNodeType="unigram_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Unigram dataset")}
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
          values={intChoices(d.vocabSize, 100)}
          min={2}
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <DiscreteMultiSelect
          label="output distribution"
          options={UNIGRAM_OUTPUT_DISTRIBUTION_OPTIONS}
          value={d.outputDistribution}
          onCommit={(outputDistribution) => update({ outputDistribution })}
          ariaLabel="Unigram token distribution"
        />
        <ComfyFloatListField
          label={String.raw`decay rate $\alpha$`}
          ariaLabel="decay rate alpha"
          values={Array.isArray(d.alpha) ? d.alpha : [d.alpha]}
          min={0.01}
          title="Exponent for power-law (∝ rank^−α) or rate for exponential (∝ exp(−α·rank)); ignored for uniform"
          onCommit={(vals) => update({ alpha: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="context length $l$"
          ariaLabel="context length l"
          values={intChoices(d.contextLength, 1)}
          min={1}
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
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
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
