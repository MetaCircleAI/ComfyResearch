import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_ICAR_PARAM_ORDER,
  DEFAULT_ICAR_SPEC_NAME,
  generateInContextAssociativeRecallDatasetSpecCode,
} from "../../graph/specCode/inContextAssociativeRecallDatasetSpecCode";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultInContextAssociativeRecallDatasetData,
  type InContextAssociativeRecallDatasetNodeData,
} from "./inContextAssociativeRecallDatasetDefaults";

function replaceNodeData(
  id: string,
  data: InContextAssociativeRecallDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function InContextAssociativeRecallDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultInContextAssociativeRecallDatasetData();
  const d = {
    ...defs,
    ...(data as Partial<InContextAssociativeRecallDatasetNodeData>),
  } as InContextAssociativeRecallDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_ICAR_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_ICAR_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateInContextAssociativeRecallDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<InContextAssociativeRecallDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="in_context_associative_recall_dataset"
        nodeId={id}
        graphNodeType="in_context_associative_recall_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "In-context associative recall dataset")}
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
          values={intChoices(d.vocabSize, 64)}
          min={2}
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label={String.raw`number pairs $n_{\mathrm{pairs}}$`}
          ariaLabel="number pairs n pairs"
          values={intChoices(d.numPairs, 32)}
          min={1}
          title="Sequence length is 2 * Npairs + 1"
          onCommit={(vals) => update({ numPairs: packIntList(vals) })}
        />
        <ComfyIntListField
          label="in-context repeat $b$"
          ariaLabel="in-context repeat b"
          values={intChoices(d.inContextRepeat, 1)}
          min={1}
          title="Expected repeats of query token as key inside context"
          onCommit={(vals) => update({ inContextRepeat: packIntList(vals) })}
        />
        <ComfyFloatListField
          label="cross-sample repeat $p$"
          ariaLabel="cross-sample repeat p"
          values={Array.isArray(d.crossSampleRepeatProb) ? d.crossSampleRepeatProb : [d.crossSampleRepeatProb]}
          min={0}
          max={1}
          title="Probability query is sampled from the repeated-token subset"
          onCommit={(vals) => update({ crossSampleRepeatProb: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="repeated token count"
          ariaLabel="repeated token count"
          values={intChoices(d.repeatedTokenCount, 2)}
          min={1}
          title="Size of repeated token subset used by cross-sample repetition"
          onCommit={(vals) => update({ repeatedTokenCount: packIntList(vals) })}
        />
        <ComfyIntListField
          label="train size"
          ariaLabel="train size"
          values={intChoices(d.trainSize, 10_000)}
          min={1}
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          ariaLabel="test size"
          values={intChoices(d.testSize, 2_000)}
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
