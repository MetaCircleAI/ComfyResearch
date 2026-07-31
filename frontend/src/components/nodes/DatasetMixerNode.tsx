import { useCallback, useMemo } from "react";
import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  DEFAULT_DATASET_MIXER_PARAM_ORDER,
  DEFAULT_DATASET_MIXER_SPEC_NAME,
  generateDatasetMixerSpecCode,
} from "../../graph/specCode/datasetMixerSpecCode";
import { normalizeDatasetMixerData, type DatasetMixerNodeData } from "./datasetMixerDefaults";

function replaceNodeData(
  id: string,
  data: DatasetMixerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function DatasetMixerNode({ id, data, selected }: NodeProps) {
  const d = normalizeDatasetMixerData((data ?? {}) as Record<string, unknown>);
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_DATASET_MIXER_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_DATASET_MIXER_SPEC_NAME;
  const generatedCode = useMemo(() => generateDatasetMixerSpecCode(d, order, specName), [d, order, specName]);
  const update = useCallback(
    (patch: Partial<DatasetMixerNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--dataset-mixer-a${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="dataset_mixer_a"
        nodeId={id}
        graphNodeType="dataset_mixer"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Dataset mixer A")}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Dataset mixer A I/O">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset_a"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset A</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset_b"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset B</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">dataset</span>
              <Handle
                type="source"
                position={Position.Right}
                id="dataset"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--dataset"
              />
            </div>
          </div>
        </div>
        <ComfyIntListField
          label={String.raw`total train samples $N_{\mathrm{train}}$`}
          ariaLabel="total train sample count"
          values={intChoices(d.trainTotalSamples, 800)}
          min={1}
          onCommit={(vals) => update({ trainTotalSamples: packIntList(vals) })}
        />
        <ComfyIntListField
          label={String.raw`total test samples $N_{\mathrm{test}}$`}
          ariaLabel="total test sample count"
          values={intChoices(d.testTotalSamples, 0)}
          min={0}
          onCommit={(vals) => update({ testTotalSamples: packIntList(vals) })}
        />
        <ComfyFloatListField
          label={String.raw`proportion dataset A $p_{\mathrm{A}}$`}
          ariaLabel="proportion for dataset A"
          values={floatChoices(d.proportionA, 0.5)}
          min={0}
          max={1}
          positiveOnly={false}
          title="Dataset B proportion is computed as 1 - p_A"
          onCommit={(vals) => update({ proportionA: packFloatList(vals) })}
        />
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
          label="init seed"
          ariaLabel="dataset mixer init seed"
          values={intChoices(d.initSeed, 0)}
          min={0}
          title="Random permutation seed used after concatenating sampled A/B examples."
          onCommit={(vals) => update({ initSeed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
