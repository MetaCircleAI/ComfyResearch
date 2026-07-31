import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import {
  BIGRAM_SPECTRUM_DECAY_OPTIONS,
  type BigramSpectrumDecayId,
} from "./bigramLowRankDatasetDefaults";
import { floatChoices, intChoices, packEnumList, packFloatList, packIntList, type ListOr1 } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_BIGRAM_LOW_RANK_DATASET_PARAM_ORDER,
  DEFAULT_BIGRAM_LOW_RANK_DATASET_SPEC_NAME,
  generateBigramLowRankDatasetSpecCode,
} from "../../graph/specCode/bigramLowRankDatasetSpecCode";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultBigramLowRankDatasetData,
  type BigramLowRankDatasetNodeData,
} from "./bigramLowRankDatasetDefaults";

function replaceNodeData(
  id: string,
  data: BigramLowRankDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

type LegacyBigramData = Partial<BigramLowRankDatasetNodeData> & { rankDecay?: ListOr1<number> };

export function BigramLowRankDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultBigramLowRankDatasetData();
  const raw = data as LegacyBigramData;
  const alphaFromLegacy =
    raw.alpha === undefined && raw.rankDecay !== undefined ? { alpha: raw.rankDecay } : {};
  const { rankDecay: _dropLegacy, ...rawClean } = raw;
  const d = { ...defs, ...rawClean, ...alphaFromLegacy } as BigramLowRankDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_BIGRAM_LOW_RANK_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_BIGRAM_LOW_RANK_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateBigramLowRankDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<BigramLowRankDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="bigram_low_rank_dataset"
        nodeId={id}
        graphNodeType="bigram_low_rank_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Bigram low-rank dataset")}
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
        <ComfyIntListField
          label="rank $r$"
          ariaLabel="rank r"
          values={intChoices(d.rank, 20)}
          min={1}
          onCommit={(vals) => update({ rank: packIntList(vals) })}
        />
        <ComfyFloatListField
          label="logit scale"
          ariaLabel="logit scale"
          values={floatChoices(d.logitScale, 1.0)}
          onCommit={(vals) => update({ logitScale: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="corrupt ratio"
          ariaLabel="corrupt ratio"
          values={floatChoices(d.corruptRatio, 0.0)}
          min={0}
          max={1}
          title="Fraction of samples whose logits are replaced by random Gaussian noise before drawing target tokens."
          onCommit={(vals) => update({ corruptRatio: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="corrupt scale"
          ariaLabel="corrupt scale"
          values={floatChoices(d.corruptScale, 5.0)}
          min={0}
          title="Scale multiplier for Gaussian corruption logits. Ignored when corrupt ratio is 0."
          onCommit={(vals) => update({ corruptScale: packFloatList(vals) })}
        />
        <DiscreteMultiSelect<BigramSpectrumDecayId>
          label="spectrum decay"
          ariaLabel="spectrum decay type"
          options={BIGRAM_SPECTRUM_DECAY_OPTIONS}
          value={d.decayType}
          singleSelect
          onCommit={(next) =>
            update({ decayType: packEnumList(Array.isArray(next) ? next : [next]) })
          }
        />
        <ComfyFloatListField
          label={String.raw`decay rate $\alpha$`}
          ariaLabel="decay rate alpha"
          values={floatChoices(d.alpha, 0.0)}
          min={0}
          title="Weights λ_n on rank columns of A: power law n^{-α} or exponential e^{-α n} (n = 1 … R). α = 0 disables decay."
          onCommit={(vals) => update({ alpha: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="train size"
          ariaLabel="train size"
          values={intChoices(d.trainSize, 1200)}
          min={1}
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          ariaLabel="test size"
          values={intChoices(d.testSize, 300)}
          min={0}
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="sample seed"
          ariaLabel="sample seed"
          values={intChoices(d.seed, 0)}
          min={0}
          title="Seed for drawing token pairs from the stationary distribution and transition rows."
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
        <ComfyIntListField
          label="init seed"
          ariaLabel="init seed"
          values={intChoices(d.initSeed, 0)}
          min={0}
          title="Seed for low-rank transition factors A and B."
          onCommit={(vals) => update({ initSeed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
