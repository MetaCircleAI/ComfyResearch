import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultTokenPredictionDatasetData,
  type TokenPredictionDatasetNodeData,
} from "./tokenPredictionDatasetDefaults";
import { enumChoices, intChoices, packEnumList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_TOKEN_PREDICTION_PARAM_ORDER,
  DEFAULT_TOKEN_PREDICTION_SPEC_NAME,
  generateTokenPredictionDatasetSpecCode,
} from "../../graph/specCode/tokenPredictionDatasetSpecCode";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

const RETRIEVAL_MODE_OPTIONS = [
  { id: "position", label: "position (which token index)" },
  { id: "content", label: "content (nearest prior token to last token)" },
] as const;
const RETRIEVAL_MODE_SET = new Set(["position", "content"] as const);

function replaceNodeData(
  id: string,
  data: TokenPredictionDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: TokenPredictionDatasetNodeData,
  patch: Partial<TokenPredictionDatasetNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: TokenPredictionDatasetNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_TOKEN_PREDICTION_PARAM_ORDER];
}

export function TokenPredictionDatasetNode({ id, data, selected, type }: NodeProps) {
  const defs = defaultTokenPredictionDatasetData();
  const d = {
    ...defs,
    ...(data as Partial<TokenPredictionDatasetNodeData>),
  } as TokenPredictionDatasetNodeData;
  const { setNodes } = useReactFlow();

  const retrievalMode = enumChoices(d.retrievalMode, RETRIEVAL_MODE_SET, "position")[0];
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_TOKEN_PREDICTION_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateTokenPredictionDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<TokenPredictionDatasetNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="token_prediction_dataset"
        nodeId={id}
        graphNodeType={type ?? "token_prediction_dataset"}
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(
          d,
          retrievalMode === "content"
            ? "Token Retrieval (content) dataset"
            : "Token Retrieval (position) dataset",
        )}
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
        <DiscreteMultiSelect
          label="retrieval mode"
          options={RETRIEVAL_MODE_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
          value={retrievalMode}
          singleSelect
          onCommit={(v) => {
            const raw = typeof v === "string" ? v : v[0] ?? "position";
            const nextMode = raw === "content" ? "content" : "position";
            update({ retrievalMode: packEnumList([nextMode]) });
          }}
          ariaLabel="Token retrieval mode position or content"
        />
        <ComfyIntListField
          label="vocab size $v$"
          values={intChoices(d.vocabSize, 4)}
          min={2}
          title="Integer tokens in [0, V − 1]"
          ariaLabel="Vocabulary size"
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="context length $l$"
          values={intChoices(d.contextLength, 4)}
          min={1}
          title="Sequence length"
          ariaLabel="Context length"
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
        />
        {retrievalMode === "position" ? (
          <ComfyIntListField
            label="which token"
            values={intChoices(d.whichToken, -1)}
            title="Python indexing target token. For L=4: 0,1,2,3 or -1,-2,-3,-4."
            ariaLabel="Which token index"
            onCommit={(vals) => update({ whichToken: packIntList(vals) })}
          />
        ) : null}
        <ComfyIntListField
          label="train size"
          values={intChoices(d.trainSize, 800)}
          min={1}
          ariaLabel="Train size"
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          values={intChoices(d.testSize, 200)}
          min={0}
          ariaLabel="Test size"
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="seed"
          values={intChoices(d.seed, 0)}
          ariaLabel="Random seed"
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
        {d.extras && Object.keys(d.extras).length > 0 ? (
          <p className="cr-node__hint cr-node__hint--extras">
            Extra params from spec (not used by training): {JSON.stringify(d.extras)}
          </p>
        ) : null}
      </div>
    </div>
  );
}
