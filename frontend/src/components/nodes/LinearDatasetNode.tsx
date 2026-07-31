import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { DatasetSamplingMode, LinearDatasetNodeData } from "./linearDatasetDefaults";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  defaultLinearDatasetData,
  INPUT_DISTRIBUTION_OPTIONS,
  MEMORIZATION_A_DATASET_SPEC_NAME,
  MEMORIZATION_B_DATASET_SPEC_NAME,
  MEMORIZATION_OUTPUT_DISTRIBUTION_OPTIONS,
  OUTPUT_DISTRIBUTION_OPTIONS,
} from "./linearDatasetDefaults";
import { ComfyIntField } from "./comfyNumberFields";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_LINEAR_DATASET_PARAM_ORDER,
  DEFAULT_LINEAR_DATASET_SPEC_NAME,
  DEFAULT_MEMORIZATION_A_DATASET_PARAM_ORDER,
  DEFAULT_MEMORIZATION_B_DATASET_PARAM_ORDER,
  generateLinearDatasetSpecCode,
} from "../../graph/specCode/linearDatasetSpecCode";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { defaultInstanceTitleBase, readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import type { DatasetNodeInfoKind } from "./datasetNodeInfoContent";

function replaceNodeData(
  id: string,
  data: LinearDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: LinearDatasetNodeData,
  patch: Partial<LinearDatasetNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(
  d: LinearDatasetNodeData,
  memKind: "none" | "a" | "b",
  graphType: string,
): string[] {
  const fallback =
    memKind === "b"
      ? DEFAULT_MEMORIZATION_B_DATASET_PARAM_ORDER
      : memKind === "a"
        ? DEFAULT_MEMORIZATION_A_DATASET_PARAM_ORDER
        : DEFAULT_LINEAR_DATASET_PARAM_ORDER;
  const raw = d.paramOrder?.length ? d.paramOrder : [...fallback];
  if (memKind === "none") return raw.filter((k) => k === "alpha" ? false : true);
  if (memKind === "b") return raw.filter((k) => k === "inputDistribution" ? false : true);
  return raw;
}

export function LinearDatasetNode({ id, data, selected, type }: NodeProps) {
  const defs = defaultLinearDatasetData();
  const d = { ...defs, ...(data as Partial<LinearDatasetNodeData>) } as LinearDatasetNodeData;
  const { setNodes, getNodes } = useReactFlow();
  const memKind = type === "memorization_b_dataset" ? "b" : type === "memorization_a_dataset" ? "a" : "none";
  const graphType = type ?? "linear_dataset";

  const order = useMemo(() => effectiveParamOrder(d, memKind, graphType), [d, memKind, graphType]);
  const specName =
    d.specCodeName ??
    (graphType === "memorization_a_dataset"
      ? MEMORIZATION_A_DATASET_SPEC_NAME
      : graphType === "memorization_b_dataset"
        ? MEMORIZATION_B_DATASET_SPEC_NAME
        : DEFAULT_LINEAR_DATASET_SPEC_NAME);
  const generatedCode = useMemo(
    () => generateLinearDatasetSpecCode(d, order, specName, graphType),
    [d, order, specName, graphType],
  );

  const update = useCallback(
    (patch: Partial<LinearDatasetNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "inputDim":
      case "vocabSize":
        return (
          <ComfyIntListField
            key={key}
            label={
              memKind === "b"
                ? "vocab size"
                : String.raw`input dim $d_{\mathrm in}$`
            }
            values={intChoices(memKind === "b" ? (full.vocabSize ?? full.inputDim) : full.inputDim, 10)}
            min={memKind === "b" ? 2 : 1}
            onCommit={(vals) => {
              const packed = packIntList(vals);
              if (memKind === "b") {
                update({ vocabSize: packed, inputDim: packed, outputDim: packed });
                return;
              }
              update({ inputDim: packed });
            }}
            ariaLabel={memKind === "b" ? "Vocabulary size" : "Input dimension"}
          />
        );
      case "outputDim":
        if (memKind === "b") return null;
        return (
          <ComfyIntListField
            key={key}
            label={String.raw`output dim $d_{\mathrm out}$`}
            values={intChoices(full.outputDim, 1)}
            min={1}
            onCommit={(vals) => update({ outputDim: packIntList(vals) })}
            ariaLabel="Output dimension"
          />
        );
      case "inputDistribution":
        if (memKind === "b") return null;
        return (
          <DiscreteMultiSelect
            key={key}
            label="input distribution"
            options={INPUT_DISTRIBUTION_OPTIONS}
            value={full.inputDistribution}
            onCommit={(inputDistribution) => update({ inputDistribution })}
            ariaLabel="Input distribution"
          />
        );
      case "outputDistribution":
        return (
          <DiscreteMultiSelect
            key={key}
            label="output distribution"
            options={memKind !== "none" ? MEMORIZATION_OUTPUT_DISTRIBUTION_OPTIONS : OUTPUT_DISTRIBUTION_OPTIONS}
            value={full.outputDistribution}
            onCommit={(outputDistribution) => update({ outputDistribution })}
            ariaLabel="Output distribution"
          />
        );
      case "alpha":
        if (memKind === "none") return null;
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`decay rate $\alpha$`}
            values={floatChoices(full.alpha, 1.0)}
            min={0}
            positiveOnly
            title="Shape for power-law / exponential class priors"
            onCommit={(vals) => update({ alpha: packFloatList(vals) })}
            ariaLabel="Memorization class-prior decay rate alpha"
          />
        );
      case "trainSize":
        return (
          <ComfyIntListField
            key={key}
            label="train size"
            values={intChoices(full.trainSize, 800)}
            min={1}
            onCommit={(vals) => update({ trainSize: packIntList(vals) })}
            ariaLabel="Training set size"
          />
        );
      case "testSize":
        return (
          <ComfyIntListField
            key={key}
            label="test size"
            values={intChoices(full.testSize, 0)}
            min={0}
            onCommit={(vals) => update({ testSize: packIntList(vals) })}
            ariaLabel="Test set size"
          />
        );
      case "noiseLevel":
        return (
          <ComfyFloatListField
            key={key}
            label="noise level (σ)"
            values={floatChoices(full.noiseLevel, 0.25)}
            min={0}
            positiveOnly={false}
            title="σ in y = W x + σ ε — comma-separated for multiple runs"
            onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
            ariaLabel="Noise level sigma"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="seed"
            values={intChoices(full.seed, 0)}
            onCommit={(vals) => update({ seed: packIntList(vals) })}
            ariaLabel="Random seed"
          />
        );
      default: {
        const ex = d.extras;
        if (!ex || !(key in ex)) return null;
        const v = ex[key as keyof typeof ex];
        if (typeof v === "number") {
          if (Number.isInteger(v)) {
            return (
              <ComfyIntField
                key={key}
                label={key}
                value={v}
                min={-10_000_000}
                onCommit={(n) => update({ extras: { ...ex, [key]: n } })}
                ariaLabel={key}
              />
            );
          }
          return (
            <ComfyFloatListField
              key={key}
              label={key}
              values={floatChoices(v, 0)}
              positiveOnly={false}
              title={`${key} (from spec)`}
              onCommit={(vals) =>
                update({
                  extras: { ...ex, [key]: packFloatList(vals) } as Record<
                    string,
                    string | number | boolean
                  >,
                })
              }
              ariaLabel={key}
            />
          );
        }
        if (typeof v === "boolean") {
          return (
            <div key={key} className="cr-comfy-widget">
              <span className="cr-comfy-widget__label">{key}</span>
              <label className="cr-comfy-widget__control">
                <input
                  type="checkbox"
                  checked={v}
                  onChange={(e) => update({ extras: { ...ex, [key]: e.target.checked } })}
                />
              </label>
            </div>
          );
        }
        if (typeof v === "string") {
          return (
            <div key={key} className="cr-comfy-widget">
              <span className="cr-comfy-widget__label">{key}</span>
              <input
                className="cr-input cr-comfy-widget__control"
                value={v}
                onChange={(e) => update({ extras: { ...ex, [key]: e.target.value } })}
                aria-label={key}
              />
            </div>
          );
        }
        return null;
      }
    }
  };

  const datasetInfoKind: DatasetNodeInfoKind =
    type === "memorization_a_dataset"
      ? "memorization_a_dataset"
      : type === "memorization_b_dataset"
        ? "memorization_b_dataset"
        : type === "random_noise_dataset"
          ? "random_noise_dataset"
          : "linear_dataset";

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType={datasetInfoKind}
        nodeId={id}
        graphNodeType={graphType}
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, defaultInstanceTitleBase(graphType))}
        {d.specCodeName &&
        !(
          graphType === "linear_dataset" &&
          (d.specCodeName === MEMORIZATION_A_DATASET_SPEC_NAME || d.specCodeName === MEMORIZATION_B_DATASET_SPEC_NAME)
        ) ? (
          <span className="cr-node__header-sub">{d.specCodeName}</span>
        ) : null}
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

        {order.map((key) => renderField(key))}

      </div>
    </div>
  );
}
