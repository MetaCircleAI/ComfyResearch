import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  INPUT_DISTRIBUTION_OPTIONS,
  OUTPUT_DISTRIBUTION_OPTIONS,
  type DatasetSamplingMode,
} from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultUniformLinearMotionDatasetData,
  type UniformLinearMotionDatasetNodeData,
} from "./uniformLinearMotionDatasetDefaults";
import {
  DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_PARAM_ORDER,
  DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_SPEC_NAME,
  generateUniformLinearMotionDatasetSpecCode,
} from "../../graph/specCode/uniformLinearMotionDatasetSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

function replaceNodeData(
  id: string,
  data: UniformLinearMotionDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: UniformLinearMotionDatasetNodeData,
  patch: Partial<UniformLinearMotionDatasetNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function dedupeParamOrder(order: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const k of order) {
    const nk = k === "x1Distribution" ? "velocityDistribution" : k;
    if (seen.has(nk)) continue;
    seen.add(nk);
    out.push(nk);
  }
  return out;
}

function effectiveParamOrder(d: UniformLinearMotionDatasetNodeData): string[] {
  if (!d.paramOrder?.length) return [...DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_PARAM_ORDER];
  let order = dedupeParamOrder(d.paramOrder);
  if (order.includes("velocityDistribution") && !order.includes("velocityScale")) {
    const i = order.indexOf("velocityDistribution");
    order = [...order.slice(0, i + 1), "velocityScale", ...order.slice(i + 1)];
  }
  return order;
}

type LegacyUniformDatasetData = Partial<UniformLinearMotionDatasetNodeData>;

export function UniformLinearMotionDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultUniformLinearMotionDatasetData();
  const raw = (data ?? {}) as LegacyUniformDatasetData;
  const d = {
    ...defs,
    ...raw,
    velocityDistribution: raw.velocityDistribution ?? raw.x1Distribution ?? defs.velocityDistribution,
    velocityScale: raw.velocityScale ?? defs.velocityScale,
  } as UniformLinearMotionDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateUniformLinearMotionDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<UniformLinearMotionDatasetNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextLength":
        return (
          <ComfyIntListField
            key={key}
            label="context length $t$"
            values={intChoices(full.contextLength, 2)}
            min={1}
            max={512}
            title="Sequence length T: input tensor is x_0…x_{T-1}, target is x_1…x_T with x_i = x0 + velocity·i."
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
            ariaLabel="Context length"
          />
        );
      case "positionDim":
        return (
          <ComfyIntListField
            key={key}
            label="position dim $d$"
            values={intChoices(full.positionDim, 1)}
            min={1}
            onCommit={(vals) => update({ positionDim: packIntList(vals) })}
            ariaLabel="Position dimension"
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
            values={intChoices(full.testSize, 200)}
            min={0}
            onCommit={(vals) => update({ testSize: packIntList(vals) })}
            ariaLabel="Test set size"
          />
        );
      case "positionDistribution":
        return (
          <DiscreteMultiSelect
            key={key}
            label="x₀ distribution"
            options={INPUT_DISTRIBUTION_OPTIONS}
            value={full.positionDistribution}
            onCommit={(positionDistribution) => update({ positionDistribution })}
            ariaLabel="Initial position distribution"
          />
        );
      case "velocityDistribution":
      case "x1Distribution":
        return (
          <DiscreteMultiSelect
            key="velocityDistribution"
            label="velocity distribution"
            options={INPUT_DISTRIBUTION_OPTIONS}
            value={full.velocityDistribution}
            onCommit={(velocityDistribution) => update({ velocityDistribution })}
            ariaLabel="Velocity distribution"
          />
        );
      case "velocityScale":
        return (
          <ComfyFloatListField
            key={key}
            label="velocity scale"
            values={floatChoices(full.velocityScale, 1)}
            min={0}
            positiveOnly={false}
            title="Multiplies the drawn velocity (same random draw shape; use to separate global scale from distribution choice)."
            onCommit={(vals) => update({ velocityScale: packFloatList(vals) })}
            ariaLabel="Velocity scale"
          />
        );
      case "outputDistribution":
        return (
          <DiscreteMultiSelect
            key={key}
            label="output distribution"
            options={OUTPUT_DISTRIBUTION_OPTIONS}
            value={full.outputDistribution}
            onCommit={(outputDistribution) => update({ outputDistribution })}
            ariaLabel="Output noise mode"
          />
        );
      case "noiseLevel":
        return (
          <ComfyFloatListField
            key={key}
            label="noise level (σ)"
            values={floatChoices(full.noiseLevel, 0)}
            min={0}
            positiveOnly={false}
            onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
            ariaLabel="Output noise level"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="NumPy seed for sampling x0, velocity, and optional Gaussian noise on the last target row"
            onCommit={(vals) => update({ seed: packIntList(vals) })}
            ariaLabel="Random seed"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--uniform-linear-motion-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="uniform_linear_motion_dataset"
        nodeId={id}
        graphNodeType="uniform_linear_motion_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Uniform linear motion dataset")}
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
        {order.map((key) => renderField(key))}
      </div>
    </div>
  );
}
