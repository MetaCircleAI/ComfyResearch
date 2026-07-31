import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  OUTPUT_DISTRIBUTION_OPTIONS,
  type DatasetSamplingMode,
} from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultKepler2dDatasetData,
  type Kepler2dDatasetNodeData,
} from "./kepler2dDatasetDefaults";
import {
  DEFAULT_KEPLER_2D_DATASET_PARAM_ORDER,
  DEFAULT_KEPLER_2D_DATASET_SPEC_NAME,
  generateKepler2dDatasetSpecCode,
} from "../../graph/specCode/kepler2dDatasetSpecCode";

function replaceNodeData(
  id: string,
  data: Kepler2dDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function Kepler2dDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultKepler2dDatasetData();
  const d = { ...defs, ...(data as Partial<Kepler2dDatasetNodeData>) } as Kepler2dDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = d.paramOrder?.length ? d.paramOrder : DEFAULT_KEPLER_2D_DATASET_PARAM_ORDER;
  const specName = d.specCodeName ?? DEFAULT_KEPLER_2D_DATASET_SPEC_NAME;
  const generatedCode = useMemo(() => generateKepler2dDatasetSpecCode(d, order, specName), [d, order, specName]);
  const update = useCallback(
    (patch: Partial<Kepler2dDatasetNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
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
            values={intChoices(full.contextLength, 8)}
            min={1}
            title="Sequence length T with x=[N,T,2] and y=next-step positions [N,T,2]."
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
            ariaLabel="Context length"
          />
        );
      case "trainSize":
        return (
          <ComfyIntListField
            key={key}
            label="train size"
            values={intChoices(full.trainSize, 1600)}
            min={1}
            onCommit={(vals) => update({ trainSize: packIntList(vals) })}
            ariaLabel="Train size"
          />
        );
      case "testSize":
        return (
          <ComfyIntListField
            key={key}
            label="test size"
            values={intChoices(full.testSize, 400)}
            min={0}
            onCommit={(vals) => update({ testSize: packIntList(vals) })}
            ariaLabel="Test size"
          />
        );
      case "semiMajorAxisMin":
        return (
          <ComfyFloatListField
            key={key}
            label="semi-major axis min"
            values={floatChoices(full.semiMajorAxisMin, 0.7)}
            min={0}
            positiveOnly={false}
            onCommit={(vals) => update({ semiMajorAxisMin: packFloatList(vals) })}
            ariaLabel="Semi-major axis minimum"
          />
        );
      case "semiMajorAxisMax":
        return (
          <ComfyFloatListField
            key={key}
            label="semi-major axis max"
            values={floatChoices(full.semiMajorAxisMax, 1.3)}
            min={0}
            positiveOnly={false}
            onCommit={(vals) => update({ semiMajorAxisMax: packFloatList(vals) })}
            ariaLabel="Semi-major axis maximum"
          />
        );
      case "eccentricityMin":
        return (
          <ComfyFloatListField
            key={key}
            label="eccentricity min"
            values={floatChoices(full.eccentricityMin, 0.0)}
            min={0}
            positiveOnly={false}
            onCommit={(vals) => update({ eccentricityMin: packFloatList(vals) })}
            ariaLabel="Eccentricity minimum"
          />
        );
      case "eccentricityMax":
        return (
          <ComfyFloatListField
            key={key}
            label="eccentricity max"
            values={floatChoices(full.eccentricityMax, 0.55)}
            min={0}
            positiveOnly={false}
            onCommit={(vals) => update({ eccentricityMax: packFloatList(vals) })}
            ariaLabel="Eccentricity maximum"
          />
        );
      case "meanMotion":
        return (
          <ComfyFloatListField
            key={key}
            label="mean motion $n$"
            values={floatChoices(full.meanMotion, 0.4)}
            min={0}
            positiveOnly={false}
            title="Angular speed in mean-anomaly updates M_t = M_0 + n*t."
            onCommit={(vals) => update({ meanMotion: packFloatList(vals) })}
            ariaLabel="Mean motion"
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
            ariaLabel="Output distribution"
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
            title="Only applied to the final target timestep when output distribution is additive_gaussian."
            onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
            ariaLabel="Noise level"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="NumPy seed for orbital parameter/phase sampling."
            onCommit={(vals) => update({ seed: packIntList(vals) })}
            ariaLabel="Initialization seed"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--kepler2d-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="kepler_2d_dataset"
        nodeId={id}
        graphNodeType="kepler_2d_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Kepler 2D dataset")}
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
          ariaLabel="Train data sampling mode"
          singleSelect
        />
        {order.map((key) => renderField(key))}
      </div>
    </div>
  );
}
