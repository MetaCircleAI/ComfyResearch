import { useCallback, useMemo } from "react";
import { useReactFlow, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  type DatasetSamplingMode,
} from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultPdeFieldDatasetData,
  type PdeFieldDatasetKind,
  type PdeFieldDatasetNodeData,
} from "./pdeFieldDatasetDefaults";
import {
  DEFAULT_PDE_FIELD_DATASET_SPEC_NAME,
  defaultParamOrderForPdeKind,
  generatePdeFieldDatasetSpecCode,
} from "../../graph/specCode/pdeFieldDatasetSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

function replaceNodeData(
  id: string,
  data: PdeFieldDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: PdeFieldDatasetNodeData,
  patch: Partial<PdeFieldDatasetNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

const TITLE_BY_KIND: Record<PdeFieldDatasetKind, string> = {
  diffusion_pde_dataset: "Diffusion PDE dataset",
  reaction_diffusion_dataset: "Reaction–diffusion dataset",
  advection_dataset: "Advection dataset",
};

const HEADER_KIND_INFO: Record<PdeFieldDatasetKind, string> = {
  diffusion_pde_dataset: "diffusion_pde_dataset",
  reaction_diffusion_dataset: "reaction_diffusion_dataset",
  advection_dataset: "advection_dataset",
};

type Props = NodeProps & { graphNodeType: PdeFieldDatasetKind };

export function PdeFieldDatasetNode({ id, data, selected, graphNodeType }: Props) {
  const defs = defaultPdeFieldDatasetData(graphNodeType);
  const d = { ...defs, ...(data as Partial<PdeFieldDatasetNodeData>) } as PdeFieldDatasetNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => (d.paramOrder?.length ? d.paramOrder : defaultParamOrderForPdeKind(graphNodeType)), [d, graphNodeType]);
  const specName = d.specCodeName ?? DEFAULT_PDE_FIELD_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generatePdeFieldDatasetSpecCode(graphNodeType, d, order, specName),
    [graphNodeType, d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<PdeFieldDatasetNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextFrames":
        return (
          <ComfyIntListField
            key={key}
            label={String.raw`context frames $T$`}
            values={intChoices(full.contextFrames, 4)}
            min={1}
            title="Number of spatial fields along time in each flattened sample (input u_0…u_{T−1}, target u_1…u_T)."
            onCommit={(vals) => update({ contextFrames: packIntList(vals) })}
            ariaLabel="Context frames"
          />
        );
      case "channels":
        return (
          <ComfyIntListField
            key={key}
            label={String.raw`channels $C$`}
            values={intChoices(full.channels, 1)}
            min={1}
            onCommit={(vals) => update({ channels: packIntList(vals) })}
            ariaLabel="Channels"
          />
        );
      case "gridSize":
        return (
          <ComfyIntListField
            key={key}
            label={String.raw`grid size $H{=}W$`}
            values={intChoices(full.gridSize, 16)}
            min={4}
            title="Square periodic grid side length (patch models require gridSize divisible by patchSize)."
            onCommit={(vals) => update({ gridSize: packIntList(vals) })}
            ariaLabel="Grid size"
          />
        );
      case "trainSize":
        return (
          <ComfyIntListField
            key={key}
            label="train size"
            values={intChoices(full.trainSize, 512)}
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
            values={intChoices(full.testSize, 128)}
            min={0}
            onCommit={(vals) => update({ testSize: packIntList(vals) })}
            ariaLabel="Test size"
          />
        );
      case "warmupSteps":
        return (
          <ComfyIntListField
            key={key}
            label="warmup steps"
            values={intChoices(full.warmupSteps, 40)}
            min={0}
            title="Euler steps to discard before recording the context window (reduces transient artifacts)."
            onCommit={(vals) => update({ warmupSteps: packIntList(vals) })}
            ariaLabel="Warmup steps"
          />
        );
      case "dt":
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`time step $\Delta t$`}
            values={floatChoices(full.dt, 0.05)}
            min={1e-6}
            title="Explicit Euler step (keep small for stability)."
            onCommit={(vals) => update({ dt: packFloatList(vals) })}
            ariaLabel="Delta t"
          />
        );
      case "diffusionCoeff":
        if (graphNodeType === "advection_dataset") return null;
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`diffusion $D$`}
            values={floatChoices(full.diffusionCoeff, 0.2)}
            min={0}
            title={graphNodeType === "reaction_diffusion_dataset" ? "Fisher–KPP diffusion coefficient." : "Heat-equation diffusion coefficient."}
            onCommit={(vals) => update({ diffusionCoeff: packFloatList(vals) })}
            ariaLabel="Diffusion coefficient"
          />
        );
      case "reactionRate":
        if (graphNodeType !== "reaction_diffusion_dataset") return null;
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`reaction rate $r$`}
            values={floatChoices(full.reactionRate, 1)}
            min={0}
            title="Logistic reaction term r·u·(1−u) in Fisher–KPP."
            onCommit={(vals) => update({ reactionRate: packFloatList(vals) })}
            ariaLabel="Reaction rate"
          />
        );
      case "velocityX":
        if (graphNodeType !== "advection_dataset") return null;
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`velocity $v_x$`}
            values={floatChoices(full.velocityX, 0.5)}
            title="Constant advection velocity (central finite differences, periodic domain)."
            onCommit={(vals) => update({ velocityX: packFloatList(vals) })}
            ariaLabel="Velocity x"
          />
        );
      case "velocityY":
        if (graphNodeType !== "advection_dataset") return null;
        return (
          <ComfyFloatListField
            key={key}
            label={String.raw`velocity $v_y$`}
            values={floatChoices(full.velocityY, 0.2)}
            onCommit={(vals) => update({ velocityY: packFloatList(vals) })}
            ariaLabel="Velocity y"
          />
        );
      case "icScale":
        return (
          <ComfyFloatListField
            key={key}
            label="IC scale"
            values={floatChoices(full.icScale, 0.5)}
            min={0}
            title="Gaussian random initial field scale (per episode)."
            onCommit={(vals) => update({ icScale: packFloatList(vals) })}
            ariaLabel="Initial condition scale"
          />
        );
      case "initSeed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.initSeed, 0)}
            min={0}
            title="NumPy seed for synthetic field episodes."
            onCommit={(vals) => update({ initSeed: packIntList(vals) })}
            ariaLabel="Init seed"
          />
        );
      default:
        return null;
    }
  };

  const mod =
    graphNodeType === "diffusion_pde_dataset"
      ? "diffusion-pde-dataset"
      : graphNodeType === "reaction_diffusion_dataset"
        ? "reaction-diffusion-dataset"
        : "advection-dataset";

  return (
    <div className={`cr-node cr-node--${mod}${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}>
      <DatasetNodeHeaderWithInfo
        nodeType={HEADER_KIND_INFO[graphNodeType]}
        nodeId={id}
        graphNodeType={graphNodeType}
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, TITLE_BY_KIND[graphNodeType])}
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

export function DiffusionPdeDatasetNode(props: NodeProps) {
  return <PdeFieldDatasetNode {...props} graphNodeType="diffusion_pde_dataset" />;
}

export function ReactionDiffusionDatasetNode(props: NodeProps) {
  return <PdeFieldDatasetNode {...props} graphNodeType="reaction_diffusion_dataset" />;
}

export function AdvectionDatasetNode(props: NodeProps) {
  return <PdeFieldDatasetNode {...props} graphNodeType="advection_dataset" />;
}
