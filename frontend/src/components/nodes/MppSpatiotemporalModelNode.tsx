import { useCallback, useMemo } from "react";
import { useReactFlow, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultMppSpatiotemporalModelData,
  type MppSpatiotemporalModelNodeData,
} from "./mppSpatiotemporalModelDefaults";
import {
  DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER,
  DEFAULT_MPP_SPATIOTEMPORAL_SPEC_NAME,
  generateMppSpatiotemporalModelSpecCode,
} from "../../graph/specCode/mppSpatiotemporalModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

function replaceNodeData(
  id: string,
  data: MppSpatiotemporalModelNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: MppSpatiotemporalModelNodeData,
  patch: Partial<MppSpatiotemporalModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: MppSpatiotemporalModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER];
}

export function MppSpatiotemporalModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultMppSpatiotemporalModelData();
  const d = { ...defs, ...(data as Partial<MppSpatiotemporalModelNodeData>) } as MppSpatiotemporalModelNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_MPP_SPATIOTEMPORAL_SPEC_NAME;
  const generatedCode = useMemo(() => generateMppSpatiotemporalModelSpecCode(d, order, specName), [d, order, specName]);

  const update = useCallback(
    (patch: Partial<MppSpatiotemporalModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextFrames":
        return (
          <ComfyIntListField
            key={key}
            label={String.raw`frames $T$`}
            values={intChoices(full.contextFrames, 4)}
            min={1}
            title="Must match the PDE dataset contextFrames."
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
            label={String.raw`grid $H$`}
            values={intChoices(full.gridSize, 16)}
            min={4}
            title="Square grid; must match dataset and be divisible by patchSize."
            onCommit={(vals) => update({ gridSize: packIntList(vals) })}
            ariaLabel="Grid size"
          />
        );
      case "inputDim":
        return (
          <ComfyIntListField
            key={key}
            label="input dim (flat)"
            values={intChoices(full.inputDim, 1024)}
            min={1}
            title="Flattened size T·C·H·W; must match trainer tensors."
            onCommit={(vals) => update({ inputDim: packIntList(vals) })}
            ariaLabel="Input dimension"
          />
        );
      case "outputDim":
        return (
          <ComfyIntListField
            key={key}
            label="output dim (flat)"
            values={intChoices(full.outputDim, 1024)}
            min={1}
            onCommit={(vals) => update({ outputDim: packIntList(vals) })}
            ariaLabel="Output dimension"
          />
        );
      case "patchSize":
        return (
          <ComfyIntListField
            key={key}
            label="patch size"
            values={intChoices(full.patchSize, 4)}
            min={1}
            title="ViT patch edge length (must divide gridSize)."
            onCommit={(vals) => update({ patchSize: packIntList(vals) })}
            ariaLabel="Patch size"
          />
        );
      case "embedDim":
        return (
          <ComfyIntListField
            key={key}
            label="embed dim"
            values={intChoices(full.embedDim, 128)}
            min={8}
            onCommit={(vals) => update({ embedDim: packIntList(vals) })}
            ariaLabel="Embedding dimension"
          />
        );
      case "depth":
        return (
          <ComfyIntListField
            key={key}
            label="Transformer depth"
            values={intChoices(full.depth, 4)}
            min={1}
            onCommit={(vals) => update({ depth: packIntList(vals) })}
            ariaLabel="Encoder depth"
          />
        );
      case "numHeads":
        return (
          <ComfyIntListField
            key={key}
            label="num heads"
            values={intChoices(full.numHeads, 4)}
            min={1}
            title="embedDim must be divisible by numHeads."
            onCommit={(vals) => update({ numHeads: packIntList(vals) })}
            ariaLabel="Attention heads"
          />
        );
      case "ffRatio":
        return (
          <ComfyFloatListField
            key={key}
            label="FF multiplier"
            values={floatChoices(full.ffRatio, 4)}
            min={0.25}
            title="Feedforward hidden ≈ ffRatio × embedDim."
            onCommit={(vals) => update({ ffRatio: packFloatList(vals) })}
            ariaLabel="FF ratio"
          />
        );
      case "dropout":
        return (
          <ComfyFloatListField
            key={key}
            label="dropout"
            values={floatChoices(full.dropout, 0)}
            min={0}
            max={1}
            onCommit={(vals) => update({ dropout: packFloatList(vals) })}
            ariaLabel="Dropout"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch seed for weight initialization."
            onCommit={(vals) => update({ seed: packIntList(vals) })}
            ariaLabel="Init seed"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--mpp-spatiotemporal-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, "MPP-style spatiotemporal ViT")}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "MPP-style spatiotemporal ViT")}
              infoText={
                "Multiple Physics Pretraining–style field model: per-sample norm over time×space, Conv2d patch embed per frame, TransformerEncoder over all space-time tokens, unpatch + denorm. See arXiv:2310.02994 and PolymathicAI/multiple_physics_pretraining (AViT-style pipeline)."
              }
            />
          </div>
        </div>
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        {order.map((key) => renderField(key))}
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
