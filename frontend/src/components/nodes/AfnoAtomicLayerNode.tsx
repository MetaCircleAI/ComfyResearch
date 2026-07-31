import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { defaultAfnoAtomicLayerData, type AfnoAtomicLayerNodeData } from "./afnoAtomicLayerDefaults";
import {
  DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER,
  defaultAfnoAtomicSpecName,
  generateAfnoAtomicLayerSpecCode,
  type AfnoAtomicLayerKind,
} from "../../graph/specCode/afnoAtomicLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

const KIND_TITLE: Record<AfnoAtomicLayerKind, string> = {
  afno_patch_embed_layer: "AFNO patch embed layer",
  afno_spectral_mixer_layer: "AFNO spectral mixer layer",
  afno_encoder_block_layer: "AFNO encoder block layer",
  afno_patch_decode_layer: "AFNO patch decode layer",
};

function nodeKindFromType(nodeType: string | undefined): AfnoAtomicLayerKind {
  const t = String(nodeType ?? "") as AfnoAtomicLayerKind;
  if (
    t === "afno_patch_embed_layer" ||
    t === "afno_spectral_mixer_layer" ||
    t === "afno_encoder_block_layer" ||
    t === "afno_patch_decode_layer"
  ) {
    return t;
  }
  return "afno_patch_embed_layer";
}

function effectiveParamOrder(d: AfnoAtomicLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER];
}

function replaceNodeData(
  id: string,
  data: AfnoAtomicLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: AfnoAtomicLayerNodeData,
  patch: Partial<AfnoAtomicLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

export function AfnoAtomicLayerNode({ id, type, data, selected }: NodeProps) {
  const kind = nodeKindFromType(type);
  const defs = defaultAfnoAtomicLayerData();
  const d = { ...defs, ...(data as Partial<AfnoAtomicLayerNodeData>) } as AfnoAtomicLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();
  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? defaultAfnoAtomicSpecName(kind);
  const generatedCode = useMemo(() => generateAfnoAtomicLayerSpecCode(kind, d, order, specName), [kind, d, order, specName]);
  const update = useCallback(
    (patch: Partial<AfnoAtomicLayerNodeData>) => patchData(id, d, patch, setNodes),
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
            onCommit={(vals) => update({ inputDim: packIntList(vals) })}
            ariaLabel="Input dim"
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
            ariaLabel="Output dim"
          />
        );
      case "patchSize":
        return (
          <ComfyIntListField
            key={key}
            label="patch size"
            values={intChoices(full.patchSize, 4)}
            min={1}
            onCommit={(vals) => update({ patchSize: packIntList(vals) })}
            ariaLabel="Patch size"
          />
        );
      case "embedDim":
        return (
          <ComfyIntListField
            key={key}
            label="embed dim"
            values={intChoices(full.embedDim, 64)}
            min={8}
            onCommit={(vals) => update({ embedDim: packIntList(vals) })}
            ariaLabel="Embed dim"
          />
        );
      case "numHeads":
        return (
          <ComfyIntListField
            key={key}
            label="num heads"
            values={intChoices(full.numHeads, 4)}
            min={1}
            onCommit={(vals) => update({ numHeads: packIntList(vals) })}
            ariaLabel="Num heads"
          />
        );
      case "ffRatio":
        return (
          <ComfyFloatListField
            key={key}
            label="FF multiplier"
            values={floatChoices(full.ffRatio, 2)}
            min={0.25}
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
      case "numSpectralBlocks":
        return (
          <ComfyIntListField
            key={key}
            label="spectral blocks"
            values={intChoices(full.numSpectralBlocks, 1)}
            min={1}
            onCommit={(vals) => update({ numSpectralBlocks: packIntList(vals) })}
            ariaLabel="Spectral blocks"
          />
        );
      case "maxFrequencyModes":
        return (
          <ComfyIntListField
            key={key}
            label="max freq modes"
            values={intChoices(full.maxFrequencyModes, 4)}
            min={1}
            onCommit={(vals) => update({ maxFrequencyModes: packIntList(vals) })}
            ariaLabel="Max frequency modes"
          />
        );
      case "spectralShrinkFactor":
        return (
          <ComfyFloatListField
            key={key}
            label="spectral shrink"
            values={floatChoices(full.spectralShrinkFactor, 1)}
            min={0.01}
            onCommit={(vals) => update({ spectralShrinkFactor: packFloatList(vals) })}
            ariaLabel="Spectral shrink factor"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
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
      className={`cr-node cr-node--afno-atomic-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, KIND_TITLE[kind])}
            infoText="AFNO atomic layer for manual composition. Use input-output mode to chain tensors, or model mode to connect layer tips directly into the trainer model socket."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, KIND_TITLE[kind])}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="tensor" sourceLabel="model" />
        ) : (
          <>
            <ModelInitializationTargetRow />
            <AtomicLayerIoStrip />
          </>
        )}
        {order.map((key) => renderField(key))}
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}

