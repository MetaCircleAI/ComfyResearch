import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { AbsolutePosEmbedLayerNodeData } from "./absolutePosEmbedLayerDefaults";
import { defaultAbsolutePosEmbedLayerData } from "./absolutePosEmbedLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_ABSOLUTE_POS_EMBED_LAYER_PARAM_ORDER,
  DEFAULT_ABSOLUTE_POS_EMBED_LAYER_SPEC_NAME,
  generateAbsolutePosEmbedLayerSpecCode,
} from "../../graph/specCode/absolutePosEmbedLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: AbsolutePosEmbedLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: AbsolutePosEmbedLayerNodeData,
  patch: Partial<AbsolutePosEmbedLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: AbsolutePosEmbedLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_ABSOLUTE_POS_EMBED_LAYER_PARAM_ORDER];
}

export function AbsolutePosEmbedLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultAbsolutePosEmbedLayerData();
  const d = { ...defs, ...(data as Partial<AbsolutePosEmbedLayerNodeData>) } as AbsolutePosEmbedLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_ABSOLUTE_POS_EMBED_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateAbsolutePosEmbedLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<AbsolutePosEmbedLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "maxSeqLen":
        return (
          <ComfyIntListField
            key={key}
            label="max sequence length"
            values={intChoices(full.maxSeqLen, 512)}
            min={1}
            title="Table rows; sequence axis length must not exceed this at runtime."
            onCommit={(vals) => update({ maxSeqLen: packIntList(vals) })}
            ariaLabel="Max sequence length"
          />
        );
      case "embeddingDim":
        return (
          <ComfyIntListField
            key={key}
            label="embedding dim"
            values={intChoices(full.embeddingDim, 64)}
            min={1}
            title="Must match the last dimension of incoming activations."
            onCommit={(vals) => update({ embeddingDim: packIntList(vals) })}
            ariaLabel="Embedding dimension"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch init seed for the trainer build (global seed before the chain is constructed)."
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
      className={`cr-node cr-node--absolute-pos-embed-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Absolute positional embedding")}
            infoText="Adds a learnable table [max_seq_len, dim] along the sequence axis (dimension -2 when rank >= 3; rank-2 tensors use position 0 only). Uses AbsolutePositionalEmbedding in training."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Absolute positional embedding")}
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

        {d.extras && Object.keys(d.extras).length > 0 ? (
          <p className="cr-node__hint cr-node__hint--extras">
            Extra params from spec (not used by training): {JSON.stringify(d.extras)}
          </p>
        ) : null}

        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
