import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { EmbeddingLayerNodeData } from "./embeddingLayerDefaults";
import { defaultEmbeddingLayerData } from "./embeddingLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_EMBEDDING_LAYER_PARAM_ORDER,
  DEFAULT_EMBEDDING_LAYER_SPEC_NAME,
  generateEmbeddingLayerSpecCode,
} from "../../graph/specCode/embeddingLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: EmbeddingLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: EmbeddingLayerNodeData,
  patch: Partial<EmbeddingLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: EmbeddingLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_EMBEDDING_LAYER_PARAM_ORDER];
}

export function EmbeddingLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultEmbeddingLayerData();
  const d = { ...defs, ...(data as Partial<EmbeddingLayerNodeData>) } as EmbeddingLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_EMBEDDING_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateEmbeddingLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<EmbeddingLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "numEmbeddings":
        return (
          <ComfyIntListField
            key={key}
            label="num embeddings (vocab size)"
            values={intChoices(full.numEmbeddings, 4096)}
            min={1}
            onCommit={(vals) => update({ numEmbeddings: packIntList(vals) })}
            ariaLabel="Number of embedding rows"
          />
        );
      case "embeddingDim":
        return (
          <ComfyIntListField
            key={key}
            label="embedding dim"
            values={intChoices(full.embeddingDim, 64)}
            min={1}
            onCommit={(vals) => update({ embeddingDim: packIntList(vals) })}
            ariaLabel="Embedding vector dimension"
          />
        );
      case "numIndexColumns":
        return (
          <ComfyIntListField
            key={key}
            label="index tensor width"
            values={intChoices(full.numIndexColumns, 1)}
            min={1}
            title="Last dimension of the incoming index tensor (must match dataset input width for trainer)."
            onCommit={(vals) => update({ numIndexColumns: packIntList(vals) })}
            ariaLabel="Index tensor trailing dimension"
          />
        );
      case "paddingIdx":
        return (
          <ComfyIntListField
            key={key}
            label="padding idx (-1 = none)"
            values={intChoices(full.paddingIdx, -1)}
            min={-1}
            onCommit={(vals) => update({ paddingIdx: packIntList(vals) })}
            ariaLabel="Embedding padding index"
          />
        );
      case "scaleGradByFreq":
        return (
          <ComfyIntListField
            key={key}
            label="scale grad by freq (1=yes)"
            values={intChoices(full.scaleGradByFreq, 0)}
            min={0}
            max={1}
            onCommit={(vals) => update({ scaleGradByFreq: packIntList(vals) })}
            ariaLabel="scale_grad_by_freq"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch init seed for this module"
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
      className={`cr-node cr-node--embedding-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Embedding layer")}
            infoText="Left tensor carries index values (use long at runtime; float batches from linear datasets are cast). Lookup uses torch.nn.Embedding; output is float activations for the next layer."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Embedding layer")}
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
