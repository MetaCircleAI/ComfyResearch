import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { UnembeddingLayerNodeData } from "./unembeddingLayerDefaults";
import { defaultUnembeddingLayerData } from "./unembeddingLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_UNEMBEDDING_LAYER_PARAM_ORDER,
  DEFAULT_UNEMBEDDING_LAYER_SPEC_NAME,
  generateUnembeddingLayerSpecCode,
} from "../../graph/specCode/unembeddingLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: UnembeddingLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: UnembeddingLayerNodeData,
  patch: Partial<UnembeddingLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: UnembeddingLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_UNEMBEDDING_LAYER_PARAM_ORDER];
}

export function UnembeddingLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultUnembeddingLayerData();
  const d = { ...defs, ...(data as Partial<UnembeddingLayerNodeData>) } as UnembeddingLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_UNEMBEDDING_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateUnembeddingLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<UnembeddingLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "inFeatures":
        return (
          <ComfyIntListField
            key={key}
            label="in features (d_model)"
            values={intChoices(full.inFeatures, 64)}
            min={1}
            onCommit={(vals) => update({ inFeatures: packIntList(vals) })}
            ariaLabel="Input feature dim"
          />
        );
      case "outFeatures":
        return (
          <ComfyIntListField
            key={key}
            label="out features (vocab / logits)"
            values={intChoices(full.outFeatures, 4096)}
            min={1}
            onCommit={(vals) => update({ outFeatures: packIntList(vals) })}
            ariaLabel="Output feature dim"
          />
        );
      case "bias":
        return (
          <ComfyIntListField
            key={key}
            label="bias (1=yes, 0=no)"
            values={intChoices(full.bias, 1)}
            min={0}
            max={1}
            onCommit={(vals) => update({ bias: packIntList(vals) })}
            ariaLabel="Linear bias"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch init seed for this layer"
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
      className={`cr-node cr-node--unembedding-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Unembedding layer")}
            infoText="Transformer-style output projection: torch.nn.Linear(d_model, vocab) on the right tensor. Wire targets (e.g. MSE) to the final width you configure here."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Unembedding layer")}
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
