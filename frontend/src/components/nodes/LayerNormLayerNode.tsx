import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { LayerNormLayerNodeData } from "./layerNormLayerDefaults";
import { defaultLayerNormLayerData } from "./layerNormLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_LAYER_NORM_LAYER_PARAM_ORDER,
  DEFAULT_LAYER_NORM_LAYER_SPEC_NAME,
  generateLayerNormLayerSpecCode,
} from "../../graph/specCode/layerNormLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: LayerNormLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: LayerNormLayerNodeData,
  patch: Partial<LayerNormLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: LayerNormLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_LAYER_NORM_LAYER_PARAM_ORDER];
}

export function LayerNormLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultLayerNormLayerData();
  const d = { ...defs, ...(data as Partial<LayerNormLayerNodeData>) } as LayerNormLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_LAYER_NORM_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateLayerNormLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<LayerNormLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "normalizedShape":
        return (
          <ComfyIntListField
            key={key}
            label="normalized shape"
            values={intChoices(full.normalizedShape, 64)}
            min={1}
            title="Last-dimension size LayerNorm acts on (see torch.nn.LayerNorm)"
            onCommit={(vals) => update({ normalizedShape: packIntList(vals) })}
            ariaLabel="Normalized shape"
          />
        );
      case "eps":
        return (
          <ComfyFloatListField
            key={key}
            label="eps"
            values={floatChoices(full.eps, 1e-5)}
            positiveOnly={false}
            onCommit={(vals) => update({ eps: packFloatList(vals) })}
            ariaLabel="LayerNorm epsilon"
          />
        );
      case "elementwiseAffine":
        return (
          <ComfyIntListField
            key={key}
            label="elementwise affine (1=yes, 0=no)"
            values={intChoices(full.elementwiseAffine, 1)}
            min={0}
            max={1}
            onCommit={(vals) => update({ elementwiseAffine: packIntList(vals) })}
            ariaLabel="Elementwise affine"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--layer-norm-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "LayerNorm layer")}
            infoText="Normalizes over the last normalized_shape entries of the input tensor."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "LayerNorm layer")}
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
