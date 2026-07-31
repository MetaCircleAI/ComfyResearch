import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { LinearLayerNodeData } from "./linearLayerDefaults";
import { defaultLinearLayerData } from "./linearLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_LINEAR_LAYER_PARAM_ORDER,
  DEFAULT_LINEAR_LAYER_SPEC_NAME,
  generateLinearLayerSpecCode,
} from "../../graph/specCode/linearLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";
import { applyMlpOwnerPatchForLinearExpansion } from "../../graph/mlpLowLevelExpansion";

function replaceNodeData(
  id: string,
  data: LinearLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => {
    let next = nodes.map((n) => (n.id === id ? { ...n, data } : n));
    next = applyMlpOwnerPatchForLinearExpansion(next, id, data);
    return next;
  });
}

function patchData(
  id: string,
  prev: LinearLayerNodeData,
  patch: Partial<LinearLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  const merged: LinearLayerNodeData = { ...prev, ...patch };
  setNodes((nodes) => {
    let next = nodes.map((n) => (n.id === id ? { ...n, data: merged } : n));
    next = applyMlpOwnerPatchForLinearExpansion(next, id, merged);
    return next;
  });
}

function effectiveParamOrder(d: LinearLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_LINEAR_LAYER_PARAM_ORDER];
}

export function LinearLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultLinearLayerData();
  const d = { ...defs, ...(data as Partial<LinearLayerNodeData>) } as LinearLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_LINEAR_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateLinearLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<LinearLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "inFeatures":
        return (
          <ComfyIntListField
            key={key}
            label="in features"
            values={intChoices(full.inFeatures, 10)}
            min={1}
            onCommit={(vals) => update({ inFeatures: packIntList(vals) })}
            ariaLabel="Input features"
          />
        );
      case "outFeatures":
        return (
          <ComfyIntListField
            key={key}
            label="out features"
            values={intChoices(full.outFeatures, 10)}
            min={1}
            onCommit={(vals) => update({ outFeatures: packIntList(vals) })}
            ariaLabel="Output features"
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
            ariaLabel="Linear bias flag"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch init seed for this layer (trainer applies its own ordering for full chains)"
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
      className={`cr-node cr-node--linear-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Linear layer")}
            infoText="Chain right tensor -> next layer's left tensor. Trainer model accepts the rightmost layer's tensor output; first and last submodules must be linear layers for MSE training."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Linear layer")}
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
