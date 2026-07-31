import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { ActivationLayerNodeData } from "./activationLayerDefaults";
import { defaultActivationLayerData } from "./activationLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ComfyFloatField } from "./comfyNumberFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { MLP_ACTIVATION_OPTIONS, type MlpActivationId } from "./mlpModelDefaults";
import {
  DEFAULT_ACTIVATION_LAYER_PARAM_ORDER,
  DEFAULT_ACTIVATION_LAYER_SPEC_NAME,
  generateActivationLayerSpecCode,
} from "../../graph/specCode/activationLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";
import { applyMlpOwnerPatchForActivationExpansion } from "../../graph/mlpLowLevelExpansion";

function patchData(
  id: string,
  prev: ActivationLayerNodeData,
  patch: Partial<ActivationLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  const merged: ActivationLayerNodeData = { ...prev, ...patch };
  setNodes((nodes) => {
    let next = nodes.map((n) => (n.id === id ? { ...n, data: merged } : n));
    next = applyMlpOwnerPatchForActivationExpansion(next, id, merged);
    return next;
  });
}

function activationScalarId(activation: ActivationLayerNodeData["activation"]): MlpActivationId {
  const v = Array.isArray(activation) && activation.length ? activation[0] : activation;
  return (typeof v === "string" ? v : "relu") as MlpActivationId;
}

function effectiveParamOrder(d: ActivationLayerNodeData): string[] {
  const base = d.paramOrder?.length ? [...d.paramOrder] : [...DEFAULT_ACTIVATION_LAYER_PARAM_ORDER];
  if (activationScalarId(d.activation) === "leaky_relu" && !base.includes("leakyP")) {
    const ai = base.indexOf("activation");
    if (ai >= 0) base.splice(ai + 1, 0, "leakyP");
    else base.push("leakyP");
  }
  return base;
}

export function ActivationLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultActivationLayerData();
  const d = { ...defs, ...(data as Partial<ActivationLayerNodeData>) } as ActivationLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_ACTIVATION_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateActivationLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<ActivationLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    if (key === "activation") {
      return (
        <DiscreteMultiSelect<MlpActivationId>
          key={key}
          label="activation"
          options={MLP_ACTIVATION_OPTIONS}
          value={full.activation}
          onCommit={(activation) => update({ activation })}
          ariaLabel="Activation function"
        />
      );
    }
    if (key === "leakyP") {
      if (activationScalarId(full.activation) !== "leaky_relu") return null;
      const lp = typeof full.leakyP === "number" && Number.isFinite(full.leakyP) ? full.leakyP : 0;
      return (
        <ComfyFloatField
          key={key}
          label="leaky p"
          value={Math.max(-1, Math.min(1, lp))}
          min={-1}
          max={1}
          title="LeakyReLU negative_slope in [-1, 1]. 0 → ReLU, 1 → linear, -1 → absolute value on negative inputs."
          onCommit={(n) => update({ leakyP: Math.max(-1, Math.min(1, n)) })}
          ariaLabel="Leaky ReLU p parameter"
        />
      );
    }
    return null;
  };

  return (
    <div
      className={`cr-node cr-node--activation-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Activation layer")}
            infoText="Nonlinearity only - place between linear layers. This node is not the same as the analysis Activation capture node."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Activation layer")}
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
