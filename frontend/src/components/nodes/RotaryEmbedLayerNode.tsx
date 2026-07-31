import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { RotaryEmbedLayerNodeData } from "./rotaryEmbedLayerDefaults";
import { defaultRotaryEmbedLayerData } from "./rotaryEmbedLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_ROTARY_EMBED_LAYER_PARAM_ORDER,
  DEFAULT_ROTARY_EMBED_LAYER_SPEC_NAME,
  generateRotaryEmbedLayerSpecCode,
} from "../../graph/specCode/rotaryEmbedLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: RotaryEmbedLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: RotaryEmbedLayerNodeData,
  patch: Partial<RotaryEmbedLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: RotaryEmbedLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_ROTARY_EMBED_LAYER_PARAM_ORDER];
}

export function RotaryEmbedLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultRotaryEmbedLayerData();
  const d = { ...defs, ...(data as Partial<RotaryEmbedLayerNodeData>) } as RotaryEmbedLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_ROTARY_EMBED_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateRotaryEmbedLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<RotaryEmbedLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  const renderField = (key: string) => {
    switch (key) {
      case "rotaryDim":
        return (
          <ComfyIntListField
            key={key}
            label="rotary dim (even)"
            values={intChoices(full.rotaryDim, 64)}
            min={2}
            title="Last dimension of activations; must be even (pairs of features are rotated)."
            onCommit={(vals) => update({ rotaryDim: packIntList(vals) })}
            ariaLabel="Rotary dimension"
          />
        );
      case "thetaBase":
        return (
          <ComfyFloatListField
            key={key}
            label="theta base"
            values={floatChoices(full.thetaBase, 10000)}
            positiveOnly
            title="Inverse-frequency base for RoPE (e.g. 10000 in LLaMA-style models)."
            onCommit={(vals) => update({ thetaBase: packFloatList(vals) })}
            ariaLabel="Theta base"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="Reserved for future per-layer init; trainer uses the chain tip seed when building."
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
      className={`cr-node cr-node--rotary-embed-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Rotary positional embedding")}
            infoText="Applies RoPE to the last dimension along positions on axis -2 (rank-2 inputs use a single position). Uses RotaryEmbedding in training."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Rotary positional embedding")}
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
