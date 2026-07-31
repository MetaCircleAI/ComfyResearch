import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { RmsNormLayerNodeData } from "./rmsNormLayerDefaults";
import { defaultRmsNormLayerData } from "./rmsNormLayerDefaults";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_RMS_NORM_LAYER_PARAM_ORDER,
  DEFAULT_RMS_NORM_LAYER_SPEC_NAME,
  generateRmsNormLayerSpecCode,
} from "../../graph/specCode/rmsNormLayerSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: RmsNormLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: RmsNormLayerNodeData,
  patch: Partial<RmsNormLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: RmsNormLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_RMS_NORM_LAYER_PARAM_ORDER];
}

export function RmsNormLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultRmsNormLayerData();
  const d = { ...defs, ...(data as Partial<RmsNormLayerNodeData>) } as RmsNormLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_RMS_NORM_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateRmsNormLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<RmsNormLayerNodeData>) => patchData(id, d, patch, setNodes),
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
            title="Last dimension RMSNorm scales (LLaMA-style; no mean centering)."
            onCommit={(vals) => update({ normalizedShape: packIntList(vals) })}
            ariaLabel="RMSNorm normalized shape"
          />
        );
      case "eps":
        return (
          <ComfyFloatListField
            key={key}
            label="eps"
            values={floatChoices(full.eps, 1e-6)}
            positiveOnly={false}
            onCommit={(vals) => update({ eps: packFloatList(vals) })}
            ariaLabel="RMSNorm epsilon"
          />
        );
      case "elementwiseAffine":
        return (
          <ComfyIntListField
            key={key}
            label="scale weights (1=yes, 0=no)"
            values={intChoices(full.elementwiseAffine, 1)}
            min={0}
            max={1}
            onCommit={(vals) => update({ elementwiseAffine: packIntList(vals) })}
            ariaLabel="Optional RMSNorm affine scale"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--rms-norm-layer${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "RMSNorm layer")}
            infoText="Scales the last dimension by RMS (optional affine gain); defaults match minimal stabilization knobs elsewhere."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "RMSNorm layer")}
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
