import { useCallback, useMemo } from "react";
import {
  DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER,
  DEFAULT_LOCAL_MIXING_LAYER_SPEC_NAME,
  generateLocalMixingLayerSpecCode,
} from "../../graph/specCode/localMixingLayerSpecCode";
import { useReactFlow, type NodeProps } from "@xyflow/react";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { ComfyIntListField } from "./comfyMultiFields";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { defaultLocalMixingLayerData, type LocalMixingLayerNodeData } from "./localMixingLayerDefaults";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: LocalMixingLayerNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: LocalMixingLayerNodeData,
  patch: Partial<LocalMixingLayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: LocalMixingLayerNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER];
}

export function LocalMixingLayerNode({ id, data, selected }: NodeProps) {
  const defs = defaultLocalMixingLayerData();
  const d = { ...defs, ...(data as Partial<LocalMixingLayerNodeData>) } as LocalMixingLayerNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "atomic_layer"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = (d.specCodeName ?? DEFAULT_LOCAL_MIXING_LAYER_SPEC_NAME).trim() || DEFAULT_LOCAL_MIXING_LAYER_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateLocalMixingLayerSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<LocalMixingLayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = { ...defs, ...d };

  return (
    <div
      className={`cr-node cr-node--linear-layer${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "Local mixing layer")}
            infoText={`Residual causal depthwise 1D mixing on the sequence axis: y = x + DepthwiseConv1d(x). Use after embeddings for [batch, T, C] activations; rank-2 [batch, C] is treated as T=1. Same channel width in and out (modelDim must match upstream last dim).

**References:** [Physics of LLMs — Part 4.1 (Canon / local mixing)](https://arxiv.org/abs/2512.17351), [series hub](https://physics.allen-zhu.com/).`}
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Local mixing layer")}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body cr-node__body--compact">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        ) : (
          <>
            <ModelInitializationTargetRow />
            <AtomicLayerIoStrip />
          </>
        )}
        <ComfyIntListField
          label="model dim (channels C)"
          values={intChoices(full.modelDim, 64)}
          min={1}
          title="Last dimension of activations; must match upstream output width"
          onCommit={(vals) => update({ modelDim: packIntList(vals) })}
          ariaLabel="Model dimension"
        />
        <ComfyIntListField
          label="kernel size"
          values={intChoices(full.kernelSize, 5)}
          min={1}
          title="Causal depthwise kernel (server clamps to odd ≥ 3)"
          onCommit={(vals) => update({ kernelSize: packIntList(vals) })}
          ariaLabel="Kernel size"
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(full.seed ?? 0, 0)}
          min={0}
          title="PyTorch seed when this block is constructed in the trainer"
          onCommit={(vals) => update({ seed: packIntList(vals) })}
          ariaLabel="Init seed"
        />
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
