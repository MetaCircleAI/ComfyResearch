import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { MLP_ACTIVATION_OPTIONS } from "./mlpModelDefaults";
import { defaultMoeMlpModelData, type MoeMlpModelNodeData } from "./moeMlpModelDefaults";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_MOE_MLP_PARAM_ORDER,
  DEFAULT_MOE_MLP_SPEC_NAME,
  generateMoeMlpModelSpecCode,
} from "../../graph/specCode/moeMlpModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: MoeMlpModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: MoeMlpModelNodeData,
  patch: Partial<MoeMlpModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: MoeMlpModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_MOE_MLP_PARAM_ORDER];
}

export function MoeMlpModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultMoeMlpModelData();
  const d = { ...defs, ...(data as Partial<MoeMlpModelNodeData>) } as MoeMlpModelNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "full_model"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_MOE_MLP_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateMoeMlpModelSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<MoeMlpModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "inputDim":
        return (
          <ComfyIntListField
            key={key}
            label="input dimension"
            values={intChoices(full.inputDim, 10)}
            min={1}
            onCommit={(vals) => update({ inputDim: packIntList(vals) })}
            ariaLabel="Input dimension"
          />
        );
      case "outputDim":
        return (
          <ComfyIntListField
            key={key}
            label="output dimension"
            values={intChoices(full.outputDim, 1)}
            min={1}
            onCommit={(vals) => update({ outputDim: packIntList(vals) })}
            ariaLabel="Output dimension"
          />
        );
      case "depth":
        return (
          <ComfyIntListField
            key={key}
            label="expert depth"
            values={intChoices(full.depth, 2)}
            min={1}
            title="Number of hidden layers in each expert."
            onCommit={(vals) => update({ depth: packIntList(vals) })}
            ariaLabel="Expert depth"
          />
        );
      case "width":
        return (
          <ComfyIntListField
            key={key}
            label="expert width"
            values={intChoices(full.width, 32)}
            min={1}
            title="Hidden width inside each expert MLP."
            onCommit={(vals) => update({ width: packIntList(vals) })}
            ariaLabel="Expert width"
          />
        );
      case "numExperts":
        return (
          <ComfyIntListField
            key={key}
            label="num experts"
            values={intChoices(full.numExperts, 4)}
            min={1}
            title="Number of experts mixed by softmax gating."
            onCommit={(vals) => update({ numExperts: packIntList(vals) })}
            ariaLabel="Number of experts"
          />
        );
      case "activation":
        return (
          <DiscreteMultiSelect
            key={key}
            label="expert activation"
            options={MLP_ACTIVATION_OPTIONS}
            value={full.activation}
            onCommit={(activation) => update({ activation })}
            ariaLabel="Expert activation function"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch RNG seed for gating and expert weight initialization."
            ariaLabel="Initialization seed"
            onCommit={(vals) => update({ seed: packIntList(vals) })}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--mlp-model${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "MoE MLP")}
            infoText="Uses softmax gates over experts: y = Sum_k softmax(Wg x)_k * expert_k(x)."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "MoE MLP")}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
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
