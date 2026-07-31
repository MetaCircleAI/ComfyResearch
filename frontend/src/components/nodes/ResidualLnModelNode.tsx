import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { MLP_ACTIVATION_OPTIONS, type MlpActivationId } from "./mlpModelDefaults";
import {
  defaultResidualLnModelData,
  type ResidualLnModelNodeData,
  type ResidualLnMode,
} from "./residualLnModelDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_RESIDUAL_LN_PARAM_ORDER,
  DEFAULT_RESIDUAL_LN_SPEC_NAME,
  generateResidualLnModelSpecCode,
} from "../../graph/specCode/residualLnModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

const LN_MODE_OPTIONS: { id: ResidualLnMode; label: string }[] = [
  { id: "pre_ln", label: "Pre-LN" },
  { id: "post_ln", label: "Post-LN" },
];

function replaceNodeData(
  id: string,
  data: ResidualLnModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: ResidualLnModelNodeData,
  patch: Partial<ResidualLnModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: ResidualLnModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_RESIDUAL_LN_PARAM_ORDER];
}

export function ResidualLnModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultResidualLnModelData();
  const d = { ...defs, ...(data as Partial<ResidualLnModelNodeData>) } as ResidualLnModelNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "full_model"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_RESIDUAL_LN_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateResidualLnModelSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<ResidualLnModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "dim":
        return (
          <ComfyIntListField
            key={key}
            label="dimension d"
            values={intChoices(full.dim, 256)}
            min={1}
            ariaLabel="Residual stream dimension"
            onCommit={(vals) => update({ dim: packIntList(vals) })}
          />
        );
      case "depth":
        return (
          <ComfyIntListField
            key={key}
            label="depth L"
            values={intChoices(full.depth, 100)}
            min={1}
            ariaLabel="Residual depth"
            onCommit={(vals) => update({ depth: packIntList(vals) })}
          />
        );
      case "alpha":
        return (
          <ComfyFloatListField
            key={key}
            label="alpha (FC2 scale)"
            values={floatChoices(full.alpha, 1)}
            positiveOnly
            ariaLabel="Residual FC2 scale"
            onCommit={(vals) => update({ alpha: packFloatList(vals) })}
          />
        );
      case "lnMode":
        return (
          <DiscreteMultiSelect<ResidualLnMode>
            key={key}
            label="norm mode"
            options={LN_MODE_OPTIONS}
            value={full.lnMode}
            onCommit={(lnMode) => update({ lnMode })}
            ariaLabel="Residual norm mode"
          />
        );
      case "activation":
        return (
          <DiscreteMultiSelect<MlpActivationId>
            key={key}
            label="activation"
            options={MLP_ACTIVATION_OPTIONS}
            value={full.activation}
            onCommit={(activation) => update({ activation })}
            ariaLabel="Residual block activation"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch RNG seed for weight initialization"
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
            infoTitle={readInstanceTitle(d, "Residual LN model")}
            infoText="Residual MLP block with optional pre/post LayerNorm. Useful for compact transformer-style residual stacks."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "Residual LN model")}
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
