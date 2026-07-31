import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { KanModelNodeData } from "./kanModelDefaults";
import { defaultKanModelData, KAN_BASE_FUN_OPTIONS } from "./kanModelDefaults";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  DEFAULT_KAN_PARAM_ORDER,
  DEFAULT_KAN_SPEC_NAME,
  generateKanModelSpecCode,
} from "../../graph/specCode/kanModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";

function replaceNodeData(
  id: string,
  data: KanModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: KanModelNodeData,
  patch: Partial<KanModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: KanModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_KAN_PARAM_ORDER];
}

export function KanModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultKanModelData();
  const d = { ...defs, ...(data as Partial<KanModelNodeData>) } as KanModelNodeData;
  const { setNodes, setEdges } = useReactFlow();

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "full_model"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_KAN_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateKanModelSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<KanModelNodeData>) => patchData(id, d, patch, setNodes),
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
            label="depth"
            values={intChoices(full.depth, 2)}
            min={1}
            title="Number of hidden KAN layers"
            onCommit={(vals) => update({ depth: packIntList(vals) })}
            ariaLabel="Hidden depth (number of hidden KAN layers)"
          />
        );
      case "width":
        return (
          <ComfyIntListField
            key={key}
            label="width"
            values={intChoices(full.width, 5)}
            min={1}
            title="Width of each hidden layer (pykan width list segment)"
            onCommit={(vals) => update({ width: packIntList(vals) })}
            ariaLabel="Hidden layer width"
          />
        );
      case "grid":
        return (
          <ComfyIntListField
            key={key}
            label="grid"
            values={intChoices(full.grid, 3)}
            min={1}
            title="B-spline grid intervals (pykan grid)"
            onCommit={(vals) => update({ grid: packIntList(vals) })}
            ariaLabel="KAN grid intervals"
          />
        );
      case "k":
        return (
          <ComfyIntListField
            key={key}
            label="spline order k"
            values={intChoices(full.k, 3)}
            min={1}
            title="Piecewise polynomial order (pykan k)"
            onCommit={(vals) => update({ k: packIntList(vals) })}
            ariaLabel="Spline order k"
          />
        );
      case "baseFun":
        return (
          <DiscreteMultiSelect
            key={key}
            label="base fun"
            options={KAN_BASE_FUN_OPTIONS}
            value={full.baseFun}
            onCommit={(baseFun) => update({ baseFun })}
            ariaLabel="KAN base function (MultKAN string)"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="RNG seed for pykan / PyTorch initialization"
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
      className={`cr-node cr-node--kan-model${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          <NodeSpecHeaderActions
            nodeId={id}
            generatedCode={generatedCode}
            infoTitle={readInstanceTitle(d, "KAN")}
            infoText="Kolmogorov-Arnold network via pykan; width = [input] + depth x [hidden] + [output]. Commas on numeric fields run sweeps."
          />
        }
        subtitle={d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "KAN")}
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
