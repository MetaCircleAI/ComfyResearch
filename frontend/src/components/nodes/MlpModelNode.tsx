import { useCallback, useEffect, useMemo, useRef } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { MlpModelNodeData } from "./mlpModelDefaults";
import { defaultMlpModelData, MLP_ACTIVATION_OPTIONS } from "./mlpModelDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { CombinedModelIoStrip } from "./CombinedModelIoStrip";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_MLP_PARAM_ORDER,
  DEFAULT_MLP_SPEC_NAME,
  generateMlpModelSpecCode,
} from "../../graph/specCode/mlpModelSpecCode";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";
import { useResearchGraph } from "../../context/ResearchGraphContext";
import { readNodeCanvasLevelMode } from "../../graph/nodeCanvasLevelMode";
import { reconcileMlpLowExpansion, removeMlpLowExpansionFromGraph } from "../../graph/mlpLowLevelExpansion";

function replaceNodeData(
  id: string,
  data: MlpModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: MlpModelNodeData,
  patch: Partial<MlpModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: MlpModelNodeData): string[] {
  const custom = d.paramOrder?.filter((k) => typeof k === "string" && k.length > 0) ?? [];
  if (!custom.length) return [...DEFAULT_MLP_PARAM_ORDER];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const k of custom) {
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  for (const k of DEFAULT_MLP_PARAM_ORDER) {
    if (!seen.has(String(k))) out.push(String(k));
  }
  return out;
}

export function MlpModelNode({ id, data, selected, type }: NodeProps) {
  const defs = defaultMlpModelData();
  const d = { ...defs, ...(data as Partial<MlpModelNodeData>) } as MlpModelNodeData;
  // outputScale is mlp_model-only (backend helper scopes to plain nn.Sequential MLPs);
  // non-MLP node types must not receive it through default merging.
  if (type !== "mlp_model" && !(data && "outputScale" in (data as object))) delete d.outputScale;
  const dRef = useRef(d);
  dRef.current = d;
  const research = useResearchGraph();
  const { setNodes: setNodesRf, setEdges: setEdgesRf, getNodes, getEdges } = useReactFlow();
  const setNodes = research?.setFlowNodes ?? setNodesRf;
  const setEdges = research?.setFlowEdges ?? setEdgesRf;

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const levelMode = readNodeCanvasLevelMode(d as Record<string, unknown>);

  const expansionSyncKey = useMemo(
    () =>
      JSON.stringify({
        lm: levelMode,
        io: ioMode,
        depth: d.depth,
        inputDim: d.inputDim,
        outputDim: d.outputDim,
        width: d.width,
        activation: d.activation,
        seed: d.seed,
      }),
    [levelMode, ioMode, d.depth, d.inputDim, d.outputDim, d.width, d.activation, d.seed],
  );

  const setNodesRef = useRef(setNodes);
  const setEdgesRef = useRef(setEdges);
  const getNodesRef = useRef(getNodes);
  const getEdgesRef = useRef(getEdges);
  setNodesRef.current = setNodes;
  setEdgesRef.current = setEdges;
  getNodesRef.current = getNodes;
  getEdgesRef.current = getEdges;

  useEffect(() => {
    const nds = getNodesRef.current();
    const eds = getEdgesRef.current();
    const r =
      levelMode !== "low"
        ? removeMlpLowExpansionFromGraph(id, nds, eds)
        : reconcileMlpLowExpansion(nds, eds, id, dRef.current);
    const skipped = r.nodes === nds && r.edges === eds;
    if (skipped) return;
    setNodesRef.current(r.nodes);
    setEdgesRef.current(r.edges);
  }, [expansionSyncKey, id, levelMode]);

  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      setEdges((eds) => pruneEdgesForNodeCanvasIoMode(eds, id, next, "full_model"));
    },
    [id, setEdges],
  );

  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_MLP_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateMlpModelSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<MlpModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string, row: number) => {
    const rk = `${key}-${row}`;
    const full = { ...defs, ...d };
    switch (key) {
      case "inputDim":
        return (
          <ComfyIntListField
            key={rk}
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
            key={rk}
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
            key={rk}
            label="depth"
            values={intChoices(full.depth, 2)}
            min={1}
            title="Number of hidden layers"
            onCommit={(vals) => update({ depth: packIntList(vals) })}
            ariaLabel="Hidden depth (number of hidden layers)"
          />
        );
      case "width":
        return (
          <ComfyIntListField
            key={rk}
            label="width"
            values={intChoices(full.width, 64)}
            min={1}
            title="Width of each hidden layer"
            onCommit={(vals) => update({ width: packIntList(vals) })}
            ariaLabel="Hidden layer width"
          />
        );
      case "activation":
        return (
          <DiscreteMultiSelect
            key={rk}
            label="activation"
            options={MLP_ACTIVATION_OPTIONS}
            value={full.activation}
            onCommit={(activation) => update({ activation })}
            ariaLabel="Activation function"
          />
        );
      case "outputScale":
        // mlp_model-only knob: the backend helper scopes to plain nn.Sequential MLPs,
        // so non-MLP node types must not expose it.
        if (type !== "mlp_model") return null;
        return (
          <ComfyFloatListField
            key={rk}
            label="output scale α"
            values={floatChoices(full.outputScale ?? 1, 1)}
            positiveOnly={false}
            title="Multiply output-layer weights by α after init. Small α → rich/feature-learning; large α → lazy/NTK (Chizat 2019). Use commas for a sweep."
            ariaLabel="Output scale alpha"
            onCommit={(vals) => update({ outputScale: packFloatList(vals) })}
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={rk}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch RNG seed for weight initialization (dataset sampling uses the dataset node seed)"
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
      className={`cr-node cr-node--mlp-model${levelMode === "low" ? " cr-node--mlp-model-low-shell" : ""}${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{
        ["--accent" as string]: "var(--cr-accent-model)",
        ...(levelMode === "low"
          ? { height: "100%", minHeight: "100%", boxSizing: "border-box" as const }
          : {}),
      }}
    >
      <NodeHeaderWithIoMode
        id={id}
        data={d as Record<string, unknown>}
        headerActions={
          levelMode !== "low" ? (
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "MLP")}
              infoText={`input -> ${intChoices(d.depth, 2)[0]}x(${intChoices(d.width, 64)[0]}, ...) -> output.\nUse commas on numeric fields for multiple runs.`}
            />
          ) : null
        }
        subtitle={
          levelMode !== "low" && d.specCodeName ? (
            <span className="cr-node__header-sub">{d.specCodeName}</span>
          ) : null
        }
        onIoModeChange={onIoModeChange}
      >
        {readInstanceTitle(d, "MLP")}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        ) : (
          <>
            <ModelInitializationTargetRow />
            {levelMode === "low" ? <CombinedModelIoStrip /> : <AtomicLayerIoStrip />}
          </>
        )}

        {levelMode !== "low" ? order.map((key, idx) => renderField(key, idx)) : null}

        {levelMode !== "low" && d.extras && Object.keys(d.extras).length > 0 ? (
          <p className="cr-node__hint cr-node__hint--extras">
            Extra params from spec (not used by training): {JSON.stringify(d.extras)}
          </p>
        ) : null}

        {levelMode !== "low" ? (
          <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
        ) : null}
      </div>
    </div>
  );
}
