import { useCallback, useEffect, useMemo, useRef } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { AtomicLayerIoStrip } from "./AtomicLayerIoStrip";
import { CombinedModelIoStrip } from "./CombinedModelIoStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import {
  defaultAttentionOnlyModelData,
  type AttentionOnlyCausalId,
  type AttentionOnlyModelNodeData,
  type AttentionOnlyYesNoId,
} from "./attentionOnlyModelDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_ATTENTION_ONLY_PARAM_ORDER,
  DEFAULT_ATTENTION_ONLY_SPEC_NAME,
  generateAttentionOnlyModelSpecCode,
} from "../../graph/specCode/attentionOnlyModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import {
  pruneEdgesForNodeCanvasIoMode,
  readNodeCanvasIoMode,
  type NodeCanvasIoMode,
} from "../../graph/nodeCanvasIoMode";
import { readNodeCanvasLevelMode } from "../../graph/nodeCanvasLevelMode";
import { useResearchGraph } from "../../context/ResearchGraphContext";
import {
  reconcileAttentionLowExpansion,
  removeAttentionLowExpansionFromGraph,
} from "../../graph/attentionLowLevelExpansion";

const CAUSAL_ATTENTION_OPTIONS: { id: AttentionOnlyCausalId; label: string }[] = [
  { id: "yes", label: "Causal (masked self-attention)" },
  { id: "no", label: "Bidirectional (full context)" },
];

const YES_NO_OPTIONS: { id: AttentionOnlyYesNoId; label: string }[] = [
  { id: "yes", label: "yes" },
  { id: "no", label: "no" },
];

/** Canvas fields beyond the minimal generated spec (trainer reads these from node ``data``). */
const ATTENTION_TRAINER_FIELD_ORDER: string[] = [
  "vocabSize",
  "embedDim",
  "numHeads",
  "contextLength",
  "causalAttention",
  "localMixingKernel",
  "qkNorm",
  "attnTemperature",
  "attnLogitCap",
  "attnDropout",
  "seed",
];

function replaceNodeData(
  id: string,
  data: AttentionOnlyModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: AttentionOnlyModelNodeData,
  patch: Partial<AttentionOnlyModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: AttentionOnlyModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...ATTENTION_TRAINER_FIELD_ORDER];
}

export function AttentionOnlyModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultAttentionOnlyModelData();
  const d = { ...defs, ...(data as Partial<AttentionOnlyModelNodeData>) } as AttentionOnlyModelNodeData;
  const dRef = useRef(d);
  dRef.current = d;
  const research = useResearchGraph();
  const { setNodes: setNodesRf, setEdges: setEdgesRf, getNodes, getEdges } = useReactFlow();
  const setNodes = research?.setFlowNodes ?? setNodesRf;
  const setEdges = research?.setFlowEdges ?? setEdgesRf;

  const setNodesRef = useRef(setNodes);
  const setEdgesRef = useRef(setEdges);
  const getNodesRef = useRef(getNodes);
  const getEdgesRef = useRef(getEdges);
  setNodesRef.current = setNodes;
  setEdgesRef.current = setEdges;
  getNodesRef.current = getNodes;
  getEdgesRef.current = getEdges;

  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);
  const levelMode = readNodeCanvasLevelMode(d as Record<string, unknown>);

  const expansionSyncKey = useMemo(
    () =>
      JSON.stringify({
        levelMode,
        io: ioMode,
        v: d.vocabSize,
        d: d.embedDim,
        h: d.numHeads,
        l: d.contextLength,
        c: d.causalAttention,
        mixK: d.localMixingKernel,
        qk: d.qkNorm,
        at: d.attnTemperature,
        cap: d.attnLogitCap,
        ado: d.attnDropout,
        seed: d.seed,
      }),
    [
      levelMode,
      ioMode,
      d.vocabSize,
      d.embedDim,
      d.numHeads,
      d.contextLength,
      d.causalAttention,
      d.localMixingKernel,
      d.qkNorm,
      d.attnTemperature,
      d.attnLogitCap,
      d.attnDropout,
      d.seed,
    ],
  );

  useEffect(() => {
    const nds = getNodesRef.current();
    const eds = getEdgesRef.current();
    const r =
      levelMode !== "low"
        ? removeAttentionLowExpansionFromGraph(id, nds, eds)
        : reconcileAttentionLowExpansion(nds, eds, id, dRef.current);
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
  const specName = d.specCodeName ?? DEFAULT_ATTENTION_ONLY_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateAttentionOnlyModelSpecCode(d, order, specName),
    [d, order, specName],
  );

  const update = useCallback(
    (patch: Partial<AttentionOnlyModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "vocabSize":
        return (
          <ComfyIntListField
            key={key}
            label="vocab size (weight export)"
            values={intChoices(full.vocabSize, 100)}
            min={2}
            title="Used when resolving model_weight_tensors / checkpoints; token LM vocab comes from the dataset during training."
            onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
            ariaLabel="Vocabulary size"
          />
        );
      case "embedDim":
        return (
          <ComfyIntListField
            key={key}
            label="model dim d"
            values={intChoices(full.embedDim, 32)}
            min={1}
            ariaLabel="Per-token model dimension"
            onCommit={(vals) => update({ embedDim: packIntList(vals) })}
          />
        );
      case "contextLength":
        return (
          <ComfyIntListField
            key={key}
            label="context length L"
            values={intChoices(full.contextLength, 4)}
            min={1}
            ariaLabel="Context length"
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
          />
        );
      case "numHeads":
        return (
          <ComfyIntListField
            key={key}
            label="num heads"
            values={intChoices(full.numHeads, 1)}
            min={1}
            ariaLabel="Number of attention heads"
            onCommit={(vals) => update({ numHeads: packIntList(vals) })}
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch RNG seed for weight initialization (reproducible runs)"
            ariaLabel="Initialization seed"
            onCommit={(vals) => update({ seed: packIntList(vals) })}
          />
        );
      case "causalAttention":
        return (
          <DiscreteMultiSelect<AttentionOnlyCausalId>
            key={key}
            label="self-attention"
            options={CAUSAL_ATTENTION_OPTIONS}
            singleSelect
            value={full.causalAttention}
            onCommit={(causalAttention) => update({ causalAttention })}
            ariaLabel="Causal vs bidirectional self-attention"
          />
        );
      case "localMixingKernel":
        return (
          <ComfyIntListField
            key={key}
            label="local mixing kernel"
            values={intChoices(full.localMixingKernel ?? 0, 0)}
            min={0}
            title="Causal depthwise conv after token embedding in CE training (0–2 off; odd kernels ≥3). Canon-lite horizontal mixing."
            ariaLabel="Local mixing kernel size"
            onCommit={(vals) => update({ localMixingKernel: packIntList(vals) })}
          />
        );
      case "qkNorm":
        return (
          <DiscreteMultiSelect<AttentionOnlyYesNoId>
            key={key}
            label="QK RMSNorm"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.qkNorm}
            onCommit={(qkNorm) => update({ qkNorm })}
            ariaLabel="Normalize query/key vectors"
          />
        );
      case "attnTemperature":
        return (
          <ComfyFloatListField
            key={key}
            label="attention temperature"
            values={floatChoices(full.attnTemperature, 1)}
            positiveOnly={false}
            onCommit={(vals) => update({ attnTemperature: packFloatList(vals) })}
            ariaLabel="Attention softmax temperature"
          />
        );
      case "attnLogitCap":
        return (
          <ComfyFloatListField
            key={key}
            label="attention logit cap"
            values={floatChoices(full.attnLogitCap, 0)}
            positiveOnly={false}
            title="0 = no tanh cap on attention logits before softmax."
            onCommit={(vals) => update({ attnLogitCap: packFloatList(vals) })}
            ariaLabel="Attention logit cap"
          />
        );
      case "attnDropout":
        return (
          <ComfyFloatListField
            key={key}
            label="attention dropout"
            values={floatChoices(full.attnDropout, 0)}
            positiveOnly={false}
            onCommit={(vals) => update({ attnDropout: packFloatList(vals) })}
            ariaLabel="Attention dropout probability"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`cr-node cr-node--attention-model${levelMode === "low" ? " cr-node--attention-model-low-shell" : ""}${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
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
              infoTitle={readInstanceTitle(d, "Attention layer")}
              infoText={`Multi-head self-attention on activations [batch, L, d] -> same shape. Token CE training embeds ids, optionally applies causal depthwise local mixing (Canon-lite), then attention + lm_head; wire d and L to match your dataset.

**References:** [Part 1 (CFG / structure)](https://arxiv.org/abs/2305.13673), [Part 3.1 (knowledge)](https://arxiv.org/abs/2309.14316), [Part 4.1 (local mixing)](https://arxiv.org/abs/2512.17351), [series hub](https://physics.allen-zhu.com/).`}
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
        {readInstanceTitle(d, "Attention layer")}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body cr-node__body--compact">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        ) : (
          <>
            <ModelInitializationTargetRow />
            {levelMode === "low" ? <CombinedModelIoStrip /> : <AtomicLayerIoStrip />}
          </>
        )}
        {levelMode !== "low" ? order.map((key) => renderField(key)) : null}
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
