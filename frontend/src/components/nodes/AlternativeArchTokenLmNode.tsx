import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { buildAlternativeArchTokenLmNotebookPython } from "../../graph/specCode/alternativeArchNotebookSpecCode";
import {
  defaultAlternativeArchTokenLmData,
  type AlternativeArchTokenLmNodeData,
  type ArchLmKind,
  type CausalAttnId,
} from "./alternativeArchModelDefaults";

const KIND_SET = new Set<string>([
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
]);

const TITLES: Record<ArchLmKind, string> = {
  linear_attention_model: "Linear attention (tokens)",
  diagonal_ssm_token_model: "Diagonal SSM (tokens)",
  rwkv_time_mix_token_model: "RWKV-lite time-mix (tokens)",
  hyena_like_conv_model: "Hyena-like conv (tokens)",
  slot_attention_token_model: "Slot attention (tokens)",
};

const BLURBS: Record<ArchLmKind, string> = {
  linear_attention_model:
    "Causal linear attention (ELU+1 kernel) with optional depthwise conv mixing; last-token CE logits.",
  diagonal_ssm_token_model:
    "Input-dependent diagonal state-space recurrence per timestep; residual stack + LayerNorm → logits.",
  rwkv_time_mix_token_model:
    "Gated recurrence over time + gated FF (RWKV-flavored); LM head on last position.",
  hyena_like_conv_model:
    "Causal depthwise conv blocks + gated FFN; sequence mixer alternative to attention.",
  slot_attention_token_model:
    "Slot attention reads off token embeddings; slot mean → MLP → vocab logits.",
};

const CAUSAL_OPTS: { id: CausalAttnId; label: string }[] = [
  { id: "yes", label: "Causal (past-only)" },
  { id: "no", label: "Bidirectional" },
];

function replaceNodeData(
  id: string,
  data: AlternativeArchTokenLmNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function AlternativeArchTokenLmNode({ id, data, selected, type }: NodeProps) {
  const kindRaw = String(type ?? "");
  const kind = (KIND_SET.has(kindRaw) ? kindRaw : "linear_attention_model") as ArchLmKind;
  const defs = defaultAlternativeArchTokenLmData(kind);
  const d = { ...defs, ...(data as Partial<AlternativeArchTokenLmNodeData>) } as AlternativeArchTokenLmNodeData;
  const { setNodes, getNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<AlternativeArchTokenLmNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  const title = TITLES[kind];
  const generatedCode = useMemo(() => {
    const slug = pySlugForNode(id, getNodes());
    const cellTitle = readInstanceTitle(d as Record<string, unknown>, title);
    return buildAlternativeArchTokenLmNotebookPython(slug, cellTitle, kind, d as unknown as Record<string, unknown>);
  }, [d, getNodes, id, kind, title]);
  const blurb = BLURBS[kind];

  return (
    <div
      className={`cr-node cr-node--mlp-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d as Record<string, unknown>, title)}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d as Record<string, unknown>, title)}
              infoText={blurb}
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        <ComfyIntListField
          label="vocab size V"
          values={intChoices(d.vocabSize, 100)}
          min={2}
          onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="embed dim D"
          values={intChoices(d.embedDim, 32)}
          min={1}
          onCommit={(vals) => update({ embedDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="context length L"
          values={intChoices(d.contextLength, 8)}
          min={1}
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
        />
        {kind === "linear_attention_model" ? (
          <>
            <ComfyIntListField
              label="num heads"
              values={intChoices(d.numHeads ?? 4, 4)}
              min={1}
              onCommit={(vals) => update({ numHeads: packIntList(vals) })}
            />
            <DiscreteMultiSelect
              label="causal"
              options={CAUSAL_OPTS}
              value={d.causalAttention ?? "yes"}
              singleSelect
              onCommit={(causalAttention) => update({ causalAttention })}
            />
          </>
        ) : null}
        {kind === "diagonal_ssm_token_model" ? (
          <ComfyIntListField
            label="num layers"
            values={intChoices(d.numLayers ?? 2, 2)}
            min={1}
            onCommit={(vals) => update({ numLayers: packIntList(vals) })}
          />
        ) : null}
        {kind === "rwkv_time_mix_token_model" || kind === "hyena_like_conv_model" ? (
          <ComfyIntListField
            label="depth"
            values={intChoices(d.depth ?? 2, 2)}
            min={1}
            onCommit={(vals) => update({ depth: packIntList(vals) })}
          />
        ) : null}
        {kind === "hyena_like_conv_model" ? (
          <>
            <ComfyIntListField
              label="conv kernel (odd)"
              values={intChoices(d.convKernel ?? 7, 7)}
              min={3}
              onCommit={(vals) => update({ convKernel: packIntList(vals) })}
            />
            <ComfyIntListField
              label="FF mult"
              values={intChoices(d.ffMult ?? 2, 2)}
              min={1}
              onCommit={(vals) => update({ ffMult: packIntList(vals) })}
            />
          </>
        ) : null}
        {kind === "slot_attention_token_model" ? (
          <>
            <ComfyIntListField
              label="num slots"
              values={intChoices(d.numSlots ?? 4, 4)}
              min={1}
              onCommit={(vals) => update({ numSlots: packIntList(vals) })}
            />
            <ComfyIntListField
              label="slot iters"
              values={intChoices(d.slotIters ?? 3, 3)}
              min={1}
              onCommit={(vals) => update({ slotIters: packIntList(vals) })}
            />
          </>
        ) : null}
        <ComfyIntListField
          label="local mixing kernel"
          values={intChoices(d.localMixingKernel, 0)}
          min={0}
          title="Causal depthwise conv after embedding (0–2 off; odd ≥3 enables)."
          onCommit={(vals) => update({ localMixingKernel: packIntList(vals) })}
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(d.seed, 0)}
          min={0}
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
