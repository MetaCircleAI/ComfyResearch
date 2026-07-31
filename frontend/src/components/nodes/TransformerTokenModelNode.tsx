import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultTransformerTokenModelData,
  TRANSFORMER_ENCODER_ACTIVATION_OPTIONS,
  TRANSFORMER_ENCODER_BACKEND_OPTIONS,
  type TransformerEncoderBackendId,
  type TransformerTokenCausalId,
  type TransformerTokenEncoderActivationId,
  type TransformerTokenModelNodeData,
  type TransformerTokenTieId,
} from "./transformerTokenModelDefaults";
import {
  DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER,
  DEFAULT_TRANSFORMER_TOKEN_MODEL_SPEC_NAME,
  generateTransformerTokenModelSpecCode,
} from "../../graph/specCode/transformerTokenModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

const CAUSAL_ATTENTION_OPTIONS: { id: TransformerTokenCausalId; label: string }[] = [
  { id: "yes", label: "Causal (masked self-attention)" },
  { id: "no", label: "Bidirectional (full context)" },
];

const YES_NO_OPTIONS: { id: TransformerTokenTieId; label: string }[] = [
  { id: "yes", label: "yes" },
  { id: "no", label: "no" },
];

function replaceNodeData(
  id: string,
  data: TransformerTokenModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: TransformerTokenModelNodeData,
  patch: Partial<TransformerTokenModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: TransformerTokenModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER];
}

const LAYOUT_PARAM_KEYS = new Set<string>(["contextLength", "vocabSize"]);

function displayParamOrder(d: TransformerTokenModelNodeData): string[] {
  const base = effectiveParamOrder(d);
  const leading = [...LAYOUT_PARAM_KEYS].filter((k) => !base.includes(k));
  return [...leading, ...base];
}

export function TransformerTokenModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultTransformerTokenModelData();
  const d = { ...defs, ...(data as Partial<TransformerTokenModelNodeData>) } as TransformerTokenModelNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => displayParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_TRANSFORMER_TOKEN_MODEL_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateTransformerTokenModelSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<TransformerTokenModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const tieChoice = Array.isArray(d.tieEmbeddingLmHead) ? d.tieEmbeddingLmHead[0] : d.tieEmbeddingLmHead;
  const tieChecked = tieChoice !== "no";

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextLength":
        return (
          <ComfyIntListField
            key={key}
            label="context length (T)"
            values={intChoices(full.contextLength, 4)}
            min={1}
            title="Sequence length for token ids [batch, T] (must match token datasets)"
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
            ariaLabel="Context length"
          />
        );
      case "vocabSize":
        return (
          <ComfyIntListField
            key={key}
            label="vocab size V"
            values={intChoices(full.vocabSize, 100)}
            min={2}
            onCommit={(vals) => update({ vocabSize: packIntList(vals) })}
            ariaLabel="Vocabulary size"
          />
        );
      case "modelDim":
        return (
          <ComfyIntListField
            key={key}
            label="model dim (d_model)"
            values={intChoices(full.modelDim, 32)}
            min={1}
            onCommit={(vals) => update({ modelDim: packIntList(vals) })}
            ariaLabel="Transformer hidden size"
          />
        );
      case "numHeads":
        return (
          <ComfyIntListField
            key={key}
            label="num heads"
            values={intChoices(full.numHeads, 1)}
            min={1}
            onCommit={(vals) => update({ numHeads: packIntList(vals) })}
            ariaLabel="Number of attention heads"
          />
        );
      case "numLayers":
        return (
          <ComfyIntListField
            key={key}
            label="num layers"
            values={intChoices(full.numLayers, 1)}
            min={1}
            onCommit={(vals) => update({ numLayers: packIntList(vals) })}
            ariaLabel="Number of encoder layers"
          />
        );
      case "ffDim":
        return (
          <ComfyIntListField
            key={key}
            label="FF dim"
            values={intChoices(full.ffDim, 64)}
            min={1}
            onCommit={(vals) => update({ ffDim: packIntList(vals) })}
            ariaLabel="Feedforward hidden size"
          />
        );
      case "activation":
        return (
          <DiscreteMultiSelect<TransformerTokenEncoderActivationId>
            key={key}
            label="encoder activation"
            options={TRANSFORMER_ENCODER_ACTIVATION_OPTIONS}
            singleSelect
            value={full.activation}
            onCommit={(activation) => update({ activation })}
            ariaLabel="Transformer encoder feed-forward activation"
          />
        );
      case "encoderBackend":
        return (
          <DiscreteMultiSelect<TransformerEncoderBackendId>
            key={key}
            label="encoder backend"
            options={TRANSFORMER_ENCODER_BACKEND_OPTIONS}
            singleSelect
            value={full.encoderBackend}
            onCommit={(encoderBackend) => update({ encoderBackend })}
            ariaLabel="Transformer encoder implementation"
          />
        );
      case "encoderDropout":
        return (
          <ComfyFloatListField
            key={key}
            label="encoder dropout"
            values={floatChoices(full.encoderDropout, 0)}
            positiveOnly={false}
            title="Dropout inside encoder layers (0–1)."
            onCommit={(vals) => update({ encoderDropout: packFloatList(vals) })}
            ariaLabel="Encoder dropout probability"
          />
        );
      case "spectralNormLinears":
        return (
          <DiscreteMultiSelect<TransformerTokenTieId>
            key={key}
            label="spectral norm (encoder Linear)"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.spectralNormLinears}
            onCommit={(spectralNormLinears) => update({ spectralNormLinears })}
            ariaLabel="Spectral normalization on encoder linear maps"
          />
        );
      case "lmLogitScale":
        return (
          <ComfyFloatListField
            key={key}
            label="LM logit scale"
            values={floatChoices(full.lmLogitScale, 1)}
            positiveOnly={false}
            title="Multiplies logits before CE (1 = identity)."
            onCommit={(vals) => update({ lmLogitScale: packFloatList(vals) })}
            ariaLabel="LM head logit scale"
          />
        );
      case "stableQkNorm":
        return (
          <DiscreteMultiSelect<TransformerTokenTieId>
            key={key}
            label="stable QK norm"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.stableQkNorm}
            onCommit={(stableQkNorm) => update({ stableQkNorm })}
            ariaLabel="RMSNorm on Q/K in stable encoder"
          />
        );
      case "stableAttnTemperature":
        return (
          <ComfyFloatListField
            key={key}
            label="stable attn temperature"
            values={floatChoices(full.stableAttnTemperature, 1)}
            positiveOnly={false}
            onCommit={(vals) => update({ stableAttnTemperature: packFloatList(vals) })}
            ariaLabel="Attention softmax temperature"
          />
        );
      case "stableAttnLogitCap":
        return (
          <ComfyFloatListField
            key={key}
            label="stable attn logit cap"
            values={floatChoices(full.stableAttnLogitCap, 0)}
            positiveOnly={false}
            title="0 = no cap on pre-softmax logits."
            onCommit={(vals) => update({ stableAttnLogitCap: packFloatList(vals) })}
            ariaLabel="Attention logit cap"
          />
        );
      case "stableAttnDropout":
        return (
          <ComfyFloatListField
            key={key}
            label="stable attn dropout"
            values={floatChoices(full.stableAttnDropout, 0)}
            positiveOnly={false}
            onCommit={(vals) => update({ stableAttnDropout: packFloatList(vals) })}
            ariaLabel="Attention dropout probability"
          />
        );
      case "tieEmbeddingLmHead":
        return (
          <label key={key} className="cr-node__field cr-node__field--checkbox">
            <input
              type="checkbox"
              checked={tieChecked}
              onChange={(e) => update({ tieEmbeddingLmHead: e.target.checked ? "yes" : "no" })}
            />
            <span>Tie embedding and lm_head weights</span>
          </label>
        );
      case "causalAttention":
        return (
          <DiscreteMultiSelect<TransformerTokenCausalId>
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
            title="Causal depthwise conv on embeddings before encoder (0–2 off; odd ≥3). Canon-lite horizontal mixing."
            ariaLabel="Local mixing kernel size"
            onCommit={(vals) => update({ localMixingKernel: packIntList(vals) })}
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
      className={`cr-node cr-node--transformer-token-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, "Transformer (tokens)")}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "Transformer (tokens)")}
              infoText={`Token ids [batch, T] -> last-position logits [batch, V] (cross-entropy trainer). Optional tied embedding / lm_head like GPT-style models.

**References:** [Physics of LLMs — Part 3.1 (knowledge in LMs)](https://arxiv.org/abs/2309.14316), [Part 1 (synthetic LMs)](https://arxiv.org/abs/2305.13673), [series hub](https://physics.allen-zhu.com/).`}
            />
          </div>
        </div>
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        {order.map((key) => renderField(key))}
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
