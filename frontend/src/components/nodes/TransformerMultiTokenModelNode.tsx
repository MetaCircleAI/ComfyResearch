import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList, type ListOr1 } from "./multiValueUtils";
import {
  defaultTransformerMultiTokenModelData,
  type TransformerMultiTokenCausalId,
  type TransformerMultiTokenModelNodeData,
  type TransformerMultiTokenTieId,
} from "./transformerMultiTokenModelDefaults";
import {
  TRANSFORMER_ENCODER_BACKEND_OPTIONS,
  type TransformerEncoderBackendId,
} from "./transformerTokenModelDefaults";
import {
  DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER,
  DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_SPEC_NAME,
  generateTransformerMultiTokenModelSpecCode,
} from "../../graph/specCode/transformerMultiTokenModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

const CAUSAL_ATTENTION_OPTIONS: { id: TransformerMultiTokenCausalId; label: string }[] = [
  { id: "yes", label: "Causal (masked self-attention)" },
  { id: "no", label: "Bidirectional (full context)" },
];

const YES_NO_OPTIONS: { id: TransformerMultiTokenTieId; label: string }[] = [
  { id: "yes", label: "yes" },
  { id: "no", label: "no" },
];

function firstScalarListOr1<T>(v: ListOr1<T>): T {
  return Array.isArray(v) ? v[0]! : v;
}

function replaceNodeData(
  id: string,
  data: TransformerMultiTokenModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: TransformerMultiTokenModelNodeData,
  patch: Partial<TransformerMultiTokenModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: TransformerMultiTokenModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER];
}

const LAYOUT_PARAM_KEYS = new Set<string>(["contextLength", "vocabSize", "tokensPerPosition"]);

function displayParamOrder(d: TransformerMultiTokenModelNodeData): string[] {
  const base = effectiveParamOrder(d);
  const leading = [...LAYOUT_PARAM_KEYS].filter((k) => !base.includes(k));
  return [...leading, ...base];
}

export function TransformerMultiTokenModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultTransformerMultiTokenModelData();
  const d = { ...defs, ...(data as Partial<TransformerMultiTokenModelNodeData>) } as TransformerMultiTokenModelNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => displayParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateTransformerMultiTokenModelSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<TransformerMultiTokenModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const tieChecked = String(firstScalarListOr1(d.tieEmbeddingLmHead) ?? "no").toLowerCase() !== "no";

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextLength":
        return (
          <ComfyIntListField
            key={key}
            label="context length (timesteps L)"
            values={intChoices(full.contextLength, 4)}
            min={1}
            title="Number of timesteps along the sequence (each timestep has K token ids)"
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
            ariaLabel="Vocabulary size per slot"
          />
        );
      case "tokensPerPosition":
        return (
          <ComfyIntListField
            key={key}
            label="tokens per position (K)"
            values={intChoices(full.tokensPerPosition, 2)}
            min={1}
            title="Circular motion dataset uses K = 2 (x and y quantization per timestep)"
            onCommit={(vals) => update({ tokensPerPosition: packIntList(vals) })}
            ariaLabel="Tokens per position"
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
            onCommit={(vals) => update({ encoderDropout: packFloatList(vals) })}
            ariaLabel="Encoder dropout probability"
          />
        );
      case "spectralNormLinears":
        return (
          <DiscreteMultiSelect<TransformerMultiTokenTieId>
            key={key}
            label="spectral norm (encoder Linear)"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.spectralNormLinears}
            onCommit={(spectralNormLinears) => update({ spectralNormLinears })}
            ariaLabel="Spectral norm on encoder linear maps"
          />
        );
      case "lmLogitScale":
        return (
          <ComfyFloatListField
            key={key}
            label="LM logit scale"
            values={floatChoices(full.lmLogitScale, 1)}
            positiveOnly={false}
            onCommit={(vals) => update({ lmLogitScale: packFloatList(vals) })}
            ariaLabel="LM head logit scale"
          />
        );
      case "stableQkNorm":
        return (
          <DiscreteMultiSelect<TransformerMultiTokenTieId>
            key={key}
            label="stable QK norm"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.stableQkNorm}
            onCommit={(stableQkNorm) => update({ stableQkNorm })}
            ariaLabel="Stable encoder QK RMSNorm"
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
            ariaLabel="Attention temperature"
          />
        );
      case "stableAttnLogitCap":
        return (
          <ComfyFloatListField
            key={key}
            label="stable attn logit cap"
            values={floatChoices(full.stableAttnLogitCap, 0)}
            positiveOnly={false}
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
            ariaLabel="Attention dropout"
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
          <DiscreteMultiSelect<TransformerMultiTokenCausalId>
            key={key}
            label="self-attention"
            options={CAUSAL_ATTENTION_OPTIONS}
            singleSelect
            value={full.causalAttention}
            onCommit={(causalAttention) => update({ causalAttention })}
            ariaLabel="Causal vs bidirectional self-attention"
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
      className={`cr-node cr-node--transformer-multi-token-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, "Transformer (multiple tokens)")}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "Transformer (multiple tokens)")}
              infoText="Token ids [batch, L, K] per timestep -> last-timestep logits [batch, K, V] (cross-entropy over K heads). Pair with circular motion dataset (K = 2)."
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
