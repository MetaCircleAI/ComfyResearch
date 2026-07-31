import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultNumericTransformerModelData,
  type NumericTransformerCausalId,
  type NumericTransformerModelNodeData,
} from "./numericTransformerModelDefaults";
import {
  TRANSFORMER_ENCODER_ACTIVATION_OPTIONS,
  TRANSFORMER_ENCODER_BACKEND_OPTIONS,
  type TransformerEncoderBackendId,
  type TransformerTokenEncoderActivationId,
} from "./transformerTokenModelDefaults";
import {
  DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER,
  DEFAULT_NUMERIC_TRANSFORMER_MODEL_SPEC_NAME,
  generateNumericTransformerModelSpecCode,
} from "../../graph/specCode/numericTransformerModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

const CAUSAL_ATTENTION_OPTIONS: { id: NumericTransformerCausalId; label: string }[] = [
  { id: "yes", label: "Causal (masked self-attention)" },
  { id: "no", label: "Bidirectional (full context)" },
];

const YES_NO_OPTIONS: { id: NumericTransformerCausalId; label: string }[] = [
  { id: "yes", label: "yes" },
  { id: "no", label: "no" },
];

function replaceNodeData(
  id: string,
  data: NumericTransformerModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: NumericTransformerModelNodeData,
  patch: Partial<NumericTransformerModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: NumericTransformerModelNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER];
}

/** I/O layout keys must stay in node data even if edited spec omits them from `__init__`. */
const LAYOUT_PARAM_KEYS = new Set<string>(["contextLength", "inputDim", "outputDim"]);

function displayParamOrder(d: NumericTransformerModelNodeData): string[] {
  const base = effectiveParamOrder(d);
  const leading = [...LAYOUT_PARAM_KEYS].filter((k) => !base.includes(k));
  return [...leading, ...base];
}

export function NumericTransformerModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultNumericTransformerModelData();
  const d = { ...defs, ...(data as Partial<NumericTransformerModelNodeData>) } as NumericTransformerModelNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(() => displayParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_NUMERIC_TRANSFORMER_MODEL_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateNumericTransformerModelSpecCode(d, order, specName),
    [d, order, specName],
  );
  const update = useCallback(
    (patch: Partial<NumericTransformerModelNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const renderField = (key: string) => {
    const full = { ...defs, ...d };
    switch (key) {
      case "contextLength":
        return (
          <ComfyIntListField
            key={key}
            label="context length (T)"
            values={intChoices(full.contextLength, 2)}
            min={1}
            title="Number of sequence positions in the numeric tensor [batch, T, D]"
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
            ariaLabel="Context length"
          />
        );
      case "inputDim":
        return (
          <ComfyIntListField
            key={key}
            label="token dim (D_in)"
            values={intChoices(full.inputDim, 1)}
            min={1}
            onCommit={(vals) => update({ inputDim: packIntList(vals) })}
            ariaLabel="Per-position input width"
          />
        );
      case "outputDim":
        return (
          <ComfyIntListField
            key={key}
            label="output token dim (D_out)"
            values={intChoices(full.outputDim, 1)}
            min={1}
            onCommit={(vals) => update({ outputDim: packIntList(vals) })}
            ariaLabel="Per-position output width"
          />
        );
      case "modelDim":
        return (
          <ComfyIntListField
            key={key}
            label="model dim"
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
            ariaLabel="Encoder feed-forward activation"
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
            ariaLabel="Encoder implementation"
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
            ariaLabel="Encoder dropout"
          />
        );
      case "spectralNormLinears":
        return (
          <DiscreteMultiSelect<NumericTransformerCausalId>
            key={key}
            label="spectral norm (encoder Linear)"
            options={YES_NO_OPTIONS}
            singleSelect
            value={full.spectralNormLinears}
            onCommit={(spectralNormLinears) => update({ spectralNormLinears })}
            ariaLabel="Spectral norm on encoder linear maps"
          />
        );
      case "stableQkNorm":
        return (
          <DiscreteMultiSelect<NumericTransformerCausalId>
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
      case "causalAttention":
        return (
          <DiscreteMultiSelect<NumericTransformerCausalId>
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
            title="PyTorch RNG seed for transformer weight initialization"
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
      className={`cr-node cr-node--numeric-transformer-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, "Transformer (numerics)")}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "Transformer (numerics)")}
              infoText="Sequence model on tensors [batch, T, D_in] -> [batch, T, D_out]. With causal attention, position i only attends to tokens 0...i (good for next-step targets). Flat MLP trainers reshape to [batch, T*D]."
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
