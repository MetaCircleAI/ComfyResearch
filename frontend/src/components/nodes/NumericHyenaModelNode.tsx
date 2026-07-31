import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import {
  defaultNumericHyenaModelData,
  type NumericHyenaModelNodeData,
} from "./numericHyenaModelDefaults";
import {
  DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER,
  DEFAULT_NUMERIC_HYENA_MODEL_SPEC_NAME,
  generateNumericHyenaModelSpecCode,
} from "../../graph/specCode/numericHyenaModelSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

function replaceNodeData(
  id: string,
  data: NumericHyenaModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: NumericHyenaModelNodeData,
  patch: Partial<NumericHyenaModelNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

export function NumericHyenaModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultNumericHyenaModelData();
  const d = { ...defs, ...(data as Partial<NumericHyenaModelNodeData>) } as NumericHyenaModelNodeData;
  const { setNodes } = useReactFlow();
  const order = useMemo(
    () => (d.paramOrder?.length ? d.paramOrder : [...DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER]),
    [d.paramOrder],
  );
  const specName = d.specCodeName ?? DEFAULT_NUMERIC_HYENA_MODEL_SPEC_NAME;
  const generatedCode = useMemo(() => generateNumericHyenaModelSpecCode(d, order, specName), [d, order, specName]);
  const update = useCallback(
    (patch: Partial<NumericHyenaModelNodeData>) => patchData(id, d, patch, setNodes),
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
            values={intChoices(full.contextLength, 8)}
            min={1}
            title="Number of sequence positions in [batch, T, D]."
            onCommit={(vals) => update({ contextLength: packIntList(vals) })}
            ariaLabel="Context length"
          />
        );
      case "inputDim":
        return (
          <ComfyIntListField
            key={key}
            label="token dim (D_in)"
            values={intChoices(full.inputDim, 2)}
            min={1}
            onCommit={(vals) => update({ inputDim: packIntList(vals) })}
            ariaLabel="Input token dimension"
          />
        );
      case "outputDim":
        return (
          <ComfyIntListField
            key={key}
            label="output token dim (D_out)"
            values={intChoices(full.outputDim, 2)}
            min={1}
            onCommit={(vals) => update({ outputDim: packIntList(vals) })}
            ariaLabel="Output token dimension"
          />
        );
      case "modelDim":
        return (
          <ComfyIntListField
            key={key}
            label="model dim"
            values={intChoices(full.modelDim, 64)}
            min={1}
            onCommit={(vals) => update({ modelDim: packIntList(vals) })}
            ariaLabel="Model hidden dimension"
          />
        );
      case "depth":
        return (
          <ComfyIntListField
            key={key}
            label="depth"
            values={intChoices(full.depth, 2)}
            min={1}
            onCommit={(vals) => update({ depth: packIntList(vals) })}
            ariaLabel="Hyena block depth"
          />
        );
      case "convKernel":
        return (
          <ComfyIntListField
            key={key}
            label="conv kernel (odd)"
            values={intChoices(full.convKernel, 7)}
            min={3}
            onCommit={(vals) => update({ convKernel: packIntList(vals) })}
            ariaLabel="Convolution kernel size"
          />
        );
      case "ffMult":
        return (
          <ComfyIntListField
            key={key}
            label="FF mult"
            values={intChoices(full.ffMult, 2)}
            min={1}
            onCommit={(vals) => update({ ffMult: packIntList(vals) })}
            ariaLabel="Feed-forward multiplier"
          />
        );
      case "localMixingKernel":
        return (
          <ComfyIntListField
            key={key}
            label="local mixing kernel"
            values={intChoices(full.localMixingKernel, 0)}
            min={0}
            title="Causal depthwise conv on the projected token stream (0-2 disables; odd >=3 enables)."
            onCommit={(vals) => update({ localMixingKernel: packIntList(vals) })}
            ariaLabel="Local mixing kernel"
          />
        );
      case "seed":
        return (
          <ComfyIntListField
            key={key}
            label="init seed"
            values={intChoices(full.seed, 0)}
            min={0}
            title="PyTorch RNG seed for weight initialization."
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
      className={`cr-node cr-node--numeric-hyena-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, "Hyena-like LM (numerics)")}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, "Hyena-like LM (numerics)")}
              infoText="Causal Hyena-like sequence mixer for numeric tensors [batch, T, D_in] -> [batch, T, D_out]. Useful for next-step numeric trajectories without self-attention."
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
