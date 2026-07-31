import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER,
  DEFAULT_MLP_TOKEN_MODEL_SPEC_NAME,
  generateMlpTokenModelVariantSpecCode,
} from "../../graph/specCode/mlpTokenModelSpecCode";
import { MLP_ACTIVATION_OPTIONS } from "./mlpModelDefaults";
import { defaultMlpTokenModelData, type MlpTokenModelNodeData } from "./mlpTokenModelDefaults";

const TIE_WEIGHT_OPTIONS = [
  { id: "yes", label: "yes (embedding = unembedding)" },
  { id: "no", label: "no" },
] as const;

function replaceNodeData(
  id: string,
  data: MlpTokenModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function MlpTokenModelNode({ id, data, selected, type }: NodeProps) {
  const defs = defaultMlpTokenModelData();
  const d = { ...defs, ...(data as Partial<MlpTokenModelNodeData>) } as MlpTokenModelNodeData;
  const { setNodes } = useReactFlow();
  const nodeType = String(type ?? "");
  const variant: "plain" | "gated" | "moe" =
    nodeType === "gated_mlp_token_model" ? "gated" : nodeType === "moe_mlp_token_model" ? "moe" : "plain";
  const order = (d.paramOrder?.length ? d.paramOrder : DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER).filter((k) =>
    variant === "moe" ? true : k !== "numExperts",
  );
  const specName = d.specCodeName ?? DEFAULT_MLP_TOKEN_MODEL_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateMlpTokenModelVariantSpecCode(d, order, specName, variant),
    [d, order, specName, variant],
  );
  const update = useCallback(
    (patch: Partial<MlpTokenModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  const infoTitle =
    variant === "gated" ? "Gated MLP_token model" : variant === "moe" ? "MoE MLP_token model" : "MLP_token model";
  const infoText =
    variant === "gated"
      ? "Token ids -> embeddings -> gated hidden MLP blocks using act(Wg h) * (Wv h) -> logits."
      : variant === "moe"
        ? "Token ids -> embeddings -> softmax gate over expert MLPs -> weighted expert mixture -> logits."
        : "Token ids -> embeddings -> hidden MLP (depth x width, activation) -> logits. Tie weights shares the embedding matrix with the final linear when shapes match.";

  return (
    <div
      className={`cr-node cr-node--mlp-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d, infoTitle)}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d, infoTitle)}
              infoText={infoText}
            />
          </div>
        </div>
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
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
          values={intChoices(d.embedDim, 64)}
          min={1}
          onCommit={(vals) => update({ embedDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="tokens per input"
          values={intChoices(d.tokensPerInput, 1)}
          min={1}
          title="How many token ids are consumed per sample before projection to logits."
          onCommit={(vals) => update({ tokensPerInput: packIntList(vals) })}
        />
        <ComfyIntListField
          label="depth"
          values={intChoices(d.depth, 2)}
          min={1}
          title="Number of hidden layers in the token MLP (same convention as MLP model)."
          onCommit={(vals) => update({ depth: packIntList(vals) })}
          ariaLabel="Hidden depth (number of hidden layers)"
        />
        <ComfyIntListField
          label="width"
          values={intChoices(d.width, 64)}
          min={1}
          title="Width of each hidden layer."
          onCommit={(vals) => update({ width: packIntList(vals) })}
          ariaLabel="Hidden layer width"
        />
        {variant === "moe" ? (
          <ComfyIntListField
            label="num experts"
            values={intChoices(d.numExperts ?? 4, 4)}
            min={1}
            title="Number of experts mixed by softmax gating."
            onCommit={(vals) => update({ numExperts: packIntList(vals) })}
            ariaLabel="Number of experts"
          />
        ) : null}
        <DiscreteMultiSelect
          label="activation"
          options={MLP_ACTIVATION_OPTIONS}
          value={d.activation}
          onCommit={(activation) => update({ activation })}
          ariaLabel="Activation function"
        />
        <DiscreteMultiSelect
          label="tie weights"
          options={TIE_WEIGHT_OPTIONS as unknown as { id: string; label: string }[]}
          singleSelect
          value={d.tieWeights}
          onCommit={(value) => update({ tieWeights: value as "yes" | "no" })}
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
