import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultCrlResidualMlpData, type CrlResidualMlpNodeData } from "./crlResidualMlpDefaults";
import { MLP_ACTIVATION_OPTIONS } from "./mlpModelDefaults";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";

function patchData(
  id: string,
  prev: CrlResidualMlpNodeData,
  patch: Partial<CrlResidualMlpNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

function mergeCrlResidualMlpData(
  defs: CrlResidualMlpNodeData,
  data: unknown,
): CrlResidualMlpNodeData {
  const raw = (data ?? {}) as Partial<CrlResidualMlpNodeData> & { useReLU?: boolean };
  const base = { ...defs, ...raw } as CrlResidualMlpNodeData & { useReLU?: boolean };
  let activation = base.activation;
  if (!("activation" in raw) && typeof raw.useReLU === "boolean") {
    activation = raw.useReLU ? "relu" : "silu";
  }
  return { ...base, activation };
}

export function CrlResidualMlpNode({ id, data, selected }: NodeProps) {
  const defs = defaultCrlResidualMlpData();
  const d = mergeCrlResidualMlpData(defs, data);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<CrlResidualMlpNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--crl-residual-mlp${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header cr-node__header--dataset-info-row">
        <div className="cr-node__header-main">{readInstanceTitle(d as Record<string, unknown>, "CRL residual MLP")}</div>
        <NodeSpecHeaderActions
          nodeId={id}
          graphNodeType="crl_residual_mlp"
          generatedCode={`# CRL actor + critic encoders — server-side build in comfy_research/engine/crl_networks.py\n`}
          codeKind="model"
          infoTitle="CRL residual MLP"
          infoText="Residual blocks (4× Dense+LayerNorm+Swish) per scaling-crl train.py; depths must be multiples of 4."
        />
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="CRL model output">
          <div className="cr-trainer-io-row cr-trainer-io-row--source-out">
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">model</span>
              <Handle
                type="source"
                position={Position.Right}
                id="model"
                className="cr-handle-source cr-handle-source--trainer-row"
              />
            </div>
          </div>
        </div>

        <ComfyIntField label="state dim" value={d.stateDim} min={1} ariaLabel="State dim" onCommit={(stateDim) => update({ stateDim })} />
        <ComfyIntField label="action dim" value={d.actionDim} min={1} ariaLabel="Action dim" onCommit={(actionDim) => update({ actionDim })} />
        <ComfyIntField label="goal dim" value={d.goalDim} min={1} ariaLabel="Goal dim" onCommit={(goalDim) => update({ goalDim })} />
        <ComfyIntField label="actor width" value={d.actorWidth} min={8} ariaLabel="Actor width" onCommit={(actorWidth) => update({ actorWidth })} />
        <ComfyIntField label="critic width" value={d.criticWidth} min={8} ariaLabel="Critic width" onCommit={(criticWidth) => update({ criticWidth })} />
        <ComfyIntField
          label="actor depth"
          value={d.actorDepth}
          min={4}
          title="Total Dense layers; must be a multiple of 4"
          ariaLabel="Actor depth"
          onCommit={(actorDepth) => update({ actorDepth })}
        />
        <ComfyIntField
          label="critic depth"
          value={d.criticDepth}
          min={4}
          title="Total Dense layers; must be a multiple of 4"
          ariaLabel="Critic depth"
          onCommit={(criticDepth) => update({ criticDepth })}
        />
        <ComfyIntField label="embed dim" value={d.embedDim} min={8} ariaLabel="Embed dim" onCommit={(embedDim) => update({ embedDim })} />
        <DiscreteMultiSelect
          label="activation"
          options={MLP_ACTIVATION_OPTIONS}
          value={d.activation}
          singleSelect
          onCommit={(activation) => update({ activation })}
          ariaLabel="CRL residual MLP activation"
        />
        <ComfyIntField label="seed" value={d.seed} min={0} ariaLabel="Random seed" onCommit={(seed) => update({ seed })} />
      </div>
    </div>
  );
}
