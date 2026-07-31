import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultCrlEnvConfigData, type CrlEnvConfigNodeData, type CrlEnvPreset } from "./crlEnvDefaults";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";

function patchData(
  id: string,
  prev: CrlEnvConfigNodeData,
  patch: Partial<CrlEnvConfigNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

const PRESETS: { id: CrlEnvPreset; label: string }[] = [
  { id: "point_u4_maze", label: "Point U4 (CPU, scaling-crl layout)" },
  { id: "ant_u4_maze", label: "Ant U4 (same topology; full Ant = Brax/JAX)" },
];

export function CrlEnvConfigNode({ id, data, selected }: NodeProps) {
  const defs = defaultCrlEnvConfigData();
  const d = { ...defs, ...(data as Partial<CrlEnvConfigNodeData>) } as CrlEnvConfigNodeData;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<CrlEnvConfigNodeData>) => patchData(id, d, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--crl-env${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <div className="cr-node__header cr-node__header--dataset-info-row">
        <div className="cr-node__header-main">{readInstanceTitle(d as Record<string, unknown>, "CRL env")}</div>
        <NodeSpecHeaderActions
          nodeId={id}
          graphNodeType="crl_env_config"
          generatedCode={`# CRL environment — training runs on the server (see comfy_research/engine/crl_run.py).\n`}
          codeKind="dataset"
          infoTitle="CRL environment"
          infoText="U4 maze topology from wang-kevin3290/scaling-crl. The point preset is a fast PyTorch-friendly particle; full Ant uses Brax in the reference repo."
        />
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="CRL env output">
          <div className="cr-trainer-io-row cr-trainer-io-row--source-out">
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">env</span>
              <Handle
                type="source"
                position={Position.Right}
                id="env"
                className="cr-handle-source cr-handle-source--trainer-row"
              />
            </div>
          </div>
        </div>

        <DiscreteMultiSelect
          label="preset"
          options={PRESETS}
          value={d.preset}
          singleSelect
          onCommit={(preset) => update({ preset: preset as CrlEnvPreset })}
          ariaLabel="CRL environment preset"
        />

        <ComfyIntField label="parallel envs" value={d.numEnvs} min={1} ariaLabel="Parallel envs" onCommit={(numEnvs) => update({ numEnvs })} />
        <ComfyIntField
          label="episode length"
          value={d.episodeLength}
          min={16}
          ariaLabel="Episode length"
          onCommit={(episodeLength) => update({ episodeLength })}
        />
        <ComfyIntField label="seed" value={d.seed} min={0} ariaLabel="Env seed" onCommit={(seed) => update({ seed })} />
        <ComfyFloatField
          label="maze scale"
          value={d.mazeSizeScaling}
          min={0.5}
          ariaLabel="Maze cell scale"
          onCommit={(mazeSizeScaling) => update({ mazeSizeScaling })}
        />
      </div>
    </div>
  );
}
