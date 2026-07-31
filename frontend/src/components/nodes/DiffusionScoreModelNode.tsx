import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { buildDiffusionScoreNotebookPython } from "../../graph/specCode/diffusionScoreNotebookSpecCode";
import { defaultDiffusionScoreModelData, type DiffusionScoreModelNodeData } from "./diffusionScoreModelDefaults";

function replaceNodeData(
  id: string,
  data: DiffusionScoreModelNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function DiffusionScoreModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultDiffusionScoreModelData();
  const d = { ...defs, ...(data as Partial<DiffusionScoreModelNodeData>) } as DiffusionScoreModelNodeData;
  const { setNodes, getNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<DiffusionScoreModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );
  const infoTitle = "Diffusion score MLP";
  const generatedCode = useMemo(() => {
    const slug = pySlugForNode(id, getNodes());
    const cellTitle = readInstanceTitle(d as Record<string, unknown>, infoTitle);
    return buildDiffusionScoreNotebookPython(slug, cellTitle, d as unknown as Record<string, unknown>);
  }, [d, getNodes, id, infoTitle]);
  return (
    <div
      className={`cr-node cr-node--mlp-model${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--title-actions">
          <div className="cr-node__header-title">{readInstanceTitle(d as Record<string, unknown>, infoTitle)}</div>
          <div className="cr-node__header-actions">
            <NodeSpecHeaderActions
              nodeId={id}
              generatedCode={generatedCode}
              infoTitle={readInstanceTitle(d as Record<string, unknown>, infoTitle)}
              infoText="Predicts Gaussian noise given noisy data x_t and discrete timestep. Pair with diffusion_mse_loss."
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        <ComfyIntListField
          label="data dim (input)"
          values={intChoices(d.inputDim, 8)}
          min={1}
          onCommit={(vals) => update({ inputDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="hidden dim"
          values={intChoices(d.hiddenDim, 128)}
          min={8}
          onCommit={(vals) => update({ hiddenDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="depth"
          values={intChoices(d.depth, 3)}
          min={1}
          onCommit={(vals) => update({ depth: packIntList(vals) })}
        />
        <ComfyIntListField
          label="time embed dim"
          values={intChoices(d.timeEmbedDim, 64)}
          min={8}
          onCommit={(vals) => update({ timeEmbedDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="diffusion timesteps T"
          values={intChoices(d.diffusionTimesteps, 100)}
          min={2}
          onCommit={(vals) => update({ diffusionTimesteps: packIntList(vals) })}
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
