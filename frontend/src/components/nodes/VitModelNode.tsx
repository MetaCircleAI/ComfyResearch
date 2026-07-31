import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { buildVitModelNotebookPython } from "../../graph/specCode/visionModelSpecCode";
import { defaultVitModelData, type VitModelNodeData, type VitVariant } from "./vitModelDefaults";

const VARIANT_OPTS: { id: VitVariant; label: string }[] = [
  { id: "tiny", label: "Tiny ViT" },
  { id: "small", label: "Small ViT" },
];

function replaceNodeData(
  id: string,
  data: VitModelNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function VitModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultVitModelData();
  const d = { ...defs, ...(data as Partial<VitModelNodeData>) } as VitModelNodeData;
  const { setNodes, getNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<VitModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );
  const infoTitle = "ViT (vision)";
  const generatedCode = useMemo(() => {
    const slug = pySlugForNode(id, getNodes());
    const cellTitle = readInstanceTitle(d as Record<string, unknown>, infoTitle);
    return buildVitModelNotebookPython(slug, cellTitle, d as unknown as Record<string, unknown>);
  }, [d, getNodes, id, infoTitle]);

  const variant = (Array.isArray(d.variant) ? d.variant[0] : d.variant) ?? "tiny";

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
              infoText="Patch transformer for low-res 1-channel images. imageSize from the dataset must divide patchSize."
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        <DiscreteMultiSelect<VitVariant>
          label="variant"
          options={VARIANT_OPTS}
          value={variant}
          onCommit={(v) => {
            const one = (typeof v === "string" ? v : v[0] ?? "tiny") as VitVariant;
            update({ variant: one });
          }}
          ariaLabel="ViT variant"
          singleSelect
        />
        <ComfyIntListField
          label="patch size"
          values={intChoices(d.patchSize, 4)}
          min={2}
          max={16}
          ariaLabel="Patch size"
          onCommit={(vals) => update({ patchSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="hidden dim"
          values={intChoices(d.hiddenDim, 128)}
          min={32}
          max={512}
          ariaLabel="Hidden dim"
          onCommit={(vals) => update({ hiddenDim: packIntList(vals) })}
        />
        <ComfyIntListField
          label="depth"
          values={intChoices(d.depth, 3)}
          min={1}
          max={12}
          ariaLabel="Transformer depth"
          onCommit={(vals) => update({ depth: packIntList(vals) })}
        />
        <ComfyIntListField
          label="num heads"
          values={intChoices(d.numHeads, 4)}
          min={1}
          max={16}
          ariaLabel="Attention heads"
          onCommit={(vals) => update({ numHeads: packIntList(vals) })}
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(d.seed, 0)}
          min={0}
          ariaLabel="Init seed"
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
        <NodeSpecCodeFooter nodeId={id} generatedCode={generatedCode} />
      </div>
    </div>
  );
}
