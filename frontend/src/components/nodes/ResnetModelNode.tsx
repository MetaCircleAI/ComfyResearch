import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { pySlugForNode } from "../../graph/nodeDefinitionCode";
import { buildResnetModelNotebookPython } from "../../graph/specCode/visionModelSpecCode";
import { defaultResnetModelData, type ResnetModelNodeData, type ResnetVariant } from "./resnetModelDefaults";

const VARIANT_OPTS: { id: ResnetVariant; label: string }[] = [
  { id: "resnet18", label: "ResNet-18 style" },
  { id: "resnet34", label: "ResNet-34 style" },
  { id: "self_defined", label: "Self-defined" },
];

function replaceNodeData(
  id: string,
  data: ResnetModelNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function ResnetModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultResnetModelData();
  const d = { ...defs, ...(data as Partial<ResnetModelNodeData>) } as ResnetModelNodeData;
  const { setNodes, getNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<ResnetModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );
  const infoTitle = "ResNet (vision)";
  const generatedCode = useMemo(() => {
    const slug = pySlugForNode(id, getNodes());
    const cellTitle = readInstanceTitle(d as Record<string, unknown>, infoTitle);
    return buildResnetModelNotebookPython(slug, cellTitle, d as unknown as Record<string, unknown>);
  }, [d, getNodes, id, infoTitle]);

  const variant = (Array.isArray(d.variant) ? d.variant[0] : d.variant) ?? "resnet18";
  const showCustom = variant === "self_defined";

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
              infoText="Convolutional ResNet-style classifier on 1×H×W inputs. Wire a vision dataset (MNIST, Gaussian blob, shape world, or hole counting) and cross_entropy_loss."
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        <DiscreteMultiSelect<ResnetVariant>
          label="variant"
          options={VARIANT_OPTS}
          value={variant}
          onCommit={(v) => {
            const one = (typeof v === "string" ? v : v[0] ?? "resnet18") as ResnetVariant;
            update({ variant: one });
          }}
          ariaLabel="ResNet variant"
          singleSelect
        />
        {showCustom ? (
          <>
            <ComfyIntListField
              label="base channels"
              values={intChoices(d.baseChannels, 32)}
              min={8}
              max={256}
              ariaLabel="Base channel width"
              onCommit={(vals) => update({ baseChannels: packIntList(vals) })}
            />
            <ComfyIntListField
              label="blocks stage 1"
              values={intChoices(d.blocksStage1, 2)}
              min={1}
              max={16}
              ariaLabel="Residual blocks in stage 1"
              onCommit={(vals) => update({ blocksStage1: packIntList(vals) })}
            />
            <ComfyIntListField
              label="blocks stage 2"
              values={intChoices(d.blocksStage2, 2)}
              min={1}
              max={16}
              ariaLabel="Residual blocks in stage 2"
              onCommit={(vals) => update({ blocksStage2: packIntList(vals) })}
            />
            <ComfyIntListField
              label="blocks stage 3"
              values={intChoices(d.blocksStage3, 2)}
              min={1}
              max={16}
              ariaLabel="Residual blocks in stage 3"
              onCommit={(vals) => update({ blocksStage3: packIntList(vals) })}
            />
            <ComfyIntListField
              label="blocks stage 4"
              values={intChoices(d.blocksStage4, 2)}
              min={1}
              max={16}
              ariaLabel="Residual blocks in stage 4"
              onCommit={(vals) => update({ blocksStage4: packIntList(vals) })}
            />
            <ComfyIntListField
              label="kernel size"
              values={intChoices(d.kernelSize, 3)}
              min={3}
              max={11}
              title="Odd size 3–11; even values are rounded up on the server."
              ariaLabel="Convolution kernel size (odd)"
              onCommit={(vals) => update({ kernelSize: packIntList(vals) })}
            />
          </>
        ) : null}
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
