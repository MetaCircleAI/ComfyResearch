import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  defaultKeskarCnnModelData,
  type KeskarArchitecture,
  type KeskarCnnModelNodeData,
} from "./keskarModelDefaults";

const ARCH_OPTS: { id: KeskarArchitecture; label: string }[] = [
  { id: "c1", label: "C1 (3 conv, shallow)" },
  { id: "c2", label: "C2 (6 conv, deeper)" },
];

function replaceNodeData(
  id: string,
  data: KeskarCnnModelNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function KeskarCnnModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultKeskarCnnModelData();
  const d = { ...defs, ...(data as Partial<KeskarCnnModelNodeData>) } as KeskarCnnModelNodeData;
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<KeskarCnnModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );
  const infoTitle = "Keskar C1/C2 CNN";
  const arch = (Array.isArray(d.architecture) ? d.architecture[0] : d.architecture) ?? "c1";
  const generatedCode = useMemo(
    () =>
      `# ${infoTitle}\n# architecture=${arch}\n# Wire cifar10_dataset + cross_entropy_loss for Keskar batch-sharpness repro.`,
    [arch, infoTitle],
  );

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
              infoText="Keskar et al. C1/C2 CNN for 32×32 CIFAR-10. Use with cifar10_dataset and cross_entropy_loss."
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
        <DiscreteMultiSelect<KeskarArchitecture>
          label="architecture"
          options={ARCH_OPTS}
          value={arch}
          onCommit={(v) => update({ architecture: (typeof v === "string" ? v : v[0] ?? "c1") as KeskarArchitecture })}
          ariaLabel="Keskar architecture variant"
          singleSelect
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
