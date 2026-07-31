import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { NodeSpecCodeFooter, NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { intChoices, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  defaultSmallInceptionCifarModelData,
  defaultVgg11CifarModelData,
  type Vgg11CifarModelNodeData,
} from "./cifarModelDefaults";

function replaceNodeData(
  id: string,
  data: Vgg11CifarModelNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function Vgg11CifarModelNode({ id, data, selected, type }: NodeProps) {
  const isSmallInception = type === "small_inception_cifar_model";
  const defaults = isSmallInception ? defaultSmallInceptionCifarModelData() : defaultVgg11CifarModelData();
  const d = { ...defaults, ...(data as Partial<Vgg11CifarModelNodeData>) };
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<Vgg11CifarModelNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );
  const infoTitle = isSmallInception ? "Small Inception (CIFAR)" : "VGG-11 (CIFAR)";
  const generatedCode = useMemo(
    () => isSmallInception
      ? `# ${infoTitle}\n# Zhang et al. Small Inception; accepts 32x32 CIFAR and the paper's 28x28 preprocessing.`
      : `# ${infoTitle}\n# VGG-11 style stack for 32×32 RGB inputs.`,
    [infoTitle, isSmallInception],
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
              infoText={isSmallInception
                ? "Small Inception for Zhang et al.'s CIFAR experiments. The default CIFAR input is 32×32; select paper whitening preprocessing when reproducing its 28×28 protocol."
                : "VGG-11 style classifier for CIFAR-10. Pair with cifar10_dataset and cross_entropy_loss."}
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
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
