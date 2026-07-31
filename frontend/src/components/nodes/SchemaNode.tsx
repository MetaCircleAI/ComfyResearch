import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import type { CSSProperties, ReactElement } from "react";
import { defaultInstanceTitleBase, readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  NODE_SPEC_REGISTRY,
  type FieldDef,
  type RegisteredNodeKind,
} from "../../graph/nodeRegistrySpec";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { OptimizerLrScheduleSocketRows } from "./OptimizerLrScheduleSocketRows";

type NodeData = Record<string, unknown>;
type PatchFn = (patch: NodeData) => void;

export function nodeTypeClass(type: string): string {
  return `cr-node--${type.replaceAll("_", "-")}`;
}

export function renderFieldControl(field: FieldDef, data: NodeData, patch: PatchFn): ReactElement {
  switch (field.kind) {
    case "floatList":
      return (
        <ComfyFloatListField
          label={field.label}
          values={floatChoices(data[field.key], field.defaultValue)}
          positiveOnly={field.positiveOnly}
          min={field.min}
          max={field.max}
          title={field.tooltip}
          ariaLabel={field.ariaLabel ?? field.label}
          onCommit={(vals) => patch({ [field.key]: packFloatList(vals) })}
        />
      );
    case "intList":
      return (
        <ComfyIntListField
          label={field.label}
          values={intChoices(data[field.key], field.defaultValue)}
          min={field.min}
          title={field.tooltip}
          ariaLabel={field.ariaLabel ?? field.label}
          onCommit={(vals) => patch({ [field.key]: packIntList(vals) })}
        />
      );
    default:
      throw new Error(`SchemaNode does not support field kind: ${field.kind}`);
  }
}

export function SchemaNode({ id, type, data, selected }: NodeProps): ReactElement {
  const spec = NODE_SPEC_REGISTRY[type as RegisteredNodeKind];
  if (!spec.specCode) {
    throw new Error(`SchemaNode requires specCode for ${type}`);
  }
  const d = { ...spec.defaults?.(), ...(data as NodeData) } as NodeData;
  const { setNodes } = useReactFlow();
  const patch: PatchFn = (p) =>
    setNodes((nodes) =>
      nodes.map((n: Node) => (n.id === id ? { ...n, data: { ...d, ...p } } : n)),
    );
  const title = readInstanceTitle(d, defaultInstanceTitleBase(type));
  const ui = spec.ui;
  const accentStyle: CSSProperties | undefined = ui?.accent
    ? ({ ["--accent" as string]: `var(--cr-accent-${ui.accent})` } as CSSProperties)
    : undefined;

  return (
    <div
      className={`cr-node ${nodeTypeClass(type)}${selected ? " cr-node--selected" : ""}`}
      style={accentStyle}
    >
      <div className="cr-node__header cr-node__header--dataset-info-row">
        <div className="cr-node__header-main">{title}</div>
        <NodeSpecHeaderActions
          nodeId={id}
          graphNodeType={type}
          generatedCode={spec.specCode(d)}
          codeKind={ui?.codeKind}
          infoTitle={ui?.info?.title}
          infoText={ui?.info?.text}
        />
      </div>
      <div className="cr-node__body">
        {ui?.socketRows === "optimizerLrSchedule" && <OptimizerLrScheduleSocketRows />}
        {spec.fields?.map((f) => (
          <div key={f.key} className="cr-node__field" data-schema-field-key={f.key}>
            {renderFieldControl(f, d, patch)}
          </div>
        ))}
      </div>
    </div>
  );
}
