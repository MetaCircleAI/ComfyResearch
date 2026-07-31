import { useReactFlow, type NodeProps } from "@xyflow/react";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

/** Renderer for NodeDef-channel observables without a custom TSX. */
export function GenericObservableNode({ id, data, selected, type }: NodeProps) {
  const spec = GENERATED_NODE_SPECS[String(type)];
  const d = { ...(spec?.defaults ?? {}), ...((data ?? {}) as Record<string, unknown>) };
  const { setNodes } = useReactFlow();
  const patch = (key: string, value: unknown) =>
    setNodes((nds) =>
      nds.map((node) => (node.id === id ? { ...node, data: { ...(node.data as object), [key]: value } } : node)),
    );

  return (
    <div
      className={`cr-node cr-node--generic-observable${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType={String(type)}
        title={readInstanceTitle(data, spec?.label ?? String(type))}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        {(spec?.fields ?? []).map((f) => {
          if (f.kind === "int") {
            return (
              <ComfyIntField
                key={f.key}
                label={f.label}
                value={Number(d[f.key] ?? f.defaultValue)}
                min={"min" in f && typeof f.min === "number" ? f.min : undefined}
                max={"max" in f && typeof f.max === "number" ? f.max : undefined}
                onCommit={(n) => patch(f.key, n)}
                ariaLabel={f.label}
              />
            );
          }
          if (f.kind === "float") {
            return (
              <ComfyFloatField
                key={f.key}
                label={f.label}
                value={Number(d[f.key] ?? f.defaultValue)}
                min={"min" in f && typeof f.min === "number" ? f.min : undefined}
                max={"max" in f && typeof f.max === "number" ? f.max : undefined}
                onCommit={(n) => patch(f.key, n)}
                ariaLabel={f.label}
              />
            );
          }
          if (f.kind === "boolean") {
            return (
              <label key={f.key} className="cr-observable-hint nodrag nopan">
                <input
                  type="checkbox"
                  className="nodrag nopan"
                  checked={Boolean(d[f.key] ?? f.defaultValue)}
                  onChange={(e) => patch(f.key, e.target.checked)}
                  aria-label={f.label}
                />{" "}
                {f.label}
              </label>
            );
          }
          // 无 options 的 enum 不渲染；需要此类字段的节点使用 custom adapter。
          if (f.kind === "enum" && "options" in f && f.options.length) {
            return (
              <DiscreteMultiSelect
                key={f.key}
                label={f.label}
                options={f.options.map((o) => ({ id: o, label: o }))}
                value={String(d[f.key] ?? f.defaultValue)}
                singleSelect
                onCommit={(v) => patch(f.key, v)}
                ariaLabel={f.label}
              />
            );
          }
          return null;
        })}
        {spec?.hint ? <p className="cr-observable-hint">{spec.hint}</p> : null}
      </div>
    </div>
  );
}
