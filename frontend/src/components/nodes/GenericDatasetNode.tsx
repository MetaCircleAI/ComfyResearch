import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { intChoices, packFloatList, packIntList } from "./multiValueUtils";
import type { DatasetNodeInfoKind } from "./datasetNodeInfoContent";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

/** NodeDef-channel 通用 dataset 渲染器。
 *
 * 由 generatedNodeSpecs 的 fields(含 UI 元数据)驱动。渲染规则:
 * enum(有 options)→ DiscreteMultiSelect(默认可多选，用作 enum sweep);
 * int/float → ComfyInt/FloatListField 多值输入(sweep 框;min/max 仅约束,绝不变成 select);
 * boolean → checkbox。无 options 的 enum 不渲染(Generic 通道通则;要渲染的走 custom adapter)。
 * schema truth 全在 Python def;本组件零 schema facts。
 */
export function GenericDatasetNode({ id, type, data, selected }: NodeProps) {
  const spec = GENERATED_NODE_SPECS[type];
  const { setNodes } = useReactFlow();
  const d = { ...(spec?.defaults ?? {}), ...(data as Record<string, unknown>) };
  const patch = useCallback(
    (key: string, value: unknown) => {
      setNodes((nodes: Node[]) =>
        nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, [key]: value } } : n)),
      );
    },
    [id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType={type as DatasetNodeInfoKind}
        nodeId={id}
        graphNodeType={type}
        fallbackTitle={spec?.label}
        fallbackMarkdown={spec?.hint}
      >
        {readInstanceTitle(d, spec?.label ?? type)}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <DatasetSourceSockets />
        {(spec?.fields ?? []).map((f) => {
          if (f.kind === "int") {
            return (
              <ComfyIntListField
                key={f.key}
                label={f.label}
                ariaLabel={f.label}
                values={intChoices(d[f.key], Number(f.defaultValue))}
                min={(f as { min?: number }).min}
                max={(f as { max?: number }).max}
                onCommit={(vals) => patch(f.key, packIntList(vals))}
              />
            );
          }
          if (f.kind === "float") {
            const raw = d[f.key];
            const values = Array.isArray(raw) ? (raw as number[]) : [Number(raw ?? f.defaultValue)];
            return (
              <ComfyFloatListField
                key={f.key}
                label={f.label}
                ariaLabel={f.label}
                values={values}
                min={(f as { min?: number }).min}
                max={(f as { max?: number }).max}
                onCommit={(vals) => patch(f.key, packFloatList(vals))}
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
          // 无 options 的 enum 不渲染;有 options → DiscreteMultiSelect(可多选 = sweep)。
          if (f.kind === "enum" && "options" in f && f.options.length) {
            const value = d[f.key] ?? f.defaultValue;
            return (
              <DiscreteMultiSelect
                key={f.key}
                label={f.label}
                options={f.options.map((o) => ({ id: o, label: o }))}
                value={Array.isArray(value) ? (value as string[]) : String(value)}
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
