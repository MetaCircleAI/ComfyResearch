import { useEffect, useState, type KeyboardEvent } from "react";
import { useReactFlow, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntListField } from "./comfyMultiFields";
import { intChoices, packIntList } from "./multiValueUtils";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

function PredicateField({ label, value, onCommit, placeholder }: { label: string; value: string; onCommit: (value: string) => void; placeholder: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => onCommit(text.trim());
  const blurOnEnter = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") event.currentTarget.blur();
  };
  return <label className="cr-comfy-field"><span className="cr-comfy-widget__label">{label}</span><input className="cr-input cr-comfy-widget__control nodrag nopan" value={text} placeholder={placeholder} spellCheck={false} aria-label={label} onChange={(event) => setText(event.target.value)} onBlur={commit} onKeyDown={blurOnEnter} /></label>;
}

export function AttentionRelationScoreObservableNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as Record<string, unknown>;
  const { setNodes } = useReactFlow();
  const patch = (key: string, value: unknown) => setNodes((nodes) => nodes.map((node) => node.id === id ? { ...node, data: { ...(node.data as object), [key]: value } } : node));
  return <div className={`cr-node cr-node--generic-observable${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableNodeHeader id={id} graphNodeType="observable_attention_relation_score" title={readInstanceTitle(data, "Attention relation score")} />
    <div className="cr-node__body cr-node__body--compact">
      <ObservableSourceStrip />
      <PredicateField label="Key relation" value={String(raw.keyRelation ?? "pos(k) == pos(q) - 1")} onCommit={(value) => patch("keyRelation", value)} placeholder="pos(k) == pos(q) - 1" />
      <PredicateField label="Query filter" value={String(raw.queryFilter ?? "")} onCommit={(value) => patch("queryFilter", value)} placeholder="Blank selects all queries" />
      <ComfyIntListField label="Layer" values={intChoices(raw.layerIndex, 0)} min={0} onCommit={(values) => patch("layerIndex", packIntList(values))} ariaLabel="Attention relation layer" />
      <ComfyIntListField label="Head" values={intChoices(raw.headIndex, 0)} min={0} onCommit={(values) => patch("headIndex", packIntList(values))} ariaLabel="Attention relation head" />
      <DiscreteMultiSelect label="Key reduction" options={["mean", "max", "sum"].map((id) => ({ id, label: id }))} value={String(raw.keyReduction ?? "mean")} singleSelect onCommit={(value) => patch("keyReduction", Array.isArray(value) ? value[0] : value)} ariaLabel="Attention relation key reduction" />
      <p className="cr-observable-hint">Layer and Head lists pair by position: <code>1, 0</code> with <code>0, 1</code> plots <code>(1, 0)</code> and <code>(0, 1)</code>; a single value broadcasts. Use <code>pos(q)</code>, <code>pos(k)</code>, <code>tok(q)</code>, <code>tok(k - 1)</code>, and <code>seq_len()</code>. Example: previous-token <code>pos(k) == pos(q) - 1</code>.</p>
    </div>
  </div>;
}
