import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { defaultFourierComponentObservableData, type FourierComponentMetric, type FourierComponentObservableNodeData } from "./fourierComponentObservableDefaults";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

const OPTIONS: { id: FourierComponentMetric; label: string }[] = [
  { id: "relative_projection_mse", label: "Relative projection MSE" },
  { id: "amplitude_ratio", label: "Amplitude ratio" },
];
const METRICS = new Set<FourierComponentMetric>(OPTIONS.map((option) => option.id));
const firstNumber = (raw: unknown, fallback: number) => {
  const value = Number(Array.isArray(raw) ? raw[0] : raw);
  return Number.isFinite(value) ? value : fallback;
};

export function FourierComponentObservableNode({ id, data, selected }: NodeProps) {
  const defaults = defaultFourierComponentObservableData();
  const current = { ...defaults, ...(data as Partial<FourierComponentObservableNodeData>) };
  const { setNodes } = useReactFlow();
  const update = useCallback((patch: Partial<FourierComponentObservableNodeData>) => {
    setNodes((nodes: Node[]) => nodes.map((node) => node.id === id ? { ...node, data: { ...current, ...patch } } : node));
  }, [current, id, setNodes]);
  const metricRaw = Array.isArray(current.metric) ? current.metric[0] : current.metric;
  const metric = METRICS.has(metricRaw as FourierComponentMetric) ? metricRaw as FourierComponentMetric : defaults.metric;
  return <div className={`cr-node cr-node--observable-fourier-component${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableNodeHeader id={id} graphNodeType="observable_fourier_component" title={readInstanceTitle(data, "Fourier component")} />
    <div className="cr-node__body cr-node__body--compact"><ObservableSourceStrip />
      <ComfyFloatField label="frequency" value={Math.max(0, firstNumber(current.frequency, defaults.frequency))} min={0} onCommit={(value) => update({ frequency: Math.max(0, value) })} ariaLabel="Fourier frequency" />
      <DiscreteMultiSelect label="metric" options={OPTIONS} value={metric} onCommit={(value) => update({ metric: (Array.isArray(value) ? value[0] : value) ?? defaults.metric })} ariaLabel="Fourier component metric" singleSelect />
      <ComfyIntField label="input axis" value={Math.max(0, Math.floor(firstNumber(current.inputAxis, defaults.inputAxis)))} min={0} onCommit={(value) => update({ inputAxis: Math.max(0, value) })} ariaLabel="Input feature axis" />
      <ComfyIntField label="output index" value={Math.max(0, Math.floor(firstNumber(current.outputIndex, defaults.outputIndex)))} min={0} onCommit={(value) => update({ outputIndex: Math.max(0, value) })} ariaLabel="Output feature index" />
      <p className="cr-observable-hint">Projects target and prediction onto sin/cos at the chosen frequency. Full-batch logging gives the most stable estimate.</p>
    </div>
  </div>;
}
