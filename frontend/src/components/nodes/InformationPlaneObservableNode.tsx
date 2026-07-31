import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

type Binning = "uniform_intervals" | "idnns_equal_points" | "adaptive_minmax" | "saxe_fixed_width_0_07";
type OutputMapping = "tanh" | "probability" | "signed_probability";
type Data = {
  bins?: number;
  maxSamples?: number;
  includeOutput?: boolean;
  binning?: Binning;
  outputMapping?: OutputMapping;
};
const BINNING_OPTIONS: { id: Binning; label: string }[] = [
  { id: "uniform_intervals", label: "uniform intervals" },
  { id: "idnns_equal_points", label: "IDNNs equal points" },
  { id: "adaptive_minmax", label: "adaptive min/max" },
  { id: "saxe_fixed_width_0_07", label: "Saxe fixed width 0.07" },
];
const OUTPUT_MAPPING_OPTIONS: { id: OutputMapping; label: string }[] = [
  { id: "tanh", label: "tanh" },
  { id: "probability", label: "probability" },
  { id: "signed_probability", label: "signed probability" },
];
const number = (value: unknown, fallback: number) => {
  const parsed = Number(Array.isArray(value) ? value[0] : value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export function InformationPlaneObservableNode({ id, data, selected }: NodeProps) {
  const current = {
    bins: 30,
    maxSamples: 512,
    includeOutput: true,
    binning: "uniform_intervals" as Binning,
    outputMapping: "tanh" as OutputMapping,
    ...(data as Data),
  };
  const { setNodes } = useReactFlow();
  const update = useCallback((patch: Data) => {
    setNodes((nodes: Node[]) => nodes.map((node) => node.id === id ? { ...node, data: { ...current, ...patch } } : node));
  }, [current, id, setNodes]);
  return <div className={`cr-node cr-node--observable-information-plane${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableNodeHeader id={id} graphNodeType="observable_information_plane" title={readInstanceTitle(data, "Information plane")} />
    <div className="cr-node__body cr-node__body--compact"><ObservableSourceStrip />
      <ComfyIntField label="bins" value={Math.max(2, Math.min(256, Math.floor(number(current.bins, 30))))} min={2} max={256} onCommit={(value) => update({ bins: Math.max(2, Math.min(256, value)) })} ariaLabel="Information plane bins" />
      <ComfyIntField label="max samples" value={Math.max(16, Math.min(4096, Math.floor(number(current.maxSamples, 512))))} min={16} max={4096} onCommit={(value) => update({ maxSamples: Math.max(16, Math.min(4096, value)) })} ariaLabel="Information plane maximum samples" />
      <DiscreteMultiSelect<Binning>
        label="binning"
        options={BINNING_OPTIONS}
        value={current.binning}
        singleSelect
        onCommit={(value) => update({
          binning: (Array.isArray(value) ? value[0] : value) ?? current.binning,
        })}
        ariaLabel="Information plane binning strategy"
      />
      <DiscreteMultiSelect<OutputMapping>
        label="output mapping"
        options={OUTPUT_MAPPING_OPTIONS}
        value={current.outputMapping}
        singleSelect
        onCommit={(value) => update({
          outputMapping: (Array.isArray(value) ? value[0] : value) ?? current.outputMapping,
        })}
        ariaLabel="Information plane output mapping"
      />
      <label className="cr-field"><input type="checkbox" checked={current.includeOutput !== false} onChange={(event) => update({ includeOutput: event.target.checked })} /> include output</label>
      <p className="cr-observable-hint">Samples are chosen deterministically. Each log point copies bounded activations to CPU; lower max samples or increase Trainer log frequency to reduce cost.</p>
    </div>
  </div>;
}
