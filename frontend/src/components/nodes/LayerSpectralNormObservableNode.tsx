import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

type Estimator = "singular_value" | "author_figure1";
type StartVector = "deterministic" | "seeded_gaussian";
type Data = {
  estimator?: Estimator;
  powerIterations?: number;
  startVector?: StartVector;
  seed?: number;
};
const OPTIONS: { id: Estimator; label: string }[] = [
  { id: "singular_value", label: "Singular value (standard)" },
  { id: "author_figure1", label: "Rahaman Figure 1 notebook" },
];
const START_VECTOR_OPTIONS: { id: StartVector; label: string }[] = [
  { id: "deterministic", label: "Deterministic local vector" },
  { id: "seeded_gaussian", label: "Seeded Gaussian (paper)" },
];

export function LayerSpectralNormObservableNode({ id, data, selected }: NodeProps) {
  const current = {
    estimator: "singular_value" as Estimator,
    powerIterations: 10,
    startVector: "deterministic" as StartVector,
    seed: 0,
    ...(data as Data),
  };
  const estimator = current.estimator === "author_figure1" ? "author_figure1" : "singular_value";
  const iterations = Math.max(1, Math.floor(Number(current.powerIterations) || 10));
  const { setNodes } = useReactFlow();
  const update = useCallback((patch: Partial<Data>) => setNodes((nodes: Node[]) => nodes.map((node) => node.id === id ? { ...node, data: { ...current, ...patch } } : node)), [current, id, setNodes]);
  return <div className={`cr-node cr-node--observable-layer-spectral-norm${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableNodeHeader id={id} graphNodeType="observable_layer_spectral_norm" title={readInstanceTitle(data, "Layer spectral norm")} />
    <div className="cr-node__body cr-node__body--compact"><ObservableSourceStrip />
      <DiscreteMultiSelect label="estimator" options={OPTIONS} value={estimator} onCommit={(value) => update({ estimator: value === "author_figure1" ? value : "singular_value" })} ariaLabel="Spectral norm estimator" singleSelect />
      <ComfyIntField label="power iterations" value={iterations} min={1} max={128} onCommit={(value) => update({ powerIterations: Math.max(1, Math.min(128, value)) })} ariaLabel="Spectral norm power iterations" />
      {estimator === "author_figure1" ? (
        <>
          <DiscreteMultiSelect
            label="start vector"
            options={START_VECTOR_OPTIONS}
            value={current.startVector ?? "deterministic"}
            onCommit={(value) => update({
              startVector: value === "seeded_gaussian" ? value : "deterministic",
            })}
            ariaLabel="Spectral norm start vector"
            singleSelect
          />
          {current.startVector === "seeded_gaussian" ? (
            <ComfyIntField
              label="seed"
              value={Math.max(0, Math.floor(Number(current.seed) || 0))}
              min={0}
              onCommit={(value) => update({ seed: Math.max(0, value) })}
              ariaLabel="Spectral norm start-vector seed"
            />
          ) : null}
        </>
      ) : null}
      <p className="cr-observable-hint">One curve per Linear layer. Local generators never consume training randomness.</p>
    </div>
  </div>;
}
