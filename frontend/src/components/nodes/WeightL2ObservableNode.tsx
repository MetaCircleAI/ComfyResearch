import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  NORM_AGGREGATION_OPTIONS,
  readNormAggregation,
  type NormAggregation,
} from "../../graph/observableNormAggregation";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

export function WeightL2ObservableNode({ id, data, selected }: NodeProps) {
  const normAggregation = readNormAggregation(data as object | undefined);
  const { setNodes } = useReactFlow();

  const setAgg = (next: NormAggregation) =>
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id
          ? {
              ...node,
              data: {
                ...(node.data as object),
                normAggregation: next,
                perTopLevel: next === "top_level_module",
              },
            }
          : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--observable-weight-l2${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader id={id} graphNodeType="observable_weight_l2" title="Weight L2" />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <DiscreteMultiSelect
          label="Breakdown"
          options={NORM_AGGREGATION_OPTIONS}
          value={normAggregation}
          onCommit={(next) => setAgg((Array.isArray(next) ? next[0] : next) as NormAggregation)}
          ariaLabel="Weight L2 breakdown"
          singleSelect
        />
        <p className="cr-observable-hint">
          L2 norm of model weights each log step. Same breakdown options as gradient norm (global, submodule prefix, or
          per-parameter curves).
        </p>
      </div>
    </div>
  );
}
