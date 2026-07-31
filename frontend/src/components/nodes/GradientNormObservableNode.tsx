import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  NORM_AGGREGATION_OPTIONS,
  readNormAggregation,
  type NormAggregation,
} from "../../graph/observableNormAggregation";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

function readGradientNormNormalized(data: object | undefined): boolean {
  const raw = (data ?? {}) as { gradientNormNormalized?: unknown };
  const v = raw.gradientNormNormalized;
  if (typeof v === "boolean") return v;
  if (v === 0 || v === "0" || v === "false") return false;
  if (v === 1 || v === "1" || v === "true") return true;
  return true;
}

export function GradientNormObservableNode({ id, data, selected }: NodeProps) {
  const normAggregation = readNormAggregation(data as object | undefined);
  const normalized = readGradientNormNormalized(data as object | undefined);
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

  const setNormalized = (next: boolean) =>
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id ? { ...node, data: { ...(node.data as object), gradientNormNormalized: next } } : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--observable-gradient-norm${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_gradient_norm"
        title={readInstanceTitle(data, "Gradient norm")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <DiscreteMultiSelect
          label="Breakdown"
          options={NORM_AGGREGATION_OPTIONS}
          value={normAggregation}
          onCommit={(next) => setAgg((Array.isArray(next) ? next[0] : next) as NormAggregation)}
          ariaLabel="Gradient norm breakdown"
          singleSelect
        />
        <label className="cr-observable-gradient-norm-normalized nodrag nopan">
          <input
            type="checkbox"
            className="nodrag nopan"
            checked={normalized}
            onChange={(e) => setNormalized(e.target.checked)}
          />
          normalized
        </label>
        <p className="cr-observable-hint">
          L2 norm of parameter gradients after each logged step. With <strong>normalized</strong> on, divide by the
          square root of the parameter count in the same scope (whole model, top-level bucket, or each tensor).
          Breakdown groups by global total, first name segment (submodule), or one curve per parameter tensor (capped).
        </p>
      </div>
    </div>
  );
}
