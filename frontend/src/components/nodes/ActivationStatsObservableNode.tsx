import { useReactFlow, type NodeProps } from "@xyflow/react";
import {
  ACTIVATION_STATS_LAYER_OPTIONS,
  readActivationStatsLayerMode,
  type ActivationStatsLayerMode,
} from "../../graph/activationStatsLayerMode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

export function ActivationStatsObservableNode({ id, data, selected }: NodeProps) {
  const layerMode = readActivationStatsLayerMode(data as object | undefined);
  const { setNodes } = useReactFlow();

  const setLayerMode = (next: ActivationStatsLayerMode) =>
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id ? { ...node, data: { ...(node.data as object), activationStatsLayers: next } } : node,
      ),
    );

  return (
    <div
      className={`cr-node cr-node--observable-activation-stats${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_activation_stats"
        title={readInstanceTitle(data, "Activation mean/std")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <DiscreteMultiSelect
          label="Layers"
          options={ACTIVATION_STATS_LAYER_OPTIONS}
          value={layerMode}
          onCommit={(next) =>
            setLayerMode((Array.isArray(next) ? next[0] : next) as ActivationStatsLayerMode)
          }
          ariaLabel="Activation stats layer aggregation"
          singleSelect
        />
        <p className="cr-observable-hint">
          Hooks Linear/Conv outputs. Layers align with submodule paths such as <code className="cr-code-inline">.layers.0.</code>,{" "}
          <code className="cr-code-inline">.layers.1.</code>, …; remaining hooked modules roll into <strong>Other modules</strong>.{" "}
          <strong>Global</strong> averages bucket statistics with equal weight per bucket; <strong>All layers</strong> logs mean and std per bucket (viz shows multiple curves). Extra forward on the logging batch.
        </p>
      </div>
    </div>
  );
}
