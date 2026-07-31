import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { SourceSocketRow } from "./SourceSocketRow";
import { ComfyFloatListField } from "./comfyMultiFields";
import { floatChoices, packFloatList } from "./multiValueUtils";
import {
  defaultSymmetrizedMlpInitData,
  type SymmetrizedMlpInitNodeData,
} from "./symmetrizedMlpInitDefaults";

export function SymmetrizedMlpInitNode({ id, data, selected }: NodeProps) {
  const d = { ...defaultSymmetrizedMlpInitData(), ...(data as Partial<SymmetrizedMlpInitNodeData>) };
  const { setNodes } = useReactFlow();

  const update = useCallback(
    (patch: Partial<SymmetrizedMlpInitNodeData>) =>
      setNodes((nodes: Node[]) =>
        nodes.map((n) =>
          n.id !== id ? n : { ...n, data: { ...defaultSymmetrizedMlpInitData(), ...(n.data as object), ...patch } },
        ),
      ),
    [id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--mup-init${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "Symmetrized init (Chizat 2019)")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="initialization" label="initialization" />
        <ComfyFloatListField
          label="τ (init std)"
          values={floatChoices(d.tau ?? 1.0, 1.0)}
          positiveOnly={true}
          title="Gaussian std τ for first m/2 neurons. Second half is mirrored with negated output scalars, ensuring zero output at initialization (Chizat et al. 2019 §3.1)."
          ariaLabel="Initialization std tau"
          onCommit={(vals) => update({ tau: packFloatList(vals) })}
        />
        <p className="cr-observable-hint">
          Student init: W₁[j] ~ N(0,τ²), W₁[j+m/2] = W₁[j], a[j+m/2] = −a[j].
          Guarantees zero output at t=0. Use <strong>output scale α</strong> on the MLP to control the lazy↔rich transition.
        </p>
      </div>
    </div>
  );
}
