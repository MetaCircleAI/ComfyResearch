import { useReactFlow, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatField } from "./comfyNumberFields";
import { SourceSocketRow } from "./SourceSocketRow";

export function SaxeInitializationNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as { amplitude?: unknown };
  const amplitude = Number.isFinite(Number(raw.amplitude)) ? Math.max(1e-6, Number(raw.amplitude)) : 0.01;
  const { setNodes } = useReactFlow();

  return (
    <div
      className={`cr-node cr-node--saxe-init${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "Orthogonal Small-Scale Initialization")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="initialization" label="initialization" />
        <ComfyFloatField
          label="amplitude ε"
          value={amplitude}
          positiveOnly
          min={1e-6}
          max={10}
          title="Applies W = ε · Q, where Q is orthogonal. A small ε (for example, 0.01) keeps the initial effective weight product near zero and supports staged singular-value learning."
          onCommit={(n) =>
            setNodes((nds) =>
              nds.map((node) =>
                node.id === id
                  ? { ...node, data: { ...(node.data as object), amplitude: Math.max(1e-6, n) } }
                  : node,
              ),
            )
          }
          ariaLabel="Orthogonal initialization amplitude"
        />
        <p className="cr-observable-hint">
          Re-initialises all Linear weights as <strong>ε · Q</strong> (Q orthogonal, Haar measure). Connect to a
          model&apos;s <strong>initialization</strong> socket. With <em>identity</em> activation, the small initial
          effective weight product makes staged singular-value learning easier to observe.
        </p>
      </div>
    </div>
  );
}
