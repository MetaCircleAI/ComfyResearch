import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyIntField } from "./comfyNumberFields";
import { SourceSocketRow } from "./SourceSocketRow";

export function IdnnsInitializationNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as { seed?: unknown };
  const parsed = Number(Array.isArray(raw.seed) ? raw.seed[0] : raw.seed);
  const seed = Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
  const { setNodes } = useReactFlow();

  return (
    <div
      className={`cr-node cr-node--mup-init${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "IDNNs initialization")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="initialization" label="initialization" />
        <ComfyIntField
          label="seed"
          value={seed}
          min={0}
          ariaLabel="IDNNs initialization seed"
          onCommit={(value) =>
            setNodes((nodes: Node[]) =>
              nodes.map((node) =>
                node.id === id
                  ? { ...node, data: { ...(node.data as object), seed: value } }
                  : node,
              ),
            )
          }
        />
        <p className="cr-observable-hint">
          Fan-in truncated-normal weights (±2σ) and zero biases, matching the
          released Information Bottleneck implementation.
        </p>
      </div>
    </div>
  );
}
