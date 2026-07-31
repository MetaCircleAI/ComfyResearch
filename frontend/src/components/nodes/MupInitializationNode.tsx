import type { NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { SourceSocketRow } from "./SourceSocketRow";

export function MupInitializationNode({ data, selected }: NodeProps) {
  return (
    <div
      className={`cr-node cr-node--mup-init${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "MuP initialization")}</div>
      <div className="cr-node__body cr-node__body--compact">
        <SourceSocketRow handleId="initialization" label="initialization" />
        <p className="cr-observable-hint">
          Connect to a model&apos;s <strong>initialization</strong> socket. Training will re-init Linear / Embedding /
          LayerNorm weights with μP-style variance scaling before optimization.
        </p>
      </div>
    </div>
  );
}
