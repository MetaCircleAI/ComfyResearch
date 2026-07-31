import { useReactFlow, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";
import { ComfyIntField } from "./comfyNumberFields";

type HessianEigenOrder = "descending" | "ascending";

const ORDER_OPTIONS: { id: HessianEigenOrder; label: string }[] = [
  { id: "descending", label: "Descending (largest first)" },
  { id: "ascending", label: "Ascending (smallest first)" },
];

export function HessianEigenvaluesObservableNode({ id, data, selected }: NodeProps) {
  const raw = (data ?? {}) as { topK?: unknown; order?: unknown };
  const topK = Number.isFinite(Number(raw.topK)) ? Math.max(1, Math.floor(Number(raw.topK))) : 5;
  const order: HessianEigenOrder = raw.order === "ascending" ? "ascending" : "descending";
  const { setNodes } = useReactFlow();

  return (
    <div
      className={`cr-node cr-node--observable-hessian-eigenvalues${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType="observable_hessian_eigenvalues"
        title={readInstanceTitle(data, "Hessian eigenvalues")}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        <ComfyIntField
          label="top eigenvalues"
          value={topK}
          min={1}
          max={256}
          title="How many Hessian eigenvalue ranks to log each step."
          onCommit={(n) =>
            setNodes((nds) =>
              nds.map((node) =>
                node.id === id ? { ...node, data: { ...(node.data as object), topK: Math.max(1, n) } } : node,
              ),
            )
          }
          ariaLabel="Top Hessian eigenvalue count"
        />
        <DiscreteMultiSelect<HessianEigenOrder>
          label="rank order"
          options={ORDER_OPTIONS}
          value={order}
          singleSelect
          onCommit={(next) =>
            setNodes((nds) =>
              nds.map((node) =>
                node.id === id
                  ? {
                      ...node,
                      data: {
                        ...(node.data as object),
                        order: next === "ascending" ? "ascending" : "descending",
                      },
                    }
                  : node,
              ),
            )
          }
          ariaLabel="Hessian eigenvalue sort order"
        />
        <p className="cr-observable-hint">
          Computes Hessian eigenvalues of current training loss with respect to model parameters and logs the selected
          top ranks.
        </p>
      </div>
    </div>
  );
}
