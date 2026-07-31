import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback } from "react";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
  type FlowNodeBare,
} from "../../graph/resolveUpstreamTensor";
import { defaultShapeCheckerData, type ShapeCheckerNodeData } from "./shapeCheckerDefaults";

function patchShapeCheckerData(
  id: string,
  patch: Partial<ShapeCheckerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = defaultShapeCheckerData((n.data ?? {}) as Partial<ShapeCheckerNodeData>);
      return { ...n, data: { ...cur, ...patch } };
    }),
  );
}

export function ShapeCheckerNode({ id, data, selected }: NodeProps) {
  const d = defaultShapeCheckerData((data ?? {}) as Partial<ShapeCheckerNodeData>);
  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) => resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
    resolvedTensorEqual,
  );
  const { display } = useHydratedResolved(resolved);

  const update = useCallback(
    (patch: Partial<ShapeCheckerNodeData>) => patchShapeCheckerData(id, patch, setNodes),
    [id, setNodes],
  );

  const checkShape = useCallback(() => {
    if (display.kind !== "ok") {
      update({ lastError: display.detail, shapeText: null, sourceSummary: null });
      return;
    }
    const shapeText = display.shape.length > 0 ? `[${display.shape.join(", ")}]` : "[]";
    update({ shapeText, sourceSummary: display.sourceSummary, lastError: null });
  }, [display, update]);

  return (
    <div
      className={`cr-node cr-node--shape-checker${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis, var(--cr-accent-tensor))" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Shape checker")}</div>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Shape checker tensor input">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor</span>
            </div>
          </div>
        </div>
        <div className="cr-tensor-constant-footer nodrag nopan">
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={checkShape}>
            Check shape
          </button>
        </div>
        {d.shapeText ? (
          <p className="cr-activation-collect-summary">
            shape: <strong>{d.shapeText}</strong>
            {d.sourceSummary ? ` (${d.sourceSummary})` : ""}
          </p>
        ) : null}
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
