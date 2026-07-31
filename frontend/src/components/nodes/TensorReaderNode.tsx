import { Handle, Position, useStore, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
  type FlowNodeBare,
} from "../../graph/resolveUpstreamTensor";
import { reshapeFlatToNested } from "../../graph/tensorNestedJson";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { TensorReaderModal } from "./TensorReaderModal";

const DISPLAY_MAX_ELEMENTS = 50_000;

export function TensorReaderNode({ id, data, selected }: NodeProps) {
  const [open, setOpen] = useState(false);
  const [refetchKey, setRefetchKey] = useState(0);

  const resolved = useStore(
    useCallback(
      (state) => resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
    resolvedTensorEqual,
  );
  const { display, loading } = useHydratedResolved(resolved, refetchKey);

  const shapeText = display.kind === "ok" ? `[${display.shape.join(", ")}]` : "—";

  const jsonText = useMemo(() => {
    if (display.kind !== "ok") return "";
    if (display.textPreview) return display.textPreview;
    const { values, shape } = display;
    const n = values.length;
    if (n <= DISPLAY_MAX_ELEMENTS) {
      return JSON.stringify(reshapeFlatToNested(values, shape), null, 2);
    }
    return JSON.stringify(
      {
        _note: `Large tensor (${n} elements). Showing first ${DISPLAY_MAX_ELEMENTS} in row-major order.`,
        shape,
        valuesPrefix: values.slice(0, DISPLAY_MAX_ELEMENTS),
      },
      null,
      2,
    );
  }, [display]);

  const sourceSummary = display.kind === "ok" ? display.sourceSummary : null;
  const errText = loading ? null : display.kind === "none" ? display.detail : null;

  const onRefresh = useCallback(() => {
    setRefetchKey((k) => k + 1);
  }, []);

  return (
    <div
      className={`cr-node cr-node--tensor-reader${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis, var(--cr-accent-tensor))" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor reader")}</div>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Tensor reader input">
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
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => setOpen(true)}>
            View tensor values
          </button>
        </div>
        <p className="cr-node__hint">Read-only nested JSON, same layout as View/edit parameters.</p>
      </div>
      <TensorReaderModal
        open={open}
        shapeText={shapeText}
        sourceSummary={sourceSummary}
        jsonText={jsonText}
        loading={loading}
        error={errText}
        onRefresh={onRefresh}
        onClose={() => setOpen(false)}
      />
    </div>
  );
}
