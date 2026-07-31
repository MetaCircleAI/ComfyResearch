import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useEffect } from "react";
import type { FlowEdge, FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { resolveUpstreamTensor } from "../../graph/resolveUpstreamTensor";
import { defaultSeriesEndpointGapData, type SeriesEndpointGapNodeData } from "./seriesEndpointGapDefaults";

function patchSeriesEndpointGapData(
  id: string,
  patch: Partial<SeriesEndpointGapNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultSeriesEndpointGapData();
      const cur = (n.data ?? {}) as Partial<SeriesEndpointGapNodeData>;
      const prev: SeriesEndpointGapNodeData = {
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function SeriesEndpointGapNode({ id, data, selected }: NodeProps) {
  const def = defaultSeriesEndpointGapData();
  const raw = (data ?? {}) as Partial<SeriesEndpointGapNodeData>;
  const d: SeriesEndpointGapNodeData = {
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
  );

  const update = useCallback(
    (patch: Partial<SeriesEndpointGapNodeData>) => patchSeriesEndpointGapData(id, patch, setNodes),
    [id, setNodes],
  );

  const compute = useCallback(async () => {
    const r = await hydrateResolved(resolved);
    if (r.kind !== "ok") {
      update({ lastError: r.detail, outputTensor: null });
      return;
    }
    if (r.shape.length !== 1) {
      update({
        lastError: "Input must be a rank-1 (1D) tensor. Reduce other axes first (e.g. Statistics / slice).",
        outputTensor: null,
      });
      return;
    }
    const vals = r.values;
    if (vals.length === 0) {
      update({ lastError: "Empty series.", outputTensor: null });
      return;
    }
    const gap = vals[vals.length - 1]! - vals[0]!;
    update({
      outputTensor: { shape: [], values: [gap] },
      lastError: Number.isFinite(gap) ? null : "Non-finite endpoint gap.",
    });
  }, [resolved, update]);

  useEffect(() => {
    if (resolved.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    update({ outputTensor: null, lastError: resolved.detail });
  }, [resolved, d.outputTensor, update]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Series endpoint gap</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="1D series in, scalar gap out">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
              />
            </div>
          </div>
        </div>
        <p className="cr-statistics-hint">
          Scalar <strong>last − first</strong> along a <strong>1D</strong> tensor (one axis). Wire a vector or a
          single row/column after slicing.
        </p>
        {d.lastError ? <p className="cr-activation-scan-msg">{d.lastError}</p> : null}
        {d.outputTensor && d.outputTensor.values.length > 0 ? (
          <p className="cr-activation-collect-summary">
            Gap: <strong>{d.outputTensor.values[0]!.toFixed(6)}</strong>
          </p>
        ) : null}
      </div>
    </div>
  );
}
