import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type NodeProps,
} from "@xyflow/react";
import { useCallback } from "react";
import type { FlowEdge, FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { resolveUpstreamTensor } from "../../graph/resolveUpstreamTensor";
import { defaultEffectiveRankData, type EffectiveRankNodeData } from "./effectiveRankDefaults";

function patchEffectiveRankData(
  id: string,
  patch: Partial<EffectiveRankNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultEffectiveRankData();
      const cur = (n.data ?? {}) as Partial<EffectiveRankNodeData>;
      const prev: EffectiveRankNodeData = {
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function EffectiveRankNode({ id, data, selected }: NodeProps) {
  const def = defaultEffectiveRankData();
  const raw = (data ?? {}) as Partial<EffectiveRankNodeData>;
  const d: EffectiveRankNodeData = {
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
    (patch: Partial<EffectiveRankNodeData>) => patchEffectiveRankData(id, patch, setNodes),
    [id, setNodes],
  );

  const compute = useCallback(async () => {
    const r = await hydrateResolved(resolved);
    if (r.kind !== "ok") {
      update({ lastError: r.detail, outputTensor: null });
      return;
    }
    if (r.values.length === 0) {
      update({ lastError: "Empty tensor.", outputTensor: null });
      return;
    }
    try {
      const res = await fetch("/api/effective_rank_value", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shape: r.shape, values: r.values }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const j = (await res.json()) as { value?: number };
      const v = typeof j.value === "number" && Number.isFinite(j.value) ? j.value : NaN;
      update({
        outputTensor: { shape: [], values: [v] },
        lastError: Number.isFinite(v) ? null : "Invalid value from server.",
      });
    } catch (e) {
      update({
        lastError: e instanceof Error ? e.message : String(e),
        outputTensor: null,
      });
    }
  }, [resolved, update]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Effective rank</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Effective rank tensor in and scalar out">
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
          Entropy-based effective rank from singular values (matches training-time metric). Connect any matrix
          (or higher-rank tensor flattened like the trainer).
        </p>
        {d.lastError ? <p className="cr-activation-scan-msg">{d.lastError}</p> : null}
        {d.outputTensor && d.outputTensor.values.length > 0 ? (
          <p className="cr-activation-collect-summary">
            Scalar: <strong>{d.outputTensor.values[0]!.toFixed(6)}</strong>
          </p>
        ) : null}
      </div>
    </div>
  );
}
