import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useState } from "react";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import {
  defaultModelWeightTensorsData,
  type ModelWeightTensorsNodeData,
} from "./modelWeightTensorsDefaults";

function patchData(
  id: string,
  patch: Partial<ModelWeightTensorsNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultModelWeightTensorsData();
      const cur = (n.data ?? {}) as Partial<ModelWeightTensorsNodeData>;
      const prev: ModelWeightTensorsNodeData = {
        weightTensorPayloads: cur.weightTensorPayloads ?? def.weightTensorPayloads,
        scanMessage: cur.scanMessage ?? def.scanMessage,
        scanSummary: cur.scanSummary ?? def.scanSummary,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function ModelWeightTensorsNode({ id, data, selected }: NodeProps) {
  const def = defaultModelWeightTensorsData();
  const raw = (data ?? {}) as Partial<ModelWeightTensorsNodeData>;
  const d: ModelWeightTensorsNodeData = {
    weightTensorPayloads: raw.weightTensorPayloads ?? def.weightTensorPayloads,
    scanMessage: raw.scanMessage ?? def.scanMessage,
    scanSummary: raw.scanSummary ?? def.scanSummary,
  };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const update = useCallback(
    (patch: Partial<ModelWeightTensorsNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  const collect = useCallback(async () => {
    setErr(null);
    setLoading(true);
    try {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const res = await fetch("/api/collect_model_weights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes: g.nodes,
          edges: g.edges,
          model_weight_tensors_node_id: id,
        }),
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
      const j = (await res.json()) as {
        weights?: Record<string, { shape: number[]; values: number[] }>;
        summary?: string;
      };
      const w = j.weights ?? {};
      update({
        weightTensorPayloads: w,
        scanMessage: null,
        scanSummary: typeof j.summary === "string" ? j.summary : null,
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      update({ weightTensorPayloads: {}, scanSummary: null });
    } finally {
      setLoading(false);
    }
  }, [getEdges, getNodes, id, update]);

  const n = Object.keys(d.weightTensorPayloads).length;

  return (
    <div
      className={`cr-node cr-node--activation${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header cr-node__header--activation">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">Model weight tensors</span>
          <button type="button" className="cr-activation-collect-btn nodrag nopan" disabled={loading} onClick={() => void collect()}>
            {loading ? "…" : "Collect"}
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Model weight tensors inputs and tensor list output">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model"
              />
              <span className="cr-trainer-socket-label">model</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">tensor list</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor_list"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor-list"
              />
            </div>
          </div>
        </div>
        <p className="cr-node__hint">
          Connect a model, then Collect to materialize all <code>named_parameters</code> for downstream Tensor
          selector and analysis (architecture comes from the model node).
        </p>
        {d.scanMessage ? <p className="cr-activation-scan-msg">{d.scanMessage}</p> : null}
        {d.scanSummary ? <p className="cr-activation-collect-summary">{d.scanSummary}</p> : null}
        {err ? <p className="cr-activation-scan-msg">{err}</p> : null}
        {!d.scanSummary && !err && n === 0 ? (
          <p className="cr-activation-scan-msg">No weights loaded yet.</p>
        ) : null}
        {n > 0 ? (
          <p className="cr-activation-summary">
            {n} parameter tensor{n === 1 ? "" : "s"} available in the Tensor selector dropdown.
          </p>
        ) : null}
      </div>
    </div>
  );
}
