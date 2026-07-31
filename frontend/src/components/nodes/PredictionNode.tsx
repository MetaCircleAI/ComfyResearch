import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultPredictionNodeData,
  type PredictionNodeData,
  type PredictionRunSplit,
  type PredictionTensorPayload,
} from "./predictionDefaults";

const SPLIT_OPTIONS: { id: PredictionRunSplit; label: string }[] = [
  { id: "both", label: "Train + test" },
  { id: "train", label: "Train only" },
  { id: "test", label: "Test only" },
];

function patchData(
  id: string,
  patch: Partial<PredictionNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const prev = defaultPredictionNodeData((n.data ?? {}) as Partial<PredictionNodeData>);
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function shapeText(t: PredictionTensorPayload | null): string {
  if (!t || !Array.isArray(t.shape) || t.shape.length === 0) return "—";
  return `[${t.shape.join(", ")}]`;
}

export function PredictionNode({ id, data, selected }: NodeProps) {
  const d = defaultPredictionNodeData((data ?? {}) as Partial<PredictionNodeData>);
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [busy, setBusy] = useState(false);

  const update = useCallback(
    (patch: Partial<PredictionNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  const runPrediction = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    update({ lastError: null });
    try {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const res = await fetch("/api/predict/forward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_node_id: id,
          nodes: g.nodes,
          edges: g.edges,
          split: d.split ?? "both",
        }),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `Prediction failed (${res.status}).`);
      }
      const j = (await res.json()) as {
        trainerTask?: string;
        trainPrediction?: PredictionTensorPayload | null;
        testPrediction?: PredictionTensorPayload | null;
      };
      update({
        trainerTask: j.trainerTask ?? null,
        trainPrediction: j.trainPrediction ?? null,
        testPrediction: j.testPrediction ?? null,
        lastError: null,
      });
    } catch (e) {
      update({
        lastError: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  }, [busy, update, getNodes, getEdges, id, d.split]);

  const status = useMemo(() => {
    if (busy) return "Running forward pass…";
    if (d.lastError) return d.lastError;
    if (d.trainPrediction || d.testPrediction) {
      return `train ${shapeText(d.trainPrediction)} · test ${shapeText(d.testPrediction)}`;
    }
    return "Run prediction to materialize train/test outputs.";
  }, [busy, d.lastError, d.trainPrediction, d.testPrediction]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(d as Record<string, unknown>, "Prediction")}</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={runPrediction} disabled={busy}>
            Predict
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Prediction model and dataset inputs; train/test prediction outputs">
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
              <span className="cr-trainer-output-label">train pred</span>
              <Handle
                type="source"
                position={Position.Right}
                id="train_pred"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
              />
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">test pred</span>
              <Handle
                type="source"
                position={Position.Right}
                id="test_pred"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
              />
            </div>
          </div>
        </div>

        <DiscreteMultiSelect<PredictionRunSplit>
          label="predict split"
          options={SPLIT_OPTIONS}
          value={d.split}
          singleSelect
          onCommit={(next) => update({ split: (typeof next === "string" ? next : next[0] ?? "both") as PredictionRunSplit })}
          ariaLabel="Prediction split"
        />

        <p className="cr-activation-collect-summary">{status}</p>
        {d.trainerTask ? <p className="cr-tensor-viz__meta">task: {d.trainerTask}</p> : null}
      </div>
    </div>
  );
}

