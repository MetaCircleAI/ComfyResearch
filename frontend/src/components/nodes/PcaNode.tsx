import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ComfyIntField } from "./comfyNumberFields";
import { defaultPcaData, type PcaNodeData } from "./pcaDefaults";

function patchPcaData(
  id: string,
  patch: Partial<PcaNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultPcaData();
      const cur = (n.data ?? {}) as Partial<PcaNodeData>;
      const prev: PcaNodeData = {
        representationId: cur.representationId ?? def.representationId,
        nComponents: cur.nComponents ?? def.nComponents,
        transformedTensor: cur.transformedTensor ?? def.transformedTensor,
        principalComponents: cur.principalComponents ?? def.principalComponents,
        explainedVarianceRatio: cur.explainedVarianceRatio ?? def.explainedVarianceRatio,
        pcaSummary: cur.pcaSummary ?? def.pcaSummary,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

type PcaStreamProgress = { type: "progress"; step: number; total: number };
type PcaStreamComplete = {
  type: "complete";
  representation_id: string;
  transformed_tensor?: { shape: number[]; values: number[] };
  principal_components: { shape: number[]; values: number[] };
  explained_variance_ratio: number[];
  summary: string;
};

function PcaInWrap({ label }: { label: string }) {
  return (
    <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
      <Handle
        type="target"
        position={Position.Left}
        id="tensor"
        className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
      />
      <span className="cr-trainer-socket-label">{label}</span>
    </div>
  );
}

function PcaOutWrap({ handleId, label }: { handleId: string; label: string }) {
  return (
    <div className="cr-trainer-io-row__rightwrap">
      <span className="cr-trainer-output-label">{label}</span>
      <Handle
        type="source"
        position={Position.Right}
        id={handleId}
        className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
      />
    </div>
  );
}

export function PcaNode({ id, data, selected }: NodeProps) {
  const def = defaultPcaData();
  const raw = (data ?? {}) as Partial<PcaNodeData>;
  const d: PcaNodeData = {
    representationId: raw.representationId ?? def.representationId,
    nComponents: raw.nComponents ?? def.nComponents,
    transformedTensor: raw.transformedTensor ?? def.transformedTensor,
    principalComponents: raw.principalComponents ?? def.principalComponents,
    explainedVarianceRatio: raw.explainedVarianceRatio ?? def.explainedVarianceRatio,
    pcaSummary: raw.pcaSummary ?? def.pcaSummary,
  };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runProgressPct, setRunProgressPct] = useState(0);
  const progressHideTo = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (progressHideTo.current !== null) {
        window.clearTimeout(progressHideTo.current);
        progressHideTo.current = null;
      }
    },
    [],
  );

  const update = useCallback(
    (patch: Partial<PcaNodeData>) => patchPcaData(id, patch, setNodes),
    [id, setNodes],
  );

  const runPca = useCallback(async () => {
    setRunError(null);
    setRunLoading(true);
    setRunProgressPct(0);
    if (progressHideTo.current !== null) {
      window.clearTimeout(progressHideTo.current);
      progressHideTo.current = null;
    }

    const handleEvent = (raw: PcaStreamProgress | PcaStreamComplete) => {
      if (raw.type === "progress") {
        const total = Math.max(1, raw.total);
        const step = Math.min(Math.max(0, raw.step), total);
        const pct = Math.min(100, Math.round((step / total) * 100));
        setRunProgressPct(pct);
      }
    };

    try {
      const nodes = getNodes().map((n) => ({
        id: n.id,
        type: n.type as string,
        position: n.position,
        data: (n.data as Record<string, unknown>) ?? {},
      }));
      const edges = getEdges().map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle ?? null,
        targetHandle: e.targetHandle ?? null,
      }));
      const res = await fetch("/api/pca", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pca_node_id: id,
          nodes,
          edges,
        }),
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      let buffer = "";
      let complete: PcaStreamComplete | null = null;

      const flushLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        const ev = JSON.parse(trimmed) as PcaStreamProgress | PcaStreamComplete;
        if (ev.type === "complete") {
          complete = ev;
        } else {
          handleEvent(ev);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (value) {
          buffer += decoder.decode(value, { stream: true });
        }
        for (;;) {
          const nl = buffer.indexOf("\n");
          if (nl < 0) break;
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          flushLine(line);
        }
        if (done) break;
      }
      if (buffer.trim()) {
        flushLine(buffer);
      }

      if (!complete) {
        throw new Error("PCA stream ended without a complete event");
      }
      const donePayload = complete;

      update({
        representationId: donePayload.representation_id,
        transformedTensor: donePayload.transformed_tensor
          ? {
              shape: donePayload.transformed_tensor.shape,
              values: donePayload.transformed_tensor.values,
            }
          : null,
        principalComponents: {
          shape: donePayload.principal_components.shape,
          values: donePayload.principal_components.values,
        },
        explainedVarianceRatio: donePayload.explained_variance_ratio,
        pcaSummary: donePayload.summary,
      });
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e));
      setRunProgressPct(0);
    } finally {
      setRunLoading(false);
      progressHideTo.current = window.setTimeout(() => {
        progressHideTo.current = null;
        setRunProgressPct(0);
      }, 420);
    }
  }, [getEdges, getNodes, id, update]);

  const showRunProgress = runLoading || runProgressPct > 0;

  return (
    <div
      className={`cr-node cr-node--pca${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        {showRunProgress ? (
          <div
            className="cr-trainer-progress nodrag nopan"
            style={{ height: 28, minHeight: 28, maxHeight: 28, flexShrink: 0 }}
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(runProgressPct)}
            aria-valuetext={`${Math.min(100, Math.round(runProgressPct))}%`}
            aria-label="PCA progress"
          >
            <div
              className="cr-trainer-progress__fill"
              style={{ width: `${Math.min(100, runProgressPct)}%` }}
            />
            <span className="cr-trainer-progress__label">
              {Math.min(100, Math.round(runProgressPct))}%
            </span>
          </div>
        ) : null}
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">PCA</span>
          <button
            type="button"
            className="cr-trainer-train-btn nodrag nopan"
            disabled={runLoading}
            onClick={() => void runPca()}
          >
            {runLoading ? "…" : "Run"}
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="PCA inputs and outputs">
          <div className="cr-trainer-io-row">
            <PcaInWrap label="tensor" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" aria-hidden />
            <PcaOutWrap handleId="transformed_tensor" label="transformed tensor" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" aria-hidden />
            <PcaOutWrap handleId="principal_components" label="principal components" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" aria-hidden />
            <PcaOutWrap handleId="explained_variance_ratio" label="explained variance ratio" />
          </div>
        </div>

        <ComfyIntField
          label="n components"
          value={d.nComponents}
          min={0}
          title="0 = all (min of sample count and feature count)"
          onCommit={(nComponents) => update({ nComponents })}
          ariaLabel="Number of principal components (0 = all)"
        />

        {d.pcaSummary ? <p className="cr-activation-summary">{d.pcaSummary}</p> : null}
        {runError ? <p className="cr-trainer-train-err">{runError}</p> : null}
      </div>
    </div>
  );
}
