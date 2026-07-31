import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { defaultSvdData, type SvdNodeData } from "./svdDefaults";

function patchSvdData(
  id: string,
  patch: Partial<SvdNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultSvdData();
      const cur = (n.data ?? {}) as Partial<SvdNodeData>;
      const prev: SvdNodeData = {
        representationId: cur.representationId ?? def.representationId,
        removeMean: cur.removeMean ?? def.removeMean,
        uTensor: cur.uTensor ?? def.uTensor,
        sTensor: cur.sTensor ?? def.sTensor,
        vTensor: cur.vTensor ?? def.vTensor,
        svdSummary: cur.svdSummary ?? def.svdSummary,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

type SvdStreamProgress = { type: "progress"; step: number; total: number };
type SvdStreamComplete = {
  type: "complete";
  representation_id: string;
  u: { shape: number[]; values: number[] };
  s: { shape: number[]; values: number[] };
  v: { shape: number[]; values: number[] };
  summary: string;
};

function SvdInWrap({ label }: { label: string }) {
  return (
    <div className="cr-trainer-io-row__leftwrap">
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

function SvdOutWrap({ handleId, label }: { handleId: string; label: string }) {
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

export function SvdNode({ id, data, selected }: NodeProps) {
  const def = defaultSvdData();
  const raw = (data ?? {}) as Partial<SvdNodeData>;
  const d: SvdNodeData = {
    representationId: raw.representationId ?? def.representationId,
    removeMean: raw.removeMean ?? def.removeMean,
    uTensor: raw.uTensor ?? def.uTensor,
    sTensor: raw.sTensor ?? def.sTensor,
    vTensor: raw.vTensor ?? def.vTensor,
    svdSummary: raw.svdSummary ?? def.svdSummary,
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
    (patch: Partial<SvdNodeData>) => patchSvdData(id, patch, setNodes),
    [id, setNodes],
  );

  const runSvd = useCallback(async () => {
    setRunError(null);
    setRunLoading(true);
    setRunProgressPct(0);
    if (progressHideTo.current !== null) {
      window.clearTimeout(progressHideTo.current);
      progressHideTo.current = null;
    }

    const handleEvent = (raw: SvdStreamProgress | SvdStreamComplete) => {
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
      const res = await fetch("/api/svd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          svd_node_id: id,
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
      let complete: SvdStreamComplete | null = null;

      const flushLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        const ev = JSON.parse(trimmed) as SvdStreamProgress | SvdStreamComplete;
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
        throw new Error("SVD stream ended without a complete event");
      }
      const donePayload = complete;

      update({
        representationId: donePayload.representation_id,
        uTensor: {
          shape: donePayload.u.shape,
          values: donePayload.u.values,
        },
        sTensor: {
          shape: donePayload.s.shape,
          values: donePayload.s.values,
        },
        vTensor: {
          shape: donePayload.v.shape,
          values: donePayload.v.values,
        },
        svdSummary: donePayload.summary,
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
      className={`cr-node cr-node--svd${selected ? " cr-node--selected" : ""}`}
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
            aria-label="SVD progress"
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
          <span className="cr-node__header-title">SVD</span>
          <button
            type="button"
            className="cr-trainer-train-btn nodrag nopan"
            disabled={runLoading}
            onClick={() => void runSvd()}
          >
            {runLoading ? "…" : "Run"}
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="SVD inputs and outputs">
          <div className="cr-trainer-io-row">
            <SvdInWrap label="tensor" />
            <SvdOutWrap handleId="u" label="U (2D)" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" aria-hidden />
            <SvdOutWrap handleId="s" label="S (1D)" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" aria-hidden />
            <SvdOutWrap handleId="v" label="V (2D)" />
          </div>
        </div>

        <label className="cr-activation-rep-check nodrag nopan" style={{ marginTop: "0.35rem" }}>
          <input
            type="checkbox"
            checked={d.removeMean}
            onChange={(e) => update({ removeMean: e.target.checked })}
            aria-label="Remove column mean before SVD"
          />
          <span className="cr-trainer-socket-label" style={{ marginLeft: 0 }}>
            remove column mean
          </span>
        </label>

        {d.svdSummary ? <p className="cr-activation-summary">{d.svdSummary}</p> : null}
        {runError ? <p className="cr-trainer-train-err">{runError}</p> : null}
      </div>
    </div>
  );
}
