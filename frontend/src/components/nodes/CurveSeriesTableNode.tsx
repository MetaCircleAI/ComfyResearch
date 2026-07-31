import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  CURVE_SERIES_METRIC_LABELS,
  DEFAULT_CURVE_SERIES_CAPTURE_METRICS,
  defaultCurveSeriesTableData,
  resolveCurveParamKeyOrder,
  sortCurveSeriesRowsForDisplay,
  type CurveSeriesTableNodeData,
} from "./curveSeriesDefaults";

const METRIC_OPTIONS = Object.keys(CURVE_SERIES_METRIC_LABELS);

export function CurveSeriesTableNode({ id, selected }: NodeProps) {
  const { setNodes } = useReactFlow();
  const dragRef = useRef<{ anchor: number } | null>(null);

  const rows = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<CurveSeriesTableNodeData>;
      return Array.isArray(d.rows) ? d.rows : [];
    }, [id]),
  );

  const paramKeyOrderStored = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<CurveSeriesTableNodeData>;
      return Array.isArray(d.paramKeyOrder) ? d.paramKeyOrder : null;
    }, [id]),
  );

  const selectedSeriesIds = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<CurveSeriesTableNodeData>;
      return d.selectedSeriesIds !== undefined ? d.selectedSeriesIds : null;
    }, [id]),
  );

  const captureMetrics = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<CurveSeriesTableNodeData>;
      const raw = d.captureMetrics;
      if (Array.isArray(raw) && raw.length > 0) return raw.map(String);
      return [...DEFAULT_CURVE_SERIES_CAPTURE_METRICS];
    }, [id]),
  );

  const paramKeys = useMemo(
    () => resolveCurveParamKeyOrder(rows, paramKeyOrderStored),
    [rows, paramKeyOrderStored],
  );

  const displayRows = useMemo(
    () => sortCurveSeriesRowsForDisplay(rows, paramKeyOrderStored),
    [rows, paramKeyOrderStored],
  );

  const selectedIdSet = useMemo(() => {
    if (selectedSeriesIds === null) return null;
    return new Set(selectedSeriesIds);
  }, [selectedSeriesIds]);

  const isRowSelected = useCallback(
    (rowId: string) => {
      if (selectedSeriesIds === null) return true;
      return selectedIdSet?.has(rowId) ?? false;
    },
    [selectedSeriesIds, selectedIdSet],
  );

  const patchData = useCallback(
    (patch: Partial<CurveSeriesTableNodeData>) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = { ...defaultCurveSeriesTableData(), ...(n.data as Partial<CurveSeriesTableNodeData>) };
          return { ...n, data: { ...prev, ...patch } };
        }),
      );
    },
    [id, setNodes],
  );

  const toggleRow = useCallback(
    (rowId: string) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = { ...defaultCurveSeriesTableData(), ...(n.data as Partial<CurveSeriesTableNodeData>) };
          const allIds = sortCurveSeriesRowsForDisplay(prev.rows ?? [], prev.paramKeyOrder ?? null).map(
            (r) => r.id,
          );
          let base: string[];
          if (prev.selectedSeriesIds === null) base = [...allIds];
          else base = [...prev.selectedSeriesIds];
          const set = new Set(base);
          if (set.has(rowId)) set.delete(rowId);
          else set.add(rowId);
          return { ...n, data: { ...prev, selectedSeriesIds: [...set] } };
        }),
      );
    },
    [id, setNodes],
  );

  const onRowMouseDown = useCallback(
    (e: MouseEvent<HTMLTableRowElement>, rowIndex: number) => {
      e.stopPropagation();
      e.preventDefault();
      const row = displayRows[rowIndex];
      if (!row) return;
      if (e.ctrlKey || e.metaKey) {
        toggleRow(row.id);
        return;
      }
      dragRef.current = { anchor: rowIndex };
      patchData({ selectedSeriesIds: [row.id] });
    },
    [displayRows, toggleRow, patchData],
  );

  const onRowMouseEnter = useCallback(
    (rowIndex: number) => {
      const drag = dragRef.current;
      if (!drag) return;
      const i0 = Math.min(drag.anchor, rowIndex);
      const i1 = Math.max(drag.anchor, rowIndex);
      patchData({ selectedSeriesIds: displayRows.slice(i0, i1 + 1).map((r) => r.id) });
    },
    [displayRows, patchData],
  );

  const toggleMetric = useCallback(
    (metricId: string) => {
      const set = new Set(captureMetrics);
      if (set.has(metricId)) set.delete(metricId);
      else set.add(metricId);
      const next = METRIC_OPTIONS.filter((m) => set.has(m));
      patchData({ captureMetrics: next.length > 0 ? next : [metricId] });
    },
    [captureMetrics, patchData],
  );

  return (
    <div
      className={`cr-node cr-node--curve-series-table${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}
    >
      <div className="cr-node__header">Curve series table</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-sweep-viz__toolbar nodrag nopan">
          <button
            type="button"
            className="cr-sweep-viz__clear"
            onClick={() => patchData({ rows: [], selectedSeriesIds: null, paramKeyOrder: null })}
          >
            Clear
          </button>
          <button
            type="button"
            className="cr-sweep-viz__clear"
            title="Use every series for downstream curve viz"
            onClick={() => patchData({ selectedSeriesIds: null })}
          >
            Select all series
          </button>
        </div>
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Curve series table sockets">
          <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
            <div className="cr-tviz-socket-row__left">
              <Handle
                type="target"
                position={Position.Left}
                id="stream"
                className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
              />
              <span className="cr-tviz-socket-label">stream</span>
            </div>
            <div className="cr-tviz-socket-row__right cr-tviz-socket-row__right--dual">
              <div className="cr-tviz-socket-pair">
                <span className="cr-tviz-socket-label">series</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id="series"
                  className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
                />
              </div>
              <div className="cr-tviz-socket-pair">
                <span className="cr-tviz-socket-label">comment</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id="comment"
                  className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-comment"
                />
              </div>
            </div>
          </div>
        </div>
        <div className="cr-sweep-viz__plot-controls nodrag nopan">
          <span className="cr-sweep-viz__plot-x-label">Capture metrics</span>
          <div className="cr-tviz-check-row">
            {METRIC_OPTIONS.map((m) => (
              <label key={m} className="cr-tviz-check">
                <input
                  type="checkbox"
                  checked={captureMetrics.includes(m)}
                  onChange={() => toggleMetric(m)}
                />
                {CURVE_SERIES_METRIC_LABELS[m]}
              </label>
            ))}
          </div>
        </div>
        <p className="cr-sweep-viz__hint cr-tensor-viz__hint">
          Connect <strong>Training viz</strong> <code>stream</code> or <strong>Observable viz</strong>{" "}
          <code>tensor</code> → <code>stream</code>. Multiple trainers may share one table — set distinct{" "}
          <strong>instance titles</strong> on trainers/viz nodes so series labels differ. One row per metric per run
          is appended on train complete.
        </p>
        {rows.length === 0 ? (
          <p className="cr-tensor-viz__hint">No series yet — run training or connect upstream.</p>
        ) : (
          <div className="cr-sweep-viz__table-wrap nodrag nopan">
            <table className="cr-sweep-viz__table">
              <thead>
                <tr>
                  {paramKeys.map((k) => (
                    <th key={k} className="cr-sweep-viz__th cr-sweep-viz__th--param">
                      {k}
                    </th>
                  ))}
                  <th className="cr-sweep-viz__th">Series</th>
                  <th className="cr-sweep-viz__th">Points</th>
                </tr>
              </thead>
              <tbody>
                {displayRows.map((r, ri) => (
                  <tr
                    key={r.id}
                    className={
                      isRowSelected(r.id) ? "cr-sweep-viz__tr cr-sweep-viz__tr--selected" : "cr-sweep-viz__tr"
                    }
                    onMouseDown={(e) => onRowMouseDown(e, ri)}
                    onMouseEnter={() => onRowMouseEnter(ri)}
                  >
                    {paramKeys.map((k) => (
                      <td key={k}>{r.params[k] ?? ""}</td>
                    ))}
                    <td title={r.label}>{r.label}</td>
                    <td>{Math.min(r.x.length, r.y.length)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {selectedSeriesIds === null ? (
              <p className="cr-sweep-viz__plot-muted cr-sweep-viz__table-footer-hint">
                Selection: all series. Curve viz overlays every row.
              </p>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
