import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { createPortal } from "react-dom";
import { findTensorSelectorFeedingTensorViz } from "../../graph/findUpstreamTensorSelector";
import { findUpstreamSweepMetadata } from "../../graph/findUpstreamSweepSummary";
import { mergeSweepParamRecords, parseSweepParamsFromSummary, coerceSweepParamsNumeric } from "../../graph/sweepParamExtract";
import {
  orderedSelectedTensorKeysForPicker,
  resolveUpstreamTensor,
  tensorChoicesForTensorsInput,
  type FlowEdge,
  type FlowNodeBare,
  type Resolved,
} from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { formatTensorScalarDisplay } from "./tensorVizScalarFormat";
import {
  tensorSelectorOutputIndexFromSourceHandle,
  type TensorSelectorNodeData,
} from "./tensorSelectorDefaults";
import {
  resolveParamKeyOrder,
  sortSweepRowsForDisplay,
  tensorDisplayNameFromSourceSummary,
  type SweepDataTableNodeData,
  type SweepDataTableRow,
} from "./sweepDataTableDefaults";

const MAX_ROWS = 500;

function doubleRaf(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

function buildRowSweepParams(
  structured: Record<string, string>,
  rawSweep: string,
): { params: Record<string, string>; paramsNumeric: Record<string, number> } {
  const params = mergeSweepParamRecords(structured, parseSweepParamsFromSummary(rawSweep));
  return { params, paramsNumeric: coerceSweepParamsNumeric(params) };
}

function snapshotEqual(
  a: { resolved: Resolved; sweep: string; sweepParams: Record<string, string> },
  b: { resolved: Resolved; sweep: string; sweepParams: Record<string, string> },
): boolean {
  if (a.sweep !== b.sweep) return false;
  const ak = Object.keys(a.sweepParams).sort().join("\0");
  const bk = Object.keys(b.sweepParams).sort().join("\0");
  if (ak !== bk) return false;
  for (const k of Object.keys(a.sweepParams)) {
    if (a.sweepParams[k] !== b.sweepParams[k]) return false;
  }
  return resolvedEqual(a.resolved, b.resolved);
}

function resolvedEqual(a: Resolved, b: Resolved): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "none" && b.kind === "none") return a.detail === b.detail;
  if (a.kind === "lazy_activation" && b.kind === "lazy_activation") {
    return (
      a.runId === b.runId &&
      a.repId === b.repId &&
      a.sourceSummary === b.sourceSummary &&
      a.shape.length === b.shape.length &&
      a.shape.every((v, i) => v === b.shape[i]!)
    );
  }
  if (a.kind === "ok" && b.kind === "ok") {
    if (a.rank !== b.rank || a.sourceSummary !== b.sourceSummary) return false;
    if (a.shape.length !== b.shape.length) return false;
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) return false;
    }
    if (a.values.length !== b.values.length) return false;
    const len = a.values.length;
    const fullCompare = a.rank === 0 || len <= 256;
    const n = fullCompare ? len : Math.min(8, len);
    for (let i = 0; i < n; i++) {
      if (a.values[i] !== b.values[i]) return false;
    }
    return true;
  }
  return false;
}

type ColDragGhostState = {
  fromIndex: number;
  label: string;
  width: number;
  height: number;
  grabDx: number;
  grabDy: number;
  x: number;
  y: number;
};

type SweepStreamMeta =
  | { kind: "none" }
  | {
      kind: "tensor_selector_sweep";
      selectorId: string;
      sweeping: boolean;
      sweepSeq: number;
    };

function sweepStreamMetaEqual(a: SweepStreamMeta, b: SweepStreamMeta): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "none") return true;
  return (
    a.selectorId === b.selectorId &&
    a.sweeping === b.sweeping &&
    a.sweepSeq === b.sweepSeq
  );
}

function reorderStringArray(arr: string[], from: number, to: number): string[] {
  if (from === to || from < 0 || to < 0 || from >= arr.length || to >= arr.length) return arr;
  const next = [...arr];
  const [x] = next.splice(from, 1);
  next.splice(to, 0, x);
  return next;
}

export function SweepDataTableNode({ id, selected }: NodeProps) {
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const dragRef = useRef<{ anchor: number } | null>(null);
  const colDragFromRef = useRef<number | null>(null);
  const colDragHoverRef = useRef<number | null>(null);
  const [colDragActive, setColDragActive] = useState(false);
  const [colDragHoverIdx, setColDragHoverIdx] = useState<number | null>(null);
  const [colDragGhost, setColDragGhost] = useState<ColDragGhostState | null>(null);
  const colDragListenersRef = useRef<{ move: (e: MouseEvent) => void; up: () => void } | null>(null);
  const sweepTableMountedRef = useRef(true);
  useEffect(() => {
    sweepTableMountedRef.current = true;
    return () => {
      sweepTableMountedRef.current = false;
    };
  }, []);

  const snapshot = useStore(
    useCallback(
      (state) => {
        const nodes = state.nodes as Node[];
        const edges = state.edges as FlowEdge[];
        const resolved = resolveUpstreamTensor(nodes, edges, id, "stream");
        const meta = findUpstreamSweepMetadata(nodes, edges, id);
        return { resolved, sweep: meta.summary, sweepParams: meta.params };
      },
      [id],
    ),
    snapshotEqual,
  );

  const sweepStreamMeta = useStore(
    useCallback((state): SweepStreamMeta => {
      const nodes = state.nodes as Node[];
      const edges = state.edges as FlowEdge[];
      const streamEdge = edges.find((e) => e.target === id && (e.targetHandle ?? "") === "stream");
      if (!streamEdge?.source) return { kind: "none" };
      const srcNode = nodes.find((n) => n.id === streamEdge.source);
      if (!srcNode || srcNode.type !== "tensor_viz_0d") return { kind: "none" };
      const feed = findTensorSelectorFeedingTensorViz(nodes, edges, streamEdge.source);
      if (!feed) return { kind: "none" };
      const sel = feed.selector;
      const d = (sel.data ?? {}) as {
        tensorSelectorSweeping?: boolean;
        tensorSelectorSweepSeq?: number;
      };
      return {
        kind: "tensor_selector_sweep",
        selectorId: sel.id,
        sweeping: !!d.tensorSelectorSweeping,
        sweepSeq: typeof d.tensorSelectorSweepSeq === "number" ? d.tensorSelectorSweepSeq : 0,
      };
    }, [id]),
    sweepStreamMetaEqual,
  );

  const { display: sweepDisplay } = useHydratedResolved(snapshot.resolved);

  const rows = useStore(
    useCallback(
      (state) => {
        const n = state.nodes.find((x) => x.id === id);
        const d = (n?.data ?? {}) as Partial<SweepDataTableNodeData>;
        return Array.isArray(d.rows) ? d.rows : [];
      },
      [id],
    ),
  );

  const paramKeyOrderStored = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<SweepDataTableNodeData>;
      const o = d.paramKeyOrder;
      return Array.isArray(o) ? o : null;
    }, [id]),
  );

  const selectedRowIds = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const d = (n?.data ?? {}) as Partial<SweepDataTableNodeData> & { plotSelectedRowIds?: string[] | null };
      if (d.selectedRowIds !== undefined) return d.selectedRowIds;
      if (d.plotSelectedRowIds !== undefined) return d.plotSelectedRowIds;
      return null;
    }, [id]),
  );

  useEffect(() => {
    const sweepAccumulate =
      sweepStreamMeta.kind === "tensor_selector_sweep" &&
      sweepStreamMeta.sweeping &&
      sweepStreamMeta.sweepSeq > 0;

    if (sweepAccumulate) {
      const seq = sweepStreamMeta.sweepSeq;
      const selectorId = sweepStreamMeta.selectorId;

      void (async () => {
        await doubleRaf();
        const meta = findUpstreamSweepMetadata(getNodes() as Node[], getEdges() as FlowEdge[], id);
        const sweep = meta.summary;
        const rawSweep = sweep.trim();
        const { params, paramsNumeric } = buildRowSweepParams(meta.params, rawSweep);

        const maxAttempts = 120;
        for (let attempt = 0; attempt < maxAttempts && sweepTableMountedRef.current; attempt++) {
          const nodes = getNodes() as Node[];
          const edges = getEdges() as FlowEdge[];
          const streamEdge = edges.find((e) => e.target === id && (e.targetHandle ?? "") === "stream");
          const vizId = streamEdge?.source;
          if (!vizId) return;

          const feed = findTensorSelectorFeedingTensorViz(nodes, edges, vizId);
          if (!feed || feed.selector.id !== selectorId) return;

          const resolvedNow = resolveUpstreamTensor(nodes as FlowNodeBare[], edges, vizId, "tensor");
          const displayNow = await hydrateResolved(resolvedNow);
          if (displayNow.kind === "ok" && displayNow.rank === 0 && displayNow.values.length > 0) {
            const v = displayNow.values[0]!;
            const valueLabel = displayNow.sourceSummary;

            setNodes((nds) => {
              const edgesNow = getEdges() as FlowEdge[];
              const selNode = nds.find((x) => x.id === selectorId);
              const selDataNow = (selNode?.data ?? {}) as Partial<TensorSelectorNodeData>;
              const choicesNow = tensorChoicesForTensorsInput(
                nds as FlowNodeBare[],
                edgesNow,
                selectorId,
              );
              const orderedNow = orderedSelectedTensorKeysForPicker(selDataNow, choicesNow);
              const feedNow = findTensorSelectorFeedingTensorViz(nds as Node[], edgesNow, vizId);
              const outIdxNow = feedNow
                ? tensorSelectorOutputIndexFromSourceHandle(feedNow.selectorOutputHandle)
                : 0;
              const snapNow = selDataNow.tensorSelectorSweepSnapshots?.[seq];
              const keyNow =
                snapNow?.[outIdxNow] ?? snapNow?.[0] ?? orderedNow[outIdxNow] ?? orderedNow[0];
              const labelNow = keyNow ? choicesNow.find((c) => c.id === keyNow)?.label : undefined;
              const tensorNameFinal = keyNow
                ? (labelNow ?? keyNow).trim()
                : tensorDisplayNameFromSourceSummary(valueLabel);

              return nds.map((n) => {
                if (n.id !== id) return n;
                const prev = (n.data ?? {}) as SweepDataTableNodeData;
                const prevRows = [...(prev.rows ?? [])];
                const existingIdx = prevRows.findIndex((r) => r.tensorSweepSeq === seq);

                if (existingIdx >= 0) {
                  const existing = prevRows[existingIdx]!;
                  const same =
                    existing.value === v &&
                    existing.valueLabel === valueLabel &&
                    (existing.tensorName ?? "") === tensorNameFinal &&
                    (existing.rawSweep.trim() || "\0") === (rawSweep || "\0");
                  if (same) return n;
                  prevRows[existingIdx] = {
                    ...existing,
                    rawSweep: sweep,
                    params,
                    paramsNumeric,
                    value: v,
                    valueLabel,
                    tensorName: tensorNameFinal,
                  };
                  return { ...n, data: { ...prev, rows: prevRows } };
                }

                const row: SweepDataTableRow = {
                  id: `row-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                  rawSweep: sweep,
                  params,
                  paramsNumeric,
                  value: v,
                  valueLabel,
                  tensorName: tensorNameFinal,
                  tensorSweepSeq: seq,
                };
                prevRows.push(row);
                const trimmed = prevRows.length > MAX_ROWS ? prevRows.slice(-MAX_ROWS) : prevRows;
                return { ...n, data: { ...prev, rows: trimmed } };
              });
            });
            return;
          }

          await new Promise((r) => setTimeout(r, 50));
        }
      })();

      return;
    }

    const { sweep, sweepParams } = snapshot;
    if (sweepDisplay.kind !== "ok" || sweepDisplay.rank !== 0 || sweepDisplay.values.length === 0) {
      return;
    }
    const v = sweepDisplay.values[0]!;
    const rawSweep = sweep.trim();
    const { params, paramsNumeric } = buildRowSweepParams(sweepParams, rawSweep);
    const valueLabel = sweepDisplay.sourceSummary;

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prev = (n.data ?? {}) as SweepDataTableNodeData;
        const prevRows = [...(prev.rows ?? [])];

        const comboKey = rawSweep || "\0";
        const idx = prevRows.findIndex((r) => (r.rawSweep.trim() || "\0") === comboKey);

        if (idx >= 0) {
          const existing = prevRows[idx]!;
          if (existing.value === v && existing.valueLabel === valueLabel) {
            return n;
          }
          prevRows[idx] = {
            ...existing,
            params,
            paramsNumeric,
            value: v,
            valueLabel,
          };
          return { ...n, data: { ...prev, rows: prevRows } };
        }

        const row: SweepDataTableRow = {
          id: `row-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          rawSweep: sweep,
          params,
          paramsNumeric,
          value: v,
          valueLabel,
        };
        prevRows.push(row);
        const trimmed = prevRows.length > MAX_ROWS ? prevRows.slice(-MAX_ROWS) : prevRows;
        return {
          ...n,
          data: { ...prev, rows: trimmed },
        };
      }),
    );
  }, [id, setNodes, getNodes, getEdges, snapshot, sweepDisplay, sweepStreamMeta]);

  useEffect(() => {
    const end = () => {
      dragRef.current = null;
    };
    window.addEventListener("mouseup", end);
    return () => window.removeEventListener("mouseup", end);
  }, []);

  const paramKeys = useMemo(
    () => resolveParamKeyOrder(rows, paramKeyOrderStored),
    [rows, paramKeyOrderStored],
  );

  const displayRows = useMemo(
    () => sortSweepRowsForDisplay(rows, paramKeyOrderStored),
    [rows, paramKeyOrderStored],
  );

  const valueHeader =
    displayRows.length > 0 ? displayRows[displayRows.length - 1]!.valueLabel || "value" : "value";

  const showTensorScalarColumns = useMemo(
    () => displayRows.some((r) => typeof r.tensorName === "string" && r.tensorName.length > 0),
    [displayRows],
  );

  const selectedIdSet = useMemo(() => {
    if (selectedRowIds === null) return null;
    return new Set(selectedRowIds);
  }, [selectedRowIds]);

  const isRowSelected = useCallback(
    (rowId: string) => {
      if (selectedRowIds === null) return true;
      return selectedIdSet?.has(rowId) ?? false;
    },
    [selectedRowIds, selectedIdSet],
  );

  const endColDrag = useCallback(() => {
    const from = colDragFromRef.current;
    const to = colDragHoverRef.current;
    colDragFromRef.current = null;
    colDragHoverRef.current = null;
    setColDragActive(false);
    setColDragHoverIdx(null);
    if (from === null || to === null || from === to) return;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prev = (n.data ?? {}) as SweepDataTableNodeData;
        const order = resolveParamKeyOrder(prev.rows ?? [], prev.paramKeyOrder ?? null);
        if (from < 0 || from >= order.length || to < 0 || to >= order.length) return n;
        return { ...n, data: { ...prev, paramKeyOrder: reorderStringArray(order, from, to) } };
      }),
    );
  }, [id, setNodes]);

  useEffect(() => {
    return () => {
      const L = colDragListenersRef.current;
      if (L) {
        window.removeEventListener("mousemove", L.move);
        window.removeEventListener("mouseup", L.up);
        colDragListenersRef.current = null;
      }
      colDragFromRef.current = null;
      colDragHoverRef.current = null;
    };
  }, []);

  const onParamHeaderMouseDown = useCallback(
    (e: MouseEvent, fromIndex: number, label: string) => {
      e.preventDefault();
      e.stopPropagation();
      const th = e.currentTarget as HTMLElement;
      const rect = th.getBoundingClientRect();
      colDragFromRef.current = fromIndex;
      colDragHoverRef.current = fromIndex;
      setColDragActive(true);
      setColDragHoverIdx(fromIndex);
      setColDragGhost({
        fromIndex,
        label,
        width: rect.width,
        height: rect.height,
        grabDx: e.clientX - rect.left,
        grabDy: e.clientY - rect.top,
        x: e.clientX,
        y: e.clientY,
      });

      const L = colDragListenersRef.current;
      if (L) {
        window.removeEventListener("mousemove", L.move);
        window.removeEventListener("mouseup", L.up);
      }

      const onMove = (ev: MouseEvent) => {
        setColDragGhost((g) => (g ? { ...g, x: ev.clientX, y: ev.clientY } : null));
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        colDragListenersRef.current = null;
        setColDragGhost(null);
        endColDrag();
      };
      colDragListenersRef.current = { move: onMove, up: onUp };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp, { once: true });
    },
    [endColDrag],
  );

  const onParamHeaderMouseEnter = useCallback((index: number) => {
    if (colDragFromRef.current === null) return;
    colDragHoverRef.current = index;
    setColDragHoverIdx(index);
  }, []);

  const updateSelection = useCallback(
    (patch: Partial<Pick<SweepDataTableNodeData, "selectedRowIds">>) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = (n.data ?? {}) as SweepDataTableNodeData;
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
          const prev = (n.data ?? {}) as SweepDataTableNodeData;
          const allIds = sortSweepRowsForDisplay(prev.rows ?? [], prev.paramKeyOrder ?? null).map(
            (r) => r.id,
          );
          let base: string[];
          if (prev.selectedRowIds === null) base = [...allIds];
          else base = [...prev.selectedRowIds];
          const set = new Set(base);
          if (set.has(rowId)) set.delete(rowId);
          else set.add(rowId);
          return {
            ...n,
            data: { ...prev, selectedRowIds: [...set] },
          };
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
      updateSelection({ selectedRowIds: [row.id] });
    },
    [displayRows, toggleRow, updateSelection],
  );

  const onRowMouseEnter = useCallback(
    (rowIndex: number) => {
      const drag = dragRef.current;
      if (!drag) return;
      const i0 = Math.min(drag.anchor, rowIndex);
      const i1 = Math.max(drag.anchor, rowIndex);
      const ids = displayRows.slice(i0, i1 + 1).map((r) => r.id);
      updateSelection({ selectedRowIds: ids });
    },
    [displayRows, updateSelection],
  );

  return (
    <div
      className={`cr-node cr-node--sweep-data-table${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}
    >
      <div className="cr-node__header">Sweep data table</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-sweep-viz__toolbar nodrag nopan">
          <button
            type="button"
            className="cr-sweep-viz__clear"
            onClick={() => {
              setNodes((nds) =>
                nds.map((n) => {
                  if (n.id !== id) return n;
                  const prev = (n.data ?? {}) as SweepDataTableNodeData;
                  return {
                    ...n,
                    data: {
                      ...prev,
                      rows: [],
                      selectedRowIds: null,
                      paramKeyOrder: null,
                    },
                  };
                }),
              );
            }}
          >
            Clear
          </button>
          <button
            type="button"
            className="cr-sweep-viz__clear"
            title="Use every row for downstream table viz (all rows highlighted)"
            onClick={() => updateSelection({ selectedRowIds: null })}
          >
            Select all rows
          </button>
        </div>
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Sweep data table sockets">
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
                <span className="cr-tviz-socket-label">table</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id="table"
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
        <p className="cr-sweep-viz__hint cr-tensor-viz__hint">
          Connect <strong>0D tensor viz</strong> <code>tensor</code> → <strong>stream</strong>. Rows accumulate one per
          trainer sweep combo; when upstream includes a <strong>Tensor selector</strong> that is <strong>Sweep</strong>ing,
          each step appends a row (tensor name + scalar) without replacing prior sweep rows. Select rows for a downstream{" "}
          <strong>Table viz</strong> (connect <code>table</code> → table viz). Click / drag / Ctrl+click to change selection.
          Drag sweep column headers to reorder columns; rows sort by that order.
        </p>
        {rows.length === 0 ? (
          <p className="cr-tensor-viz__hint">No rows yet — run a sweep or connect upstream.</p>
        ) : (
          <div className="cr-sweep-viz__table-wrap nodrag nopan">
            <table className="cr-sweep-viz__table">
              <thead>
                <tr>
                  {paramKeys.map((k, ki) => (
                    <th
                      key={k}
                      className={
                        colDragGhost?.fromIndex === ki
                          ? "cr-sweep-viz__th cr-sweep-viz__th--param cr-sweep-viz__th--col-drag-source"
                          : colDragActive && colDragHoverIdx === ki
                            ? "cr-sweep-viz__th cr-sweep-viz__th--param cr-sweep-viz__th--col-drag-hover"
                            : "cr-sweep-viz__th cr-sweep-viz__th--param"
                      }
                      title="Drag header to reorder columns; rows sort by column order (left → right)."
                      onMouseDown={(e) => onParamHeaderMouseDown(e, ki, k)}
                      onMouseEnter={() => onParamHeaderMouseEnter(ki)}
                    >
                      {k}
                    </th>
                  ))}
                  {showTensorScalarColumns ? (
                    <>
                      <th className="cr-sweep-viz__th">Tensor</th>
                      <th className="cr-sweep-viz__th">Scalar</th>
                    </>
                  ) : (
                    <th className="cr-sweep-viz__th">{valueHeader}</th>
                  )}
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
                    {showTensorScalarColumns ? (
                      <>
                        <td title={r.tensorName ?? ""}>{r.tensorName ?? "—"}</td>
                        <td className="cr-sweep-viz__value" title={String(r.value)}>
                          {formatTensorScalarDisplay(r.value)}
                        </td>
                      </>
                    ) : (
                      <td className="cr-sweep-viz__value" title={String(r.value)}>
                        {formatTensorScalarDisplay(r.value)}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
            {selectedRowIds === null ? (
              <p className="cr-sweep-viz__plot-muted cr-sweep-viz__table-footer-hint">
                Selection: all rows. Table viz uses every row.
              </p>
            ) : null}
          </div>
        )}
      </div>
      {colDragGhost &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            className="cr-sweep-viz__col-drag-ghost"
            style={{
              width: colDragGhost.width,
              minHeight: colDragGhost.height,
              left: colDragGhost.x - colDragGhost.grabDx,
              top: colDragGhost.y - colDragGhost.grabDy,
            }}
            aria-hidden
          >
            {colDragGhost.label}
          </div>,
          document.body,
        )}
    </div>
  );
}
