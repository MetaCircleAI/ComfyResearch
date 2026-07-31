import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useMemo } from "react";
import { resolveTableVizUpstream } from "../../graph/resolveTableVizUpstream";
import {
  effectivePlotRows,
  sortRowsForDisplay,
} from "../../graph/tableVizRegressor";
import {
  buildPlotSeries,
  constantParamsAcrossRows,
  formatConstantsTitle,
  isNumericAxis,
  varyingParamKeys,
} from "../../graph/sweepVizPlot";
import type { SweepDataTableRow } from "./sweepDataTableDefaults";
import { SweepVizLinePlot } from "./SweepVizLinePlot";
import { defaultTableVizData, type TableVizNodeData } from "./tableVizDefaults";

export function TableVizNode({ id, selected }: NodeProps) {
  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveTableVizUpstream(state.nodes as Node[], state.edges as Edge[], id),
      [id],
    ),
  );

  const nodeData = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const raw = (n?.data ?? {}) as Partial<TableVizNodeData>;
      const def = defaultTableVizData();
      return {
        plotXParamKey: raw.plotXParamKey ?? def.plotXParamKey,
        logScaleX: raw.logScaleX ?? def.logScaleX,
        logScaleY: raw.logScaleY ?? def.logScaleY,
      };
    }, [id]),
  );

  const updatePlot = useCallback(
    (patch: Partial<TableVizNodeData>) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = { ...defaultTableVizData(), ...(n.data as Partial<TableVizNodeData>) };
          return { ...n, data: { ...prev, ...patch } };
        }),
      );
    },
    [id, setNodes],
  );

  const rows = resolved?.rows ?? [];
  const selectedRowIds = resolved?.selectedRowIds ?? null;

  const displayRows = useMemo(
    () => sortRowsForDisplay(rows as SweepDataTableRow[], resolved?.paramKeyOrder ?? null),
    [rows, resolved?.paramKeyOrder],
  );

  const valueHeader =
    displayRows.length > 0 ? displayRows[displayRows.length - 1]!.valueLabel || "value" : "value";

  const effectivePlotRowsList = useMemo(
    () => effectivePlotRows(displayRows, selectedRowIds),
    [displayRows, selectedRowIds],
  );

  /** Constants are relative to the rows used for this plot (full table or sweep data table selection). */
  const constantsAll = useMemo(
    () => constantParamsAcrossRows(effectivePlotRowsList),
    [effectivePlotRowsList],
  );
  const constantsTitle = useMemo(() => formatConstantsTitle(constantsAll), [constantsAll]);

  const varyingInSelection = useMemo(
    () => varyingParamKeys(effectivePlotRowsList),
    [effectivePlotRowsList],
  );

  const validXKey =
    nodeData.plotXParamKey && varyingInSelection.includes(nodeData.plotXParamKey)
      ? nodeData.plotXParamKey
      : null;

  const plotSeries = useMemo(() => {
    if (!validXKey || effectivePlotRowsList.length === 0) return [];
    const yLabel = effectivePlotRowsList[0]?.valueLabel ?? valueHeader;
    return buildPlotSeries(effectivePlotRowsList, validXKey, varyingInSelection, yLabel);
  }, [effectivePlotRowsList, validXKey, varyingInSelection, valueHeader]);

  const xIsNumeric = useMemo(() => {
    if (!validXKey || effectivePlotRowsList.length === 0) return false;
    const xs = effectivePlotRowsList.map((r) => (r.params[validXKey] ?? "").trim());
    return isNumericAxis(xs);
  }, [effectivePlotRowsList, validXKey]);

  const canLogX = useMemo(() => {
    if (!xIsNumeric || !validXKey) return false;
    for (const r of effectivePlotRowsList) {
      const t = (r.params[validXKey] ?? "").trim();
      const n = Number(t);
      if (!Number.isFinite(n) || n <= 0) return false;
    }
    return effectivePlotRowsList.length > 0;
  }, [effectivePlotRowsList, validXKey, xIsNumeric]);

  const canLogY = useMemo(() => {
    for (const r of effectivePlotRowsList) {
      if (!Number.isFinite(r.value) || r.value <= 0) return false;
    }
    return effectivePlotRowsList.length > 0;
  }, [effectivePlotRowsList]);

  const effectiveLogX = !!(nodeData.logScaleX && canLogX);
  const effectiveLogY = !!(nodeData.logScaleY && canLogY);

  const legendSummary = useMemo(() => {
    if (!validXKey) return "";
    const keys = varyingInSelection.filter((k) => k !== validXKey);
    if (keys.length === 0) return "";
    return `Legend: ${keys.join(", ")}`;
  }, [validXKey, varyingInSelection]);

  const varyingSummary = useMemo(() => {
    if (varyingInSelection.length === 0) return "Varying (in upstream selection): —";
    return `Varying (in upstream selection): ${varyingInSelection.join(", ")}`;
  }, [varyingInSelection]);

  const connected = resolved !== null;

  const yAxisLabel = effectivePlotRowsList[0]?.valueLabel ?? valueHeader;

  return (
    <div
      className={`cr-node cr-node--table-viz${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}
    >
      <div className="cr-node__header">Table viz</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Table viz sockets">
          <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
            <div className="cr-tviz-socket-row__left">
              <Handle
                type="target"
                position={Position.Left}
                id="table"
                className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
              />
              <span className="cr-tviz-socket-label">table</span>
            </div>
            <div className="cr-tviz-socket-row__right">
              <div className="cr-tviz-socket-pair">
                <span className="cr-tviz-socket-label">tensor</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id="tensor"
                  className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
                />
              </div>
            </div>
          </div>
        </div>
        <p className="cr-sweep-viz__hint cr-tensor-viz__hint">
          Connect <strong>Sweep data table</strong> <code>table</code> → <code>table</code>. Row selection is controlled
          on the sweep data table node. Choose the x-axis parameter here; other varying parameters define series in the
          legend. With a <strong>numeric</strong> x-axis, connect <code>tensor</code> → <strong>Regressor</strong>.
        </p>

        {!connected ? (
          <p className="cr-tensor-viz__hint">Connect a sweep data table to plot.</p>
        ) : rows.length === 0 ? (
          <p className="cr-tensor-viz__hint">Upstream table has no rows yet.</p>
        ) : (
          <div className="cr-sweep-viz__plot-wrap nodrag nopan">
            <p className="cr-sweep-viz__plot-varying">{varyingSummary}</p>
            <div className="cr-sweep-viz__plot-controls cr-sweep-viz__plot-controls--xrow">
              <span className="cr-sweep-viz__plot-x-label">X axis (sweep parameter)</span>
              <select
                className="cr-sweep-viz__plot-select"
                aria-label="X axis sweep parameter"
                value={
                  nodeData.plotXParamKey && varyingInSelection.includes(nodeData.plotXParamKey)
                    ? nodeData.plotXParamKey
                    : ""
                }
                onChange={(e) => {
                  const v = e.target.value;
                  updatePlot({ plotXParamKey: v || null });
                }}
              >
                <option value="">Choose…</option>
                {varyingInSelection.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
            <div className="cr-tviz-chart-controls nodrag nopan">
              <label className="cr-tviz-check cr-tviz-check--log-x">
                <input
                  type="checkbox"
                  checked={!!nodeData.logScaleX}
                  disabled={!canLogX}
                  onChange={(e) => updatePlot({ logScaleX: e.target.checked })}
                />
                log x
              </label>
              <label className="cr-tviz-check cr-tviz-check--log-y">
                <input
                  type="checkbox"
                  checked={!!nodeData.logScaleY}
                  disabled={!canLogY}
                  onChange={(e) => updatePlot({ logScaleY: e.target.checked })}
                />
                log y
              </label>
            </div>
            {effectivePlotRowsList.length > 0 ? (
              <div
                className={`cr-sweep-viz__lineplot-chart-title${
                  constantsTitle.trim() ? "" : " cr-sweep-viz__lineplot-chart-title--empty"
                }`}
                title={constantsTitle || undefined}
              >
                {constantsTitle.trim()
                  ? constantsTitle
                  : "— (no parameters identical on every row in this set)"}
              </div>
            ) : null}
            {varyingInSelection.length === 0 ? (
              <p className="cr-sweep-viz__plot-muted">
                Selected upstream rows share identical sweep parameters — widen selection on the sweep data table.
              </p>
            ) : !validXKey ? (
              <p className="cr-sweep-viz__plot-muted">Choose an x-axis parameter to draw the line plot.</p>
            ) : plotSeries.length === 0 ? (
              <p className="cr-sweep-viz__plot-muted">No series to plot.</p>
            ) : (
              <SweepVizLinePlot
                chartId={id}
                series={plotSeries}
                xKey={validXKey}
                xIsNumeric={xIsNumeric}
                yAxisLabel={yAxisLabel}
                legendSummary={legendSummary}
                logScaleX={effectiveLogX}
                logScaleY={effectiveLogY}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
