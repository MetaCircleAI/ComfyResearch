import type { Edge, Node } from "@xyflow/react";
import { sortSweepRowsForDisplay, type SweepDataTableRow } from "../components/nodes/sweepDataTableDefaults";
import type { TableVizNodeData } from "../components/nodes/tableVizDefaults";
import { resolveTableVizUpstream } from "./resolveTableVizUpstream";
import { buildPlotSeries, isNumericAxis, varyingParamKeys, type PlotSeries } from "./sweepVizPlot";

export type TableVizRegressorSeries =
  | { kind: "ok"; x: number[]; y: number[]; sourceSummary: string }
  | { kind: "none"; detail: string };

export function sortRowsForDisplay(
  rows: SweepDataTableRow[],
  paramKeyOrder?: string[] | null,
): SweepDataTableRow[] {
  return sortSweepRowsForDisplay(rows, paramKeyOrder ?? undefined);
}

export function effectivePlotRows(
  displayRows: SweepDataTableRow[],
  selectedRowIds: string[] | null,
): SweepDataTableRow[] {
  if (selectedRowIds === null) return displayRows;
  const set = new Set(selectedRowIds);
  return displayRows.filter((r) => set.has(r.id));
}

/** One sweep line per entry (stable `id` matches {@link PlotSeries.id}). */
export function tableVizTensorListChoices(
  nodes: Node[],
  edges: Edge[],
  tableVizId: string,
): { id: string; label: string }[] {
  const src = nodes.find((n) => n.id === tableVizId);
  if (!src || src.type !== "table_viz") return [];

  const resolved = resolveTableVizUpstream(nodes, edges, tableVizId);
  if (!resolved) return [];

  const d = (src.data ?? {}) as Partial<TableVizNodeData>;
  const plotXParamKey = d.plotXParamKey ?? null;

  const displayRows = sortRowsForDisplay(resolved.rows, resolved.paramKeyOrder);
  const effective = effectivePlotRows(displayRows, resolved.selectedRowIds);
  if (effective.length === 0) return [];

  const varyingInSelection = varyingParamKeys(effective);
  const validXKey =
    plotXParamKey && varyingInSelection.includes(plotXParamKey) ? plotXParamKey : null;
  if (!validXKey) return [];

  const xsRaw = effective.map((r) => (r.params[validXKey] ?? "").trim());
  if (!isNumericAxis(xsRaw)) return [];

  const yLabel = effective[0]?.valueLabel ?? "value";
  const series = buildPlotSeries(effective, validXKey, varyingInSelection, yLabel);
  return series
    .filter((s) => s.points.length >= 3)
    .map((s) => ({ id: s.id, label: s.label }));
}

function pickTableVizSeries(series: PlotSeries[], seriesId: string | null | undefined): PlotSeries | null {
  if (series.length === 0) return null;
  if (seriesId) return series.find((s) => s.id === seriesId) ?? null;
  if (series.length === 1) return series[0]!;
  return null;
}

/**
 * Points for one sweep line (numeric x only). With multiple legend lines, pass `seriesId`
 * (tensor selector key) or connect a single series only.
 */
export function resolveTableVizTensorSeries(
  nodes: Node[],
  edges: Edge[],
  tableVizId: string,
  seriesId?: string | null,
): TableVizRegressorSeries {
  const src = nodes.find((n) => n.id === tableVizId);
  if (!src || src.type !== "table_viz") {
    return { kind: "none", detail: "Table viz node missing." };
  }

  const resolved = resolveTableVizUpstream(nodes, edges, tableVizId);
  if (!resolved) {
    return { kind: "none", detail: "Connect a sweep data table to table viz first." };
  }

  const d = (src.data ?? {}) as Partial<TableVizNodeData>;
  const plotXParamKey = d.plotXParamKey ?? null;

  const displayRows = sortRowsForDisplay(resolved.rows, resolved.paramKeyOrder);
  const effective = effectivePlotRows(displayRows, resolved.selectedRowIds);

  if (effective.length === 0) {
    return { kind: "none", detail: "No rows to plot on table viz." };
  }

  const varyingInSelection = varyingParamKeys(effective);
  const validXKey =
    plotXParamKey && varyingInSelection.includes(plotXParamKey) ? plotXParamKey : null;

  if (!validXKey) {
    return {
      kind: "none",
      detail: "Table viz: pick an x-axis parameter (varying in the selection) for numeric regression.",
    };
  }

  const xsRaw = effective.map((r) => (r.params[validXKey] ?? "").trim());
  if (!isNumericAxis(xsRaw)) {
    return {
      kind: "none",
      detail: "Table viz: x-axis must be numeric to export a tensor series for the regressor.",
    };
  }

  const yLabel = effective[0]?.valueLabel ?? "value";
  const series = buildPlotSeries(effective, validXKey, varyingInSelection, yLabel);

  const s = pickTableVizSeries(series, seriesId);
  if (!s) {
    if (seriesId) {
      return { kind: "none", detail: "Table viz: that series is not available for regression." };
    }
    if (series.length > 1) {
      return {
        kind: "none",
        detail: "Table viz: multiple lines — use Tensor selector (or connect one series only).",
      };
    }
    return { kind: "none", detail: "No series to plot on table viz." };
  }
  const x = s.points.map((p) => p.x);
  const y = s.points.map((p) => p.y);

  if (x.length < 3) {
    return { kind: "none", detail: "Table viz needs at least three numeric points for regression." };
  }

  const summary =
    series.length > 1
      ? `Table viz · ${validXKey} vs ${yLabel} · ${s.label}`
      : `Table viz · ${validXKey} vs ${yLabel}`;

  return {
    kind: "ok",
    x,
    y,
    sourceSummary: summary,
  };
}

export function tableVizTensorConnectable(nodes: Node[], edges: Edge[], tableVizId: string): boolean {
  const choices = tableVizTensorListChoices(nodes, edges, tableVizId);
  return choices.length >= 1;
}
