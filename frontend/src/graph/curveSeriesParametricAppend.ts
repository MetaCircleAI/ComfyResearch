/**
 * Pure append path for the parametric path sampler → curve_series_table wire.
 * Train-complete accumulation lives in curveSeriesAppend.ts; this module contains
 * only the sampler push path and no debug hooks.
 */
import type { Edge, Node } from "@xyflow/react";
import type {
  CurveSeriesTableNodeData,
  CurveSeriesTableRow,
} from "../components/nodes/curveSeriesDefaults";
import { formatSweepParamsSummary } from "./sweepParamExtract";

const MAX_ROWS = 200;

export type ExtractedCurve = {
  metricId: string;
  label: string;
  x: number[];
  epochX?: number[];
  y: number[];
};

function curveSeriesRowComboKey(
  rawSweep: string,
  params: Record<string, string>,
  metricId: string,
): string {
  const rawPart = rawSweep.trim();
  const paramsPart = formatSweepParamsSummary(params);
  return `${rawPart}\0${paramsPart}\0${metricId}`;
}

function seriesLabelWithParams(base: string, params: Record<string, string>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v.trim())
    .map(([k, v]) => `${k}=${v}`);
  if (parts.length === 0) return base;
  return `${base} · ${parts.join(", ")}`;
}

/** Shared row upsert, also used by curveSeriesAppend train-complete accumulation. */
export function appendRowsToTable(
  prev: CurveSeriesTableNodeData,
  curves: ExtractedCurve[],
  rawSweep: string,
  params: Record<string, string>,
  paramsNumeric: Record<string, number>,
): CurveSeriesTableNodeData {
  const prevRows = [...(prev.rows ?? [])];
  for (const curve of curves) {
    const comboKey = curveSeriesRowComboKey(rawSweep, params, curve.metricId);
    const idx = prevRows.findIndex(
      (r) => curveSeriesRowComboKey(r.rawSweep, r.params ?? {}, r.metricId ?? "") === comboKey,
    );
    const row: CurveSeriesTableRow = {
      id:
        idx >= 0
          ? prevRows[idx]!.id
          : `cser-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      label: seriesLabelWithParams(curve.label, params),
      x: curve.x,
      epochX: curve.epochX,
      y: curve.y,
      params,
      paramsNumeric,
      rawSweep,
      metricId: curve.metricId,
    };
    if (idx >= 0) prevRows[idx] = row;
    else prevRows.push(row);
  }
  const trimmed = prevRows.length > MAX_ROWS ? prevRows.slice(-MAX_ROWS) : prevRows;
  return { ...prev, rows: trimmed };
}

function downstreamCurveTableIds(edges: Edge[], samplerId: string): string[] {
  return edges
    .filter((e) => e.source === samplerId && e.sourceHandle === "stream" && e.targetHandle === "stream")
    .map((e) => e.target);
}

/** Push parametric path sampler output into wired curve_series_table nodes. */
export function appendParametricPathToCurveTables(
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
  nodes: Node[],
  edges: Edge[],
  samplerId: string,
  payload: {
    alphaSeries: number[];
    series: Array<{ metricId: string; label: string; values: number[] }>;
  },
): void {
  void nodes;
  const tableIds = new Set(downstreamCurveTableIds(edges, samplerId));
  if (tableIds.size === 0) return;

  const curves: ExtractedCurve[] = payload.series.map((s) => ({
    metricId: s.metricId,
    label: s.label,
    x: payload.alphaSeries,
    y: s.values,
  }));

  setNodes((prev) =>
    prev.map((n) => {
      if (!tableIds.has(n.id)) return n;
      const prevData = (n.data ?? {}) as CurveSeriesTableNodeData;
      let data = prevData;
      for (const curve of curves) {
        data = appendRowsToTable(data, [curve], "", {}, {});
      }
      return { ...n, data };
    }),
  );
}
