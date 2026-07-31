import type { PlotSeries } from "./sweepVizPlot";
import { decoratePlotSeries } from "./sweepVizPlot";
import type { CurveSeriesTableRow } from "../components/nodes/curveSeriesDefaults";
import { effectiveCurveSeriesRows } from "../components/nodes/curveSeriesDefaults";

/* Theme-aware: resolved per active theme by tokens.css (classic values are
 * the exact legacy hex palette). Consumed as SVG inline styles, which
 * resolve var() at render time. */
const SERIES_COLORS = [
  "var(--cr-chart-1)",
  "var(--cr-chart-2)",
  "var(--cr-chart-3)",
  "var(--cr-chart-4)",
  "var(--cr-chart-5)",
  "var(--cr-chart-6)",
  "var(--cr-chart-7)",
  "var(--cr-chart-8)",
];

/* Run-mean prefix stays fixed in both themes (intentional, plan § rev.7);
 * the spread tail follows the themed series palette. */
const RUN_MEAN_COLORS = ["#2166ac", "#b2182b", ...SERIES_COLORS];

export type CurveSeriesPlotXMode = "step" | "progress" | "epoch" | "param";

export function curveSeriesPlotAxisLabel(mode: CurveSeriesPlotXMode, storedKey?: string): string {
  if (mode === "progress") return "progress %";
  if (mode === "step") return "step";
  if (mode === "epoch") return "epoch";
  if (mode === "param") return "α";
  return storedKey?.trim() || "step";
}

/** Map each series x to 0–100% of its own final step. */
export function normalizeCurveSeriesProgress(x: number[]): number[] {
  if (x.length === 0) return [];
  let max = -Infinity;
  for (const xi of x) {
    if (Number.isFinite(xi) && xi > max) max = xi;
  }
  if (!Number.isFinite(max) || max <= 0) {
    if (x.length === 1) return [100];
    return x.map((_, i) => (i / Math.max(1, x.length - 1)) * 100);
  }
  return x.map((xi) => (Number.isFinite(xi) ? (xi / max) * 100 : NaN));
}

function curveSeriesRunLengths(row: CurveSeriesTableRow): { epochs: number; steps: number } | null {
  const epochs = row.paramsNumeric?.["trainer.trainingEpochs"]
    ?? Number(row.params["trainer.trainingEpochs"]);
  const steps = row.paramsNumeric?.["trainer.trainingSteps"]
    ?? Number(row.params["trainer.trainingSteps"]);
  if (!Number.isFinite(epochs) || !Number.isFinite(steps) || steps <= 0) return null;
  return { epochs, steps };
}

function legacyXIsEpoch(row: CurveSeriesTableRow, lengths: { epochs: number; steps: number }): boolean {
  if (row.x.length === 0 || lengths.steps <= lengths.epochs * 2) return false;
  const last = row.x[row.x.length - 1];
  if (typeof last !== "number" || !Number.isFinite(last) || lengths.epochs <= 0) return false;
  return Math.abs(last - lengths.epochs) <= Math.max(1, lengths.epochs * 0.01);
}

function curveSeriesEpochs(row: CurveSeriesTableRow): number[] | null {
  if (row.epochX?.length === row.x.length) return row.epochX;
  const lengths = curveSeriesRunLengths(row);
  if (!lengths) return null;
  if (legacyXIsEpoch(row, lengths)) return row.x;
  return row.x.map((x) => x * lengths.epochs / lengths.steps);
}

function curveSeriesSteps(row: CurveSeriesTableRow): number[] {
  const lengths = curveSeriesRunLengths(row);
  if (!lengths || !legacyXIsEpoch(row, lengths)) return row.x;
  return row.x.map((x) => x * lengths.steps / lengths.epochs);
}

function curveSeriesXValues(row: CurveSeriesTableRow, mode: CurveSeriesPlotXMode): number[] {
  if (mode === "progress") return normalizeCurveSeriesProgress(row.x);
  if (mode === "epoch") return curveSeriesEpochs(row) ?? row.x;
  if (mode === "step") return curveSeriesSteps(row);
  return row.x;
}

export function buildCurveOverlayPlotSeries(
  rows: CurveSeriesTableRow[],
  selectedSeriesIds: string[] | null,
  plotXMode: CurveSeriesPlotXMode = "progress",
  meanByRun = false,
): PlotSeries[] {
  const effective = effectiveCurveSeriesRows(rows, selectedSeriesIds);
  if (meanByRun) return buildRunMeanSeries(effective, plotXMode);
  return effective.map((r, i) => {
    const xs = curveSeriesXValues(r, plotXMode);
    const base = {
      id: r.id,
      label: r.label,
      color: SERIES_COLORS[i % SERIES_COLORS.length]!,
      metricId: r.metricId,
      points: xs
        .map((x, j) => {
          const y = r.y[j];
          if (typeof y !== "number" || !Number.isFinite(y)) return null;
          if (typeof x !== "number" || !Number.isFinite(x)) return null;
          return {
            x,
            xDisplay: plotXMode === "progress" ? `${x.toFixed(1)}%` : String(x),
            y,
            rowId: r.id,
          };
        })
        .filter((p): p is NonNullable<typeof p> => p !== null),
    };
    return decoratePlotSeries([base])[0]!;
  });
}

function buildRunMeanSeries(
  rows: CurveSeriesTableRow[],
  plotXMode: CurveSeriesPlotXMode,
): PlotSeries[] {
  const groups = new Map<string, CurveSeriesTableRow[]>();
  const runColors = new Map<string, string>();
  for (const row of rows) {
    const run = row.params["trainer.run"]?.trim() || "run";
    const metric = row.metricId?.trim() || row.label;
    if (!runColors.has(run)) {
      runColors.set(run, RUN_MEAN_COLORS[runColors.size % RUN_MEAN_COLORS.length]!);
    }
    const key = `${run}\x1e${metric}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }

  const series: PlotSeries[] = [];
  for (const [key, group] of groups) {
    const first = group[0]!;
    const run = first.params["trainer.run"]?.trim() || "run";
    const metricId = first.metricId?.trim() || first.label;
    const xRows = group.map((row) => curveSeriesXValues(row, plotXMode));
    const length = Math.min(...group.map((row, i) => Math.min(row.y.length, xRows[i]!.length)));
    const points = [];
    for (let i = 0; i < length; i += 1) {
      const x = xRows[0]![i];
      const ys = group.map((row) => row.y[i]).filter((y): y is number => typeof y === "number" && Number.isFinite(y));
      if (typeof x !== "number" || !Number.isFinite(x) || ys.length !== group.length) continue;
      if (!xRows.every((xs) => xs[i] === x)) continue;
      points.push({
        x,
        xDisplay: plotXMode === "progress" ? `${x.toFixed(1)}%` : String(x),
        y: ys.reduce((sum, y) => sum + y, 0) / ys.length,
        rowId: key,
      });
    }
    const base = {
      id: key,
      label: `${run} ${metricId}`,
      color: runColors.get(run)!,
      metricId,
      points,
    };
    series.push(decoratePlotSeries([base])[0]!);
  }
  return series;
}

export function curveSeriesCanLogX(rows: CurveSeriesTableRow[]): boolean {
  for (const r of rows) {
    for (const x of r.x) {
      if (!Number.isFinite(x) || x <= 0) return false;
    }
  }
  return rows.some((r) => r.x.length > 0);
}

export function curveSeriesCanLogY(rows: CurveSeriesTableRow[]): boolean {
  for (const r of rows) {
    for (const y of r.y) {
      if (!Number.isFinite(y) || y <= 0) return false;
    }
  }
  return rows.some((r) => r.y.length > 0);
}
