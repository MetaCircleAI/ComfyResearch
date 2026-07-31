/** Row shape for sweep table / plot (matches `SweepDataTableRow` in the UI). */
export type SweepPlotRow = {
  id: string;
  rawSweep: string;
  params: Record<string, string>;
  value: number;
  valueLabel: string;
};

/** Parameters whose value is identical on every row (sweep caption keys). */
export function constantParamsAcrossRows(rows: SweepPlotRow[]): Record<string, string> {
  if (rows.length === 0) return {};
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r.params)))];
  const out: Record<string, string> = {};
  for (const k of keys) {
    const first = rows[0]!.params[k] ?? "";
    const same = rows.every((r) => (r.params[k] ?? "") === first);
    if (same && first !== "") out[k] = first;
  }
  return out;
}

/** Parameters that take more than one distinct value among `rows`. */
export function varyingParamKeys(rows: SweepPlotRow[]): string[] {
  if (rows.length === 0) return [];
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r.params)))].sort((a, b) =>
    a.localeCompare(b),
  );
  const varying: string[] = [];
  for (const k of keys) {
    const set = new Set(rows.map((r) => (r.params[k] ?? "").trim()));
    set.delete("");
    if (set.size > 1) varying.push(k);
  }
  return varying;
}

export function formatConstantsTitle(constants: Record<string, string>): string {
  const entries = Object.entries(constants);
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${v}`).join(" · ");
}

function parseNumLoose(s: string): number | null {
  const t = s.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** True if every non-empty value parses as a finite number. */
export function isNumericAxis(values: string[]): boolean {
  const nonEmpty = values.filter((v) => v.trim() !== "");
  if (nonEmpty.length === 0) return false;
  return nonEmpty.every((v) => parseNumLoose(v) !== null);
}

export type PlotPoint = { x: number; xDisplay: string; y: number; rowId: string };

export type PlotSeries = {
  id: string;
  label: string;
  points: PlotPoint[];
  color: string;
  yAxis?: "left" | "right";
  strokeDasharray?: string;
  metricId?: string;
};

/** Right Y-axis for accuracy-like and observable series; left for loss/sharpness. */
export function inferSeriesYAxis(label: string, metricId?: string): "left" | "right" {
  if (metricId?.startsWith("observable:")) return "right";
  const hay = `${metricId ?? ""} ${label}`.toLowerCase();
  if (hay.includes("acc")) return "right";
  return "left";
}

/** Per-axis log: with dual axes, log-y applies to the left (loss) axis only —
 * the right axis (accuracy / observable) stays linear (paper Fig-1 form). */
export function logYForAxisSide(side: "left" | "right", dualAxis: boolean, logScaleY: boolean): boolean {
  if (dualAxis && side === "right") return false;
  return logScaleY;
}

/** Dashed line for test split series. */
export function inferSeriesStrokeDash(label: string, metricId?: string): string | undefined {
  const hay = `${metricId ?? ""} ${label}`.toLowerCase();
  if (hay.includes("test")) return "4 3";
  return undefined;
}

export function decoratePlotSeries(series: PlotSeries[]): PlotSeries[] {
  return series.map((s) => ({
    ...s,
    yAxis: s.yAxis ?? inferSeriesYAxis(s.label, s.metricId),
    strokeDasharray: s.strokeDasharray ?? inferSeriesStrokeDash(s.label, s.metricId),
  }));
}

export function dualAxisWarranted(series: PlotSeries[]): boolean {
  const decorated = decoratePlotSeries(series);
  const sides = new Set(decorated.map((s) => s.yAxis ?? "left"));
  return sides.has("left") && sides.has("right");
}

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

/**
 * Build line series: x from `xKey`, y from row value, one series per distinct
 * combination of legend keys (all varying keys except x).
 */
export function buildPlotSeries(
  rows: SweepPlotRow[],
  xKey: string,
  varyingKeys: string[],
  valueLabel: string,
): PlotSeries[] {
  const legendKeys = varyingKeys.filter((k) => k !== xKey);
  if (rows.length === 0) return [];

  const xVals = rows.map((r) => (r.params[xKey] ?? "").trim());
  const numericX = isNumericAxis(xVals);

  const groups = new Map<string, SweepPlotRow[]>();
  for (const r of rows) {
    const sig = legendKeys.map((k) => (r.params[k] ?? "").trim()).join("\x1e");
    const list = groups.get(sig) ?? [];
    list.push(r);
    groups.set(sig, list);
  }

  const series: PlotSeries[] = [];
  let ci = 0;
  for (const [sig, groupRows] of groups) {
    const sample = groupRows[0]!;
    const label =
      legendKeys.length === 0
        ? valueLabel || "y"
        : legendKeys.map((k) => `${k}=${(sample.params[k] ?? "").trim()}`).join(", ");

    const pointsRaw = groupRows.map((r) => {
      const xs = (r.params[xKey] ?? "").trim();
      const yn = r.value;
      const xn = numericX ? (parseNumLoose(xs) ?? 0) : NaN;
      return { xs, yn, xn, r };
    });

    let points: PlotPoint[];
    if (numericX) {
      points = pointsRaw
        .filter((p) => Number.isFinite(p.xn))
        .map((p) => ({
          x: p.xn,
          xDisplay: p.xs,
          y: p.yn,
          rowId: p.r.id,
        }))
        .sort((a, b) => a.x - b.x);
    } else {
      const order = [...new Set(pointsRaw.map((p) => p.xs))].sort((a, b) =>
        a.localeCompare(b, undefined, { numeric: true }),
      );
      const xToIndex = new Map(order.map((x, i) => [x, i]));
      points = pointsRaw.map((p) => {
        const idx = xToIndex.get(p.xs) ?? 0;
        return {
          x: idx,
          xDisplay: p.xs,
          y: p.yn,
          rowId: p.r.id,
        };
      });
      points.sort((a, b) => a.x - b.x);
    }

    if (points.length === 0) continue;

    series.push({
      id: sig || `series-${ci}`,
      label,
      points,
      color: SERIES_COLORS[ci % SERIES_COLORS.length]!,
    });
    ci++;
  }

  return series;
}

export function yRange(series: PlotSeries[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      if (p.y < min) min = p.y;
      if (p.y > max) max = p.y;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (min === max) {
    const d = Math.abs(min) > 1e-12 ? Math.abs(min) * 0.05 : 0.05;
    return { min: min - d, max: max + d };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

export function xRange(series: PlotSeries[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      if (p.x < min) min = p.x;
      if (p.x > max) max = p.x;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (min === max) {
    const d = Math.abs(min) > 1e-12 ? Math.abs(min) * 0.05 : 0.05;
    return { min: min - d, max: max + d };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}
