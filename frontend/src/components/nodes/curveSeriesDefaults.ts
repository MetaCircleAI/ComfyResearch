export type CurveSeriesTableRow = {
  id: string;
  label: string;
  x: number[];
  y: number[];
  params: Record<string, string>;
  /** Numeric sweep params where values parse as finite numbers. */
  paramsNumeric?: Record<string, number>;
  rawSweep: string;
  /** Metric key captured for this row (e.g. `train_acc`). */
  metricId?: string;
};

export type CurveSeriesTableNodeData = {
  rows: CurveSeriesTableRow[];
  /** Explicit subset of series ids for downstream curve viz; `null` = use all rows. */
  selectedSeriesIds: string[] | null;
  /** Metrics to capture on train complete (see {@link CURVE_SERIES_METRIC_LABELS}). */
  captureMetrics: string[];
  /** Sweep param column order; `null` = alphabetical. */
  paramKeyOrder: string[] | null;
};

export const CURVE_SERIES_METRIC_LABELS: Record<string, string> = {
  train_acc: "train acc",
  test_acc: "test acc",
  train_loss: "train loss",
  test_loss: "test loss",
  reg_loss: "reg loss",
  // 通用观测量捕获:抽 paired observable_viz 的 valueHistory,
  // 落表 metricId 为 observable:<obsType>,label 取节点标签。
  observable: "observable value",
};

export const DEFAULT_CURVE_SERIES_CAPTURE_METRICS = ["train_acc", "test_acc"];

export function defaultCurveSeriesTableData(): CurveSeriesTableNodeData {
  return {
    rows: [],
    selectedSeriesIds: null,
    captureMetrics: [...DEFAULT_CURVE_SERIES_CAPTURE_METRICS],
    paramKeyOrder: null,
  };
}

export function resolveCurveParamKeyOrder(
  rows: CurveSeriesTableRow[],
  stored: string[] | null | undefined,
): string[] {
  const present = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r.params)) present.add(k);
  }
  if (present.size === 0) return [];
  if (!stored || stored.length === 0) {
    return [...present].sort((a, b) => a.localeCompare(b));
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const k of stored) {
    if (present.has(k) && !seen.has(k)) {
      out.push(k);
      seen.add(k);
    }
  }
  const rest = [...present].filter((k) => !seen.has(k)).sort((a, b) => a.localeCompare(b));
  out.push(...rest);
  return out;
}

export function sortCurveSeriesRowsForDisplay(
  rows: CurveSeriesTableRow[],
  paramKeyOrder: string[] | null | undefined,
): CurveSeriesTableRow[] {
  const keyOrder = resolveCurveParamKeyOrder(rows, paramKeyOrder);
  if (rows.length <= 1) return rows;
  if (keyOrder.length === 0) return [...rows];
  return [...rows].sort((a, b) => {
    for (const k of keyOrder) {
      const cmp = (a.params[k] ?? "").localeCompare(b.params[k] ?? "", undefined, { numeric: true });
      if (cmp !== 0) return cmp;
    }
    const sw = a.rawSweep.localeCompare(b.rawSweep);
    if (sw !== 0) return sw;
    return (a.label ?? "").localeCompare(b.label ?? "", undefined, { numeric: true });
  });
}

export function effectiveCurveSeriesRows(
  rows: CurveSeriesTableRow[],
  selectedSeriesIds: string[] | null,
): CurveSeriesTableRow[] {
  if (selectedSeriesIds === null) return rows;
  const set = new Set(selectedSeriesIds);
  return rows.filter((r) => set.has(r.id));
}
