export type SweepDataTableRow = {
  id: string;
  rawSweep: string;
  params: Record<string, string>;
  /** Numeric sweep params where values parse as finite numbers (for downstream calculators). */
  paramsNumeric?: Record<string, number>;
  value: number;
  valueLabel: string;
  /** Human-readable tensor id when logging tensor-selector sweep rows (e.g. activation rep id). */
  tensorName?: string;
  /** Matches tensor selector node `tensorSelectorSweepSeq` when the row was captured during a sweep. */
  tensorSweepSeq?: number;
};

export type SweepDataTableNodeData = {
  rows: SweepDataTableRow[];
  /** Explicit subset of row ids for downstream table viz; `null` = use all rows. */
  selectedRowIds: string[] | null;
  /**
   * Sweep param column order. `null` = alphabetical by key.
   * Row sort uses this order (first column primary, then next, …).
   */
  paramKeyOrder: string[] | null;
};

/** Param keys present in `rows`, ordered by `stored` then any new keys (sorted). */
export function resolveParamKeyOrder(
  rows: SweepDataTableRow[],
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

/** Sort rows for display using {@link resolveParamKeyOrder}. */
export function sortSweepRowsForDisplay(
  rows: SweepDataTableRow[],
  paramKeyOrder: string[] | null | undefined,
): SweepDataTableRow[] {
  const keyOrder = resolveParamKeyOrder(rows, paramKeyOrder);
  if (rows.length <= 1) return rows;
  if (keyOrder.length === 0) return [...rows];
  return [...rows].sort((a, b) => {
    for (const k of keyOrder) {
      const cmp = (a.params[k] ?? "").localeCompare(b.params[k] ?? "", undefined, { numeric: true });
      if (cmp !== 0) return cmp;
    }
    const sw = a.rawSweep.localeCompare(b.rawSweep);
    if (sw !== 0) return sw;
    const tn = (a.tensorName ?? "").localeCompare(b.tensorName ?? "", undefined, { numeric: true });
    if (tn !== 0) return tn;
    return (a.tensorSweepSeq ?? 0) - (b.tensorSweepSeq ?? 0);
  });
}

export function defaultSweepDataTableData(): SweepDataTableNodeData {
  return { rows: [], selectedRowIds: null, paramKeyOrder: null };
}

/** Short label after ``Tensor selector ·`` or last `` · `` segment of a resolve summary. */
export function tensorDisplayNameFromSourceSummary(summary: string): string {
  const s = summary.trim();
  const m = /^Tensor selector · (.+)$/i.exec(s);
  if (m) return m[1]!.trim();
  const idx = s.lastIndexOf(" · ");
  if (idx >= 0) return s.slice(idx + " · ".length).trim();
  return s;
}
