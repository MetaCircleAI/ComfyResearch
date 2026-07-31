import type { CurvePoint } from "./observableCurvePayload";
import type { CurveStarerAnalyzedEntry, CurveStarerRankBy } from "./lpdTypes";

export type CurveStarerInterestingnessAttributeId = Exclude<
  CurveStarerRankBy,
  "default" | "interestingness"
>;

export type CurveStarerAttributeScore = {
  id: CurveStarerInterestingnessAttributeId;
  label: string;
  score: number;
  value: number | null;
  valueLabel: string;
};

export type CurveStarerInterestingnessBreakdown = {
  overall: number;
  attributes: CurveStarerAttributeScore[];
  /** Attribute label with the highest score (why this curve stands out). */
  topReason: string | null;
};

function finiteLossValues(points: CurvePoint[]): number[] {
  return points.map((p) => p.loss).filter((v) => Number.isFinite(v));
}

export function meanSegmentR2(entry: CurveStarerAnalyzedEntry): number | null {
  const lpd = entry.lpd;
  if (!lpd) return null;
  if (typeof lpd.mean_segment_r2 === "number" && Number.isFinite(lpd.mean_segment_r2)) {
    return lpd.mean_segment_r2;
  }
  if (typeof lpd.global_r2 === "number" && Number.isFinite(lpd.global_r2)) {
    return lpd.global_r2;
  }
  return null;
}

export function stageCount(entry: CurveStarerAnalyzedEntry): number | null {
  const lpd = entry.lpd;
  if (!lpd) return null;
  if (typeof lpd.num_phases === "number" && Number.isFinite(lpd.num_phases)) {
    return lpd.num_phases;
  }
  if (lpd.segments && lpd.segments.length > 0) return lpd.segments.length;
  if (typeof lpd.selected_k === "number" && Number.isFinite(lpd.selected_k)) {
    return lpd.selected_k;
  }
  return null;
}

/** Relative span vs overall magnitude: smaller ⇒ more conserved. */
export function relativeVariation(entry: CurveStarerAnalyzedEntry): number | null {
  const values = finiteLossValues(entry.points);
  if (values.length < 2) return null;
  let min = values[0]!;
  let max = values[0]!;
  let sum = 0;
  for (const v of values) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
  }
  const span = max - min;
  const mean = sum / values.length;
  const base = Math.max(Math.abs(mean), span * 0.5, 1e-9);
  return span / base;
}

type InterestingnessMetricSpec = {
  id: CurveStarerInterestingnessAttributeId;
  label: string;
  ascending: boolean;
  metric: (entry: CurveStarerAnalyzedEntry) => number | null;
  formatValue: (value: number | null) => string;
};

export const CURVE_STARER_INTERESTINGNESS_METRICS: InterestingnessMetricSpec[] = [
  {
    id: "hard_to_fit",
    label: "Hard to fit (low R²)",
    ascending: true,
    metric: meanSegmentR2,
    formatValue: (v) => (v != null ? `avg R² ${v.toFixed(3)}` : "—"),
  },
  {
    id: "easy_to_fit",
    label: "Easy to fit (high R²)",
    ascending: false,
    metric: meanSegmentR2,
    formatValue: (v) => (v != null ? `avg R² ${v.toFixed(3)}` : "—"),
  },
  {
    id: "few_stages",
    label: "Few stages",
    ascending: true,
    metric: stageCount,
    formatValue: (v) => (v != null ? `${v} stages` : "—"),
  },
  {
    id: "many_stages",
    label: "Many stages",
    ascending: false,
    metric: stageCount,
    formatValue: (v) => (v != null ? `${v} stages` : "—"),
  },
  {
    id: "small_variation",
    label: "Small variation / conserved",
    ascending: true,
    metric: relativeVariation,
    formatValue: (v) => (v != null ? `rel. span ${v.toFixed(3)}` : "—"),
  },
  {
    id: "large_variation",
    label: "Large variation / unstable",
    ascending: false,
    metric: relativeVariation,
    formatValue: (v) => (v != null ? `rel. span ${v.toFixed(3)}` : "—"),
  },
];

function compareNullable(
  a: number | null,
  b: number | null,
  ascending: boolean,
  entryIdA: string,
  entryIdB: string,
): number {
  if (a == null && b == null) return entryIdA.localeCompare(entryIdB);
  if (a == null) return 1;
  if (b == null) return -1;
  const diff = ascending ? a - b : b - a;
  if (diff !== 0) return diff;
  return entryIdA.localeCompare(entryIdB);
}

function sortIndicesByMetric(
  entries: CurveStarerAnalyzedEntry[],
  metric: (entry: CurveStarerAnalyzedEntry) => number | null,
  ascending: boolean,
): number[] {
  const indices = entries.map((_, i) => i);
  indices.sort((ia, ib) =>
    compareNullable(metric(entries[ia]!), metric(entries[ib]!), ascending, entries[ia]!.entryId, entries[ib]!.entryId),
  );
  return indices;
}

/**
 * Linear rank score: 1st place = 1, last place = 0, evenly spaced between.
 * Tied metrics share the average rank position (e.g. two at top both score 1).
 */
function rankScoresFromOrder(
  sortedIndices: number[],
  metric: (entry: CurveStarerAnalyzedEntry) => number | null,
  entries: CurveStarerAnalyzedEntry[],
): Map<number, number> {
  const n = sortedIndices.length;
  const scores = new Map<number, number>();
  if (n === 0) return scores;
  if (n === 1) {
    scores.set(sortedIndices[0]!, 1);
    return scores;
  }

  let rank = 0;
  while (rank < n) {
    const idx = sortedIndices[rank]!;
    const value = metric(entries[idx]!);
    let end = rank + 1;
    while (end < n) {
      const nextIdx = sortedIndices[end]!;
      const nextValue = metric(entries[nextIdx]!);
      if (value == null && nextValue == null) {
        end++;
        continue;
      }
      if (value == null || nextValue == null || value !== nextValue) break;
      end++;
    }
    const avgRank = (rank + end - 1) / 2;
    const score = 1 - avgRank / (n - 1);
    for (let r = rank; r < end; r++) {
      scores.set(sortedIndices[r]!, score);
    }
    rank = end;
  }
  return scores;
}

export function interestingnessBreakdownForEntries(
  entries: CurveStarerAnalyzedEntry[],
): Map<string, CurveStarerInterestingnessBreakdown> {
  const out = new Map<string, CurveStarerInterestingnessBreakdown>();
  if (entries.length === 0) return out;

  const perEntryAttributes: CurveStarerAttributeScore[][] = entries.map(() => []);

  for (const spec of CURVE_STARER_INTERESTINGNESS_METRICS) {
    const order = sortIndicesByMetric(entries, spec.metric, spec.ascending);
    const scores = rankScoresFromOrder(order, spec.metric, entries);
    entries.forEach((entry, i) => {
      const value = spec.metric(entry);
      perEntryAttributes[i]!.push({
        id: spec.id,
        label: spec.label,
        score: scores.get(i) ?? 0,
        value,
        valueLabel: spec.formatValue(value),
      });
    });
  }

  entries.forEach((entry, i) => {
    const attributes = perEntryAttributes[i]!;
    const overall = Math.max(...attributes.map((a) => a.score), 0);
    const topAttr = attributes.reduce(
      (best, attr) => (attr.score > best.score ? attr : best),
      attributes[0]!,
    );
    out.set(entry.entryId, {
      overall,
      attributes,
      topReason: topAttr.score > 0 ? topAttr.label : null,
    });
  });
  return out;
}

export function overallInterestingnessScore(entries: CurveStarerAnalyzedEntry[]): Map<string, number> {
  const breakdowns = interestingnessBreakdownForEntries(entries);
  const out = new Map<string, number>();
  for (const [entryId, bd] of breakdowns) {
    out.set(entryId, bd.overall);
  }
  return out;
}

export function rankCurveStarerEntries(
  entries: CurveStarerAnalyzedEntry[],
  rankBy: CurveStarerRankBy,
): CurveStarerAnalyzedEntry[] {
  if (rankBy === "default" || entries.length <= 1) {
    return entries;
  }

  if (rankBy === "interestingness") {
    const scores = overallInterestingnessScore(entries);
    return [...entries].sort((a, b) => {
      const sa = scores.get(a.entryId) ?? 0;
      const sb = scores.get(b.entryId) ?? 0;
      if (sb !== sa) return sb - sa;
      return a.entryId.localeCompare(b.entryId);
    });
  }

  const ascending =
    rankBy === "hard_to_fit" || rankBy === "few_stages" || rankBy === "small_variation";

  const metric =
    rankBy === "hard_to_fit" || rankBy === "easy_to_fit"
      ? meanSegmentR2
      : rankBy === "few_stages" || rankBy === "many_stages"
        ? stageCount
        : relativeVariation;

  const order = sortIndicesByMetric(entries, metric, ascending);
  return order.map((i) => entries[i]!);
}

export const CURVE_STARER_RANK_BY_OPTIONS: { id: CurveStarerRankBy; label: string }[] = [
  { id: "default", label: "Default" },
  { id: "hard_to_fit", label: "Hard to fit (low R²)" },
  { id: "easy_to_fit", label: "Easy to fit (high R²)" },
  { id: "few_stages", label: "Few stages" },
  { id: "many_stages", label: "Many stages" },
  { id: "small_variation", label: "Small variation / conserved" },
  { id: "large_variation", label: "Large variation / unstable" },
  { id: "interestingness", label: "Overall interestingness" },
];

export function isCurveStarerRankBy(value: string): value is CurveStarerRankBy {
  return CURVE_STARER_RANK_BY_OPTIONS.some((o) => o.id === value);
}
