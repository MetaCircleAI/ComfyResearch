import type { ObservableCurveEntry } from "./collectObservableCurves";
import { curveStarerScenarioFingerprint } from "./collectObservableCurves";
import type { CurveStarerAnalyzedEntry } from "./lpdTypes";

/** Bump when LPD request defaults change so stale analyses are not reused. */
const LPD_CACHE_VERSION = "K3_auto_k6_target0.99_simp0.25";

type CurveStarerCacheSnapshot = {
  fingerprint: string;
  entries: CurveStarerAnalyzedEntry[];
};

let cache: CurveStarerCacheSnapshot | null = null;

export function curveStarerCacheFingerprint(curves: ObservableCurveEntry[]): string {
  return `${LPD_CACHE_VERSION}\n${curveStarerScenarioFingerprint(curves)}`;
}

export function getCurveStarerCachedEntries(
  curves: ObservableCurveEntry[],
): CurveStarerAnalyzedEntry[] | null {
  if (!cache) return null;
  const fingerprint = curveStarerCacheFingerprint(curves);
  if (cache.fingerprint !== fingerprint) return null;

  const byEntryId = new Map(cache.entries.map((entry) => [entry.entryId, entry]));
  const merged: CurveStarerAnalyzedEntry[] = [];
  for (const curve of curves) {
    const hit = byEntryId.get(curve.entryId);
    if (!hit) return null;
    merged.push({
      ...hit,
      ...curve,
      lpd: hit.lpd,
      lpdError: hit.lpdError,
    });
  }
  return merged;
}

export function setCurveStarerCache(
  curves: ObservableCurveEntry[],
  entries: CurveStarerAnalyzedEntry[],
): void {
  cache = {
    fingerprint: curveStarerCacheFingerprint(curves),
    entries: entries.map((entry) => ({
      ...entry,
      points: entry.points.map((p) => ({ ...p })),
      lpd: entry.lpd
        ? {
            ...entry.lpd,
            data: entry.lpd.data?.map((p) => ({ ...p })),
            fitted: entry.lpd.fitted?.map((p) => ({ ...p })),
            segments: entry.lpd.segments?.map((s) => ({ ...s })),
            breakpoints: entry.lpd.breakpoints ? [...entry.lpd.breakpoints] : undefined,
          }
        : null,
    })),
  };
}
