import type { CurveStarerAnalyzedEntry } from "./lpdTypes";

export type CommonBoundaryCluster = {
  /** Normalized position in [0, 1] along the curve's step axis. */
  center: number;
  /** Fraction of analyzed curves that contain a boundary near this cluster. */
  strength: number;
  entryCount: number;
  totalEntries: number;
};

const DEFAULT_EPS = 0.05;

function tRangeForEntry(entry: CurveStarerAnalyzedEntry): { minT: number; maxT: number } | null {
  const points = entry.lpd?.data ?? entry.points;
  let minT = Infinity;
  let maxT = -Infinity;
  for (const p of points) {
    if (!Number.isFinite(p.t)) continue;
    minT = Math.min(minT, p.t);
    maxT = Math.max(maxT, p.t);
  }
  if (!Number.isFinite(minT) || !Number.isFinite(maxT) || maxT <= minT) return null;
  return { minT, maxT };
}

/** Internal phase boundaries normalized to [0, 1] for cross-curve comparison. */
export function normalizedBoundariesForEntry(entry: CurveStarerAnalyzedEntry): number[] {
  const range = tRangeForEntry(entry);
  if (!range) return [];
  const span = range.maxT - range.minT;
  const raw: number[] = [];

  const bps = entry.lpd?.breakpoints;
  if (bps && bps.length > 0) {
    for (const bp of bps) {
      if (Number.isFinite(bp)) raw.push((bp - range.minT) / span);
    }
  } else {
    const segments = entry.lpd?.segments ?? [];
    for (let i = 0; i < segments.length - 1; i++) {
      const tEnd = segments[i]?.t_end;
      if (Number.isFinite(tEnd)) raw.push((tEnd - range.minT) / span);
    }
  }

  return raw.filter((n) => n > 0.01 && n < 0.99);
}

function clusterCenter(positions: number[]): number {
  return positions.reduce((sum, p) => sum + p, 0) / positions.length;
}

/**
 * Greedy 1D clustering of normalized boundary positions across curves.
 * Returns clusters shared by at least two curves, sorted by strength descending.
 */
export function computeCommonBoundaryClusters(
  entries: CurveStarerAnalyzedEntry[],
  eps = DEFAULT_EPS,
): CommonBoundaryCluster[] {
  if (entries.length < 2) return [];

  type Point = { pos: number; entryId: string };
  const points: Point[] = [];
  for (const entry of entries) {
    for (const pos of normalizedBoundariesForEntry(entry)) {
      points.push({ pos, entryId: entry.entryId });
    }
  }
  if (points.length === 0) return [];

  points.sort((a, b) => a.pos - b.pos);

  const clusters: { positions: number[]; entryIds: Set<string> }[] = [];
  for (const pt of points) {
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      const center = clusterCenter(clusters[i]!.positions);
      const dist = Math.abs(pt.pos - center);
      if (dist <= eps && dist < bestDist) {
        bestIdx = i;
        bestDist = dist;
      }
    }
    if (bestIdx >= 0) {
      const cl = clusters[bestIdx]!;
      cl.positions.push(pt.pos);
      cl.entryIds.add(pt.entryId);
    } else {
      clusters.push({ positions: [pt.pos], entryIds: new Set([pt.entryId]) });
    }
  }

  const totalEntries = entries.length;
  return clusters
    .map((cl) => ({
      center: clusterCenter(cl.positions),
      entryCount: cl.entryIds.size,
      totalEntries,
      strength: cl.entryIds.size / totalEntries,
    }))
    .filter((c) => c.entryCount >= 2)
    .sort((a, b) => b.strength - a.strength || a.center - b.center);
}

export function matchCommonBoundaryStrength(
  normalizedPos: number,
  clusters: CommonBoundaryCluster[],
  eps = DEFAULT_EPS,
): number {
  let best = 0;
  for (const cl of clusters) {
    if (Math.abs(normalizedPos - cl.center) <= eps) {
      best = Math.max(best, cl.strength);
    }
  }
  return best;
}

/** Golden highlight style — stronger common-boundary signal → brighter gold. */
export function commonBoundaryGoldStyle(strength: number): {
  stroke: string;
  strokeWidth: number;
  strokeOpacity: number;
  filter?: string;
} {
  const t = Math.max(0, Math.min(1, strength));
  const r = Math.round(168 + t * 87);
  const g = Math.round(128 + t * 87);
  const b = Math.round(32 + t * 8);
  return {
    stroke: `rgb(${r}, ${g}, ${b})`,
    strokeWidth: 1.2 + t * 2.2,
    strokeOpacity: 0.5 + t * 0.5,
    filter: t >= 0.6 ? "drop-shadow(0 0 3px rgba(255, 215, 0, 0.55))" : undefined,
  };
}
