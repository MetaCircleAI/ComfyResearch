import type { CurvePoint } from "./observableCurvePayload";
import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import { normalizedBoundariesForEntry } from "./commonBoundaryClusters";

export type TargetObjective = "higher" | "lower";

export type TargetTransitionInterval = {
  tStart: number;
  tEnd: number;
  segmentIndex: number | null;
};

export type TargetPhaseAnalysis = {
  entryId: string;
  label: string;
  objective: TargetObjective;
  threshold: number;
  crossingStep: number | null;
  normalizedTransition: number | null;
  transitionInterval: TargetTransitionInterval | null;
  peakValue: number;
  finalValue: number;
};

function sortedPoints(points: CurvePoint[]): CurvePoint[] {
  return points
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.loss))
    .slice()
    .sort((a, b) => a.t - b.t);
}

/** Logged training steps — not LPD resampling (fit curves can shift threshold crossings). */
export function measuredCurvePoints(entry: CurveStarerAnalyzedEntry): CurvePoint[] {
  return sortedPoints(entry.points);
}

function stepRange(points: CurvePoint[]): { minT: number; maxT: number } | null {
  if (points.length === 0) return null;
  return { minT: points[0]!.t, maxT: points[points.length - 1]!.t };
}

export function firstThresholdCrossingStep(
  points: CurvePoint[],
  objective: TargetObjective,
  threshold: number,
): number | null {
  if (!Number.isFinite(threshold)) return null;
  for (const p of sortedPoints(points)) {
    if (objective === "higher" && p.loss >= threshold) return p.t;
    if (objective === "lower" && p.loss <= threshold) return p.t;
  }
  return null;
}

function peakForObjective(values: number[], objective: TargetObjective): number {
  if (values.length === 0) return NaN;
  return objective === "higher" ? Math.max(...values) : Math.min(...values);
}

function fallbackNormalizedTransition(entry: CurveStarerAnalyzedEntry): number | null {
  const bps = normalizedBoundariesForEntry(entry);
  if (bps.length === 0) return null;
  return bps[bps.length - 1] ?? null;
}

function segmentContainingStep(
  entry: CurveStarerAnalyzedEntry,
  step: number,
): { tStart: number; tEnd: number; segmentIndex: number } | null {
  const segments = entry.lpd?.segments ?? [];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (step >= seg.t_start && step <= seg.t_end) {
      return { tStart: seg.t_start, tEnd: seg.t_end, segmentIndex: i };
    }
  }
  return null;
}

function largestDeltaSegment(
  entry: CurveStarerAnalyzedEntry,
): { tStart: number; tEnd: number; segmentIndex: number } | null {
  const points = sortedPoints(entry.lpd?.data ?? entry.points);
  const segments = entry.lpd?.segments ?? [];
  if (segments.length === 0 || points.length < 2) return null;

  let bestIdx = 0;
  let bestDelta = -Infinity;
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    const inSeg = points.filter((p) => p.t >= seg.t_start && p.t <= seg.t_end);
    if (inSeg.length < 2) continue;
    const delta = Math.abs(inSeg[inSeg.length - 1]!.loss - inSeg[0]!.loss);
    if (delta > bestDelta) {
      bestDelta = delta;
      bestIdx = i;
    }
  }
  const seg = segments[bestIdx]!;
  return { tStart: seg.t_start, tEnd: seg.t_end, segmentIndex: bestIdx };
}

export function findTargetTransitionInterval(
  entry: CurveStarerAnalyzedEntry,
  crossingStep: number | null,
): TargetTransitionInterval | null {
  if (crossingStep != null) {
    const hit = segmentContainingStep(entry, crossingStep);
    if (hit) return hit;
  }

  const segments = entry.lpd?.segments ?? [];
  if (segments.length >= 2) {
    const penultimate = segments[segments.length - 2]!;
    return {
      tStart: penultimate.t_start,
      tEnd: penultimate.t_end,
      segmentIndex: segments.length - 2,
    };
  }

  const byDelta = largestDeltaSegment(entry);
  if (byDelta) return byDelta;

  const points = sortedPoints(entry.lpd?.data ?? entry.points);
  const range = stepRange(points);
  if (!range || range.maxT <= range.minT) return null;
  const norm = fallbackNormalizedTransition(entry);
  if (norm == null) return null;
  const span = range.maxT - range.minT;
  const center = range.minT + norm * span;
  const halfWindow = span * 0.08;
  return {
    tStart: Math.max(range.minT, center - halfWindow),
    tEnd: Math.min(range.maxT, center + halfWindow),
    segmentIndex: null,
  };
}

function deltaEpsilon(deltas: number[]): number {
  const abs = deltas.map(Math.abs).filter(Number.isFinite);
  if (abs.length === 0) return 1e-12;
  const peak = Math.max(...abs);
  return Math.max(1e-12, peak * 1e-6);
}

export type TargetMonotonicPhase = {
  tStart: number;
  tEnd: number;
  /** +1 rising, −1 falling */
  direction: 1 | -1;
  /** How monotonic the target is within [tStart, tEnd] (0–1). */
  targetMonotonicity: number;
  /** |Δtarget| across the window. */
  netChange: number;
};

/** Fraction of step deltas that agree with the overall trend (flat steps count as OK). */
export function monotonicityScore(values: number[]): number {
  if (values.length < 3) return 0;
  const net = values[values.length - 1]! - values[0]!;
  const direction = Math.sign(net);
  if (direction === 0) return 0;

  const deltas: number[] = [];
  for (let i = 1; i < values.length; i++) deltas.push(values[i]! - values[i - 1]!);
  const eps = deltaEpsilon(deltas);
  let agree = 0;
  for (const d of deltas) {
    if (Math.abs(d) <= eps) agree += 1;
    else if (Math.sign(d) === direction) agree += 1;
  }
  return agree / deltas.length;
}

function totalAbsDelta(values: number[]): number {
  let sum = 0;
  for (let i = 1; i < values.length; i++) sum += Math.abs(values[i]! - values[i - 1]!);
  return sum;
}

const MIN_TARGET_PHASE_MONOTONICITY = 0.55;

/** Rising when objective is higher; falling when lower. */
export function phaseDirectionForObjective(objective: TargetObjective): 1 | -1 {
  return objective === "higher" ? 1 : -1;
}

/**
 * Pick the contiguous step window where the target clearly rises or falls per objective:
 * among windows with target monotonicity ≥ threshold and matching direction, maximize
 * total |Δ| × mono / √length so flat tails do not inflate the window.
 */
export function findDominantTargetMonotonicPhase(
  points: CurvePoint[],
  objective: TargetObjective,
): TargetMonotonicPhase | null {
  const pts = sortedPoints(points);
  if (pts.length < 3) return null;

  const wantedDirection = phaseDirectionForObjective(objective);
  let best: TargetMonotonicPhase | null = null;
  let bestMerit = -Infinity;

  for (let i = 0; i < pts.length - 2; i++) {
    for (let j = i + 2; j < pts.length; j++) {
      const slice = pts.slice(i, j + 1);
      const ys = slice.map((p) => p.loss);
      const net = ys[ys.length - 1]! - ys[0]!;
      const direction = Math.sign(net);
      if (direction === 0 || direction !== wantedDirection) continue;

      const mono = monotonicityScore(ys);
      if (mono < MIN_TARGET_PHASE_MONOTONICITY) continue;

      const stepCount = ys.length - 1;
      const merit = (totalAbsDelta(ys) * mono) / Math.sqrt(stepCount);
      if (merit > bestMerit) {
        bestMerit = merit;
        best = {
          tStart: slice[0]!.t,
          tEnd: slice[slice.length - 1]!.t,
          direction: direction as 1 | -1,
          targetMonotonicity: mono,
          netChange: net,
        };
      }
    }
  }
  return best;
}

export function analyzeTargetPhase(
  entry: CurveStarerAnalyzedEntry,
  objective: TargetObjective,
  threshold: number,
): TargetPhaseAnalysis {
  const measured = measuredCurvePoints(entry);
  const ys = measured.map((p) => p.loss).filter(Number.isFinite);
  const crossingStep = firstThresholdCrossingStep(measured, objective, threshold);
  const range = stepRange(measured);
  let normalizedTransition: number | null = null;
  if (crossingStep != null && range && range.maxT > range.minT) {
    normalizedTransition = (crossingStep - range.minT) / (range.maxT - range.minT);
  } else if (crossingStep == null) {
    normalizedTransition = fallbackNormalizedTransition(entry);
  }

  const transitionInterval = findTargetTransitionInterval(entry, crossingStep);

  return {
    entryId: entry.entryId,
    label: entry.label,
    objective,
    threshold,
    crossingStep,
    normalizedTransition,
    transitionInterval,
    peakValue: peakForObjective(ys, objective),
    finalValue: ys.length ? ys[ys.length - 1]! : NaN,
  };
}
