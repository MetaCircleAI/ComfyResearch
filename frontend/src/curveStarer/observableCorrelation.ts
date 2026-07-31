import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import type {
  DirectionRelation,
  ObservableSourceKind,
  RelatedObservableMatch,
  TrickTypeCategory,
} from "./speedUpTrickTypes";
import { labelContainsWildcard } from "./curveStarerLabels";
import { inferTrickKindsForEntry, supportsProjectionForEntry } from "./speedUpTrickRegistry";
import {
  findDominantTargetMonotonicPhase,
  monotonicityScore,
  type TargetMonotonicPhase,
  type TargetObjective,
} from "./targetPhaseTransition";
import type { CurvePoint } from "./observableCurvePayload";

export const MIN_CORRELATION_SCORE = 0.15;

const REDUCTION_OPS = [
  "l2_norm",
  "l1_norm",
  "mean",
  "median",
  "max",
  "min",
  "std",
  "entropy",
] as const;

export function parseReductionOpsFromLabel(label: string): string[] {
  const lower = label.toLowerCase();
  const out: string[] = [];
  for (const op of REDUCTION_OPS) {
    if (lower.includes(op)) out.push(op);
  }
  return out;
}

function sortedPoints(points: CurvePoint[]): CurvePoint[] {
  return points
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.loss))
    .slice()
    .sort((a, b) => a.t - b.t);
}

type AlignedStep = { t: number; targetY: number; obsY: number };

function alignCurvesByStep(targetPoints: CurvePoint[], obsPoints: CurvePoint[]): AlignedStep[] {
  const targetByT = new Map<number, number>();
  for (const p of sortedPoints(targetPoints)) targetByT.set(p.t, p.loss);
  const out: AlignedStep[] = [];
  for (const p of sortedPoints(obsPoints)) {
    const targetY = targetByT.get(p.t);
    if (targetY != null && Number.isFinite(targetY)) {
      out.push({ t: p.t, targetY, obsY: p.loss });
    }
  }
  return out;
}

function alignedInPhase(aligned: AlignedStep[], phase: TargetMonotonicPhase): AlignedStep[] {
  return aligned.filter((p) => p.t >= phase.tStart && p.t <= phase.tEnd);
}

/**
 * Ranking R²: observable monotonicity within the target's dominant rising/falling window.
 * Roughly monotone → ~1; many reversals → low.
 */
export function rankingRSquaredInPhase(
  targetPoints: CurvePoint[],
  obsPoints: CurvePoint[],
  phase: TargetMonotonicPhase,
): number {
  const aligned = alignCurvesByStep(targetPoints, obsPoints);
  const slice = alignedInPhase(aligned, phase);
  if (slice.length < 3) return 0;
  return monotonicityScore(slice.map((p) => p.obsY));
}

export function findTargetMonotonicPhaseForEntry(
  targetEntry: CurveStarerAnalyzedEntry,
  objective: TargetObjective,
): TargetMonotonicPhase | null {
  return findDominantTargetMonotonicPhase(targetEntry.lpd?.data ?? targetEntry.points, objective);
}

function valueDeltaInPhase(aligned: AlignedStep[], phase: TargetMonotonicPhase): number {
  const slice = alignedInPhase(aligned, phase);
  if (slice.length < 2) return 0;
  return slice[slice.length - 1]!.obsY - slice[0]!.obsY;
}

function lastObsYAligned(aligned: AlignedStep[]): number {
  if (aligned.length === 0) return NaN;
  return aligned[aligned.length - 1]!.obsY;
}

function inferSource(entry: CurveStarerAnalyzedEntry): ObservableSourceKind {
  const label = entry.label.toLowerCase();
  if (
    label.includes("weight l2") ||
    label.includes("weight l1") ||
    label.includes("gradient") ||
    label.includes("accuracy") ||
    label.includes("loss")
  ) {
    return "built_in";
  }
  if (parseReductionOpsFromLabel(entry.label).length > 0) return "algebra_user";
  return "unknown";
}

function defaultTrickCategory(_entry: CurveStarerAnalyzedEntry): TrickTypeCategory {
  return "reg";
}

export function correlateObservables(
  targetEntry: CurveStarerAnalyzedEntry,
  entries: CurveStarerAnalyzedEntry[],
  objective: TargetObjective,
): RelatedObservableMatch[] {
  const targetPts = sortedPoints(targetEntry.lpd?.data ?? targetEntry.points);
  if (targetPts.length < 2) return [];

  const phase = findDominantTargetMonotonicPhase(targetPts, objective);
  if (!phase) return [];

  const memberCountByViz = new Map<string, number>();
  for (const entry of entries) {
    if (!entry.entryId.includes(":member:")) continue;
    const vizId = entry.entryId.split(":")[0] ?? "";
    if (!vizId) continue;
    memberCountByViz.set(vizId, (memberCountByViz.get(vizId) ?? 0) + 1);
  }

  const out: RelatedObservableMatch[] = [];
  for (const entry of entries) {
    if (entry.entryId === targetEntry.entryId) continue;

    const vizId = entry.entryId.split(":")[0] ?? "";
    if (
      labelContainsWildcard(entry.label) &&
      !entry.entryId.includes(":member:") &&
      (memberCountByViz.get(vizId) ?? 0) > 0
    ) {
      continue;
    }

    const obsPts = sortedPoints(entry.lpd?.data ?? entry.points);
    const aligned = alignCurvesByStep(targetPts, obsPts);
    const correlationScore = rankingRSquaredInPhase(targetPts, obsPts, phase);
    if (correlationScore < MIN_CORRELATION_SCORE) continue;

    const candDelta = valueDeltaInPhase(aligned, phase);
    let directionRelation: DirectionRelation = "unknown";
    if (phase.netChange !== 0 && candDelta !== 0) {
      directionRelation = Math.sign(phase.netChange) === Math.sign(candDelta) ? "same" : "opposite";
    }

    const reductionOps = parseReductionOpsFromLabel(entry.label);
    const trickKinds = inferTrickKindsForEntry(entry);
    const supportsProjection = supportsProjectionForEntry(entry);
    const shellValue = lastObsYAligned(aligned);

    out.push({
      entryId: entry.entryId,
      entry,
      label: entry.label,
      correlationScore,
      alignmentScore: correlationScore,
      directionRelation,
      shellValue,
      source: inferSource(entry),
      reductionOps,
      hasAutomatedTrick: trickKinds.length > 0,
      trickKinds,
      defaultTrickCategory: defaultTrickCategory(entry),
      supportsProjection,
    });
  }

  return out.sort(
    (a, b) => b.correlationScore - a.correlationScore || a.label.localeCompare(b.label),
  );
}
