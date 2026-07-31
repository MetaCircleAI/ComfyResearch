import type { CurvePoint } from "./observableCurvePayload";
import type { TargetObjective } from "./targetPhaseTransition";

export type TargetCurveCandidate = {
  entryId: string;
  label: string;
  points: CurvePoint[];
  lpd?: { data?: CurvePoint[] } | null;
};

export type SuggestedTargetDefaults = {
  entryId: string;
  objective: TargetObjective;
  threshold: number;
};

function sortedYs(points: CurvePoint[]): number[] {
  return points
    .filter((p) => Number.isFinite(p.loss))
    .slice()
    .sort((a, b) => a.t - b.t)
    .map((p) => p.loss);
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx]!;
}

function labelHintsAccuracy(label: string): boolean {
  const s = label.trim().toLowerCase();
  return s.includes("accuracy") || /\bacc\b/.test(s);
}

function labelHintsLoss(label: string): boolean {
  return label.trim().toLowerCase().includes("loss");
}

function suggestThreshold(entry: TargetCurveCandidate, objective: TargetObjective): number {
  const ys = sortedYs(entry.lpd?.data ?? entry.points);
  if (ys.length === 0) return objective === "higher" ? 1 : 0;
  if (labelHintsAccuracy(entry.label)) return 0.95;
  if (labelHintsLoss(entry.label)) {
    const head = ys.slice(0, Math.max(1, Math.ceil(ys.length * 0.2)));
    const med = head.reduce((a, b) => a + b, 0) / head.length;
    return objective === "lower" ? med : med;
  }
  return objective === "higher" ? percentile(ys, 0.8) : percentile(ys, 0.2);
}

export function suggestDefaultTarget(entries: TargetCurveCandidate[]): SuggestedTargetDefaults | null {
  if (entries.length === 0) return null;
  const accTest = entries.find(
    (e) => labelHintsAccuracy(e.label) && e.label.toLowerCase().includes("test"),
  );
  const accAny = entries.find((e) => labelHintsAccuracy(e.label));
  const lossTest = entries.find(
    (e) => labelHintsLoss(e.label) && e.label.toLowerCase().includes("test"),
  );
  const pick = accTest ?? accAny ?? lossTest ?? entries[0]!;
  const objective: TargetObjective =
    labelHintsLoss(pick.label) && !labelHintsAccuracy(pick.label) ? "lower" : "higher";
  return {
    entryId: pick.entryId,
    objective,
    threshold: suggestThreshold(pick, objective),
  };
}
