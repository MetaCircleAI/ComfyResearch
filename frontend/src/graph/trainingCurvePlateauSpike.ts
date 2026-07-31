/**
 * Plateau / spike / phase-transition helpers for training curves (same Y units as the Training viz chart:
 * pass loss or perplexity = exp(loss) per point).
 */

export type StepValueSpan = { stepMin: number; stepMax: number };

const STEP_EPS = 1e-12;

/** Map raw stored train loss to chart Y (matches TrainingVisualizationNode). */
export function rawTrainToPlotY(raw: number, yPlotMetric: "loss" | "perplexity"): number {
  return yPlotMetric === "perplexity" ? Math.exp(raw) : raw;
}

export function buildTrainPlotYSeries(rawLoss: number[], yPlotMetric: "loss" | "perplexity"): number[] {
  return rawLoss.map((v) => rawTrainToPlotY(v, yPlotMetric));
}

function validSeries(steps: number[], values: number[]): boolean {
  if (steps.length < 3 || steps.length !== values.length) return false;
  for (let i = 0; i < steps.length; i++) {
    if (!Number.isFinite(steps[i]!) || !Number.isFinite(values[i]!)) return false;
  }
  return true;
}

function baselineSlope(steps: number[], values: number[]): number {
  if (steps.length < 2 || values.length < 2) return 0;
  const dt = steps[steps.length - 1]! - steps[0]!;
  if (!Number.isFinite(dt) || Math.abs(dt) < STEP_EPS) return 0;
  const dy = values[0]! - values[values.length - 1]!;
  if (!Number.isFinite(dy)) return 0;
  return Math.abs(dy / dt);
}

/**
 * Plateau: maximal contiguous runs where each segment's |Δy/Δstep| ≤ `maxAbsSlope`.
 * Requires at least two flat segments in a row (three sample points).
 */
export function detectPlateauRegions(
  steps: number[],
  values: number[],
  maxAbsSlope: number,
): { regions: StepValueSpan[] } {
  if (!validSeries(steps, values) || !Number.isFinite(maxAbsSlope) || maxAbsSlope < 0) {
    return { regions: [] };
  }

  const n = steps.length;
  /** Segment k connects point k-1 → k (k = 1..n-1). */
  const segFlat: boolean[] = new Array(n);
  segFlat[0] = false;
  for (let k = 1; k < n; k++) {
    const ds = steps[k]! - steps[k - 1]!;
    const dv = values[k]! - values[k - 1]!;
    const slope = dv / (Math.abs(ds) < STEP_EPS ? STEP_EPS : ds);
    segFlat[k] = Math.abs(slope) <= maxAbsSlope;
  }

  const regions: StepValueSpan[] = [];
  let runStart = -1;
  for (let k = 1; k < n; k++) {
    if (segFlat[k]) {
      if (runStart < 0) runStart = k;
    } else {
      if (runStart >= 0) {
        const runEnd = k - 1;
        if (runEnd - runStart + 1 >= 2) {
          const stepMin = steps[runStart - 1]!;
          const stepMax = steps[runEnd]!;
          if (stepMin <= stepMax) regions.push({ stepMin, stepMax });
        }
        runStart = -1;
      }
    }
  }
  if (runStart >= 0) {
    const runEnd = n - 1;
    if (runEnd - runStart + 1 >= 2) {
      const stepMin = steps[runStart - 1]!;
      const stepMax = steps[runEnd]!;
      if (stepMin <= stepMax) regions.push({ stepMin, stepMax });
    }
  }

  return { regions };
}

const BASELINE_LOOKBACK = 3;
const SPIKE_RECOVERY_WINDOW = 14;
const RECOVERY_FRACTION = 0.15;

/**
 * Spike: local peak whose relative rise slope is at least `minRelativeSlope`, then recovery
 * within `SPIKE_RECOVERY_WINDOW` steps to near pre-spike level (within RECOVERY_FRACTION of spike height).
 * Greedy left-to-right by end index to limit overlaps.
 */
export function detectSpikeRegions(
  steps: number[],
  values: number[],
  minRelativeSlope: number,
): { regions: StepValueSpan[] } {
  if (!validSeries(steps, values) || !Number.isFinite(minRelativeSlope) || minRelativeSlope <= 0) {
    return { regions: [] };
  }
  const baseSlope = baselineSlope(steps, values);
  if (baseSlope < STEP_EPS) return { regions: [] };

  const n = values.length;
  const regions: StepValueSpan[] = [];
  let lastEndIdx = -1;

  for (let i = 1; i < n - 1; i++) {
    if (i <= lastEndIdx) continue;

    const lo = Math.max(0, i - BASELINE_LOOKBACK);
    let baseline = values[i - 1]!;
    for (let t = lo; t < i; t++) baseline = Math.min(baseline, values[t]!);

    const peak = values[i]!;
    const rise = peak - baseline;
    if (!Number.isFinite(rise) || rise <= 0) continue;
    const dtRise = steps[i]! - steps[Math.max(0, i - 1)]!;
    const riseSlope = rise / (Math.abs(dtRise) < STEP_EPS ? STEP_EPS : dtRise);
    const relativeSlope = Math.abs(riseSlope) / baseSlope;
    if (relativeSlope < minRelativeSlope) continue;

    const recoveryTol = Math.max(Number.EPSILON * (Math.abs(baseline) + 1), RECOVERY_FRACTION * rise);
    const ceiling = baseline + recoveryTol;

    let j = -1;
    const jMax = Math.min(n - 1, i + SPIKE_RECOVERY_WINDOW);
    for (let t = i + 1; t <= jMax; t++) {
      if (values[t]! <= ceiling) {
        j = t;
        break;
      }
    }
    if (j < 0) continue;

    const stepMin = steps[i - 1]!;
    const stepMax = steps[j]!;
    if (stepMin > stepMax) continue;

    regions.push({ stepMin, stepMax });
    lastEndIdx = j;
  }

  return { regions };
}

/**
 * Phase transition: sharp loss drop (relative fall slope ≥ `minRelativeSlope` vs global |Δy/Δstep|),
 * then stabilization — first step where loss rebounds from the running minimum by RECOVERY_FRACTION of
 * the initial drop, or end of recovery window if the curve keeps falling. Greedy left-to-right by end index.
 */
export function detectPhaseTransitionRegions(
  steps: number[],
  values: number[],
  minRelativeSlope: number,
): { regions: StepValueSpan[] } {
  if (!validSeries(steps, values) || !Number.isFinite(minRelativeSlope) || minRelativeSlope <= 0) {
    return { regions: [] };
  }
  const baseSlope = baselineSlope(steps, values);
  if (baseSlope < STEP_EPS) return { regions: [] };

  const n = values.length;
  const regions: StepValueSpan[] = [];
  let lastEndIdx = -1;

  for (let i = 1; i < n - 1; i++) {
    if (i <= lastEndIdx) continue;

    const lo = Math.max(0, i - BASELINE_LOOKBACK);
    let baselineHigh = values[i - 1]!;
    for (let t = lo; t < i; t++) baselineHigh = Math.max(baselineHigh, values[t]!);

    const trough = values[i]!;
    const drop = baselineHigh - trough;
    if (!Number.isFinite(drop) || drop <= 0) continue;
    const dtFall = steps[i]! - steps[Math.max(0, i - 1)]!;
    const fallSlope = drop / (Math.abs(dtFall) < STEP_EPS ? STEP_EPS : dtFall);
    const relativeSlope = fallSlope / baseSlope;
    if (relativeSlope < minRelativeSlope) continue;

    const recoveryTol = Math.max(Number.EPSILON * (Math.abs(trough) + 1), RECOVERY_FRACTION * drop);

    let runningMin = trough;
    let j = -1;
    const jMax = Math.min(n - 1, i + SPIKE_RECOVERY_WINDOW);
    for (let t = i + 1; t <= jMax; t++) {
      const v = values[t]!;
      if (v < runningMin) {
        runningMin = v;
        continue;
      }
      if (v >= runningMin + recoveryTol) {
        j = t;
        break;
      }
    }
    if (j < 0) j = jMax;

    const stepMin = steps[i - 1]!;
    const stepMax = steps[j]!;
    if (stepMin > stepMax) continue;

    regions.push({ stepMin, stepMax });
    lastEndIdx = j;
  }

  return { regions };
}
