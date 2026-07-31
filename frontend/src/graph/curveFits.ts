/**
 * Shared curve-fit helpers (R^2, OLS, exponential fit y = A e^{-Bx} + C, power law y = A x^B + C).
 *
 * Extracted from RegressorNode.tsx so other analysis nodes (e.g. interestingness teller) can
 * reuse the exact same fits without duplicating the math.
 */

export type FitResult = {
  name: string;
  /** LaTeX source for display (e.g. `y = A\\,\\mathrm{e}^{-Bx} + C`). */
  latex: string;
  params: Record<string, number>;
  r2: number;
};

export function r2Score(y: number[], pred: number[]): number {
  if (y.length === 0 || pred.length !== y.length) return Number.NaN;
  const mean = y.reduce((a, b) => a + b, 0) / y.length;
  let sst = 0;
  let sse = 0;
  for (let i = 0; i < y.length; i++) {
    const d = y[i]! - mean;
    sst += d * d;
    const e = y[i]! - pred[i]!;
    sse += e * e;
  }
  if (sst <= 1e-12) return sse <= 1e-12 ? 1 : 0;
  return 1 - sse / sst;
}

export function fitLinearTwoVar(z: number[], y: number[]): { a: number; c: number; sse: number } | null {
  if (z.length !== y.length || z.length < 3) return null;
  let sz = 0;
  let sz2 = 0;
  let sy = 0;
  let szy = 0;
  for (let i = 0; i < z.length; i++) {
    const zi = z[i]!;
    const yi = y[i]!;
    sz += zi;
    sz2 += zi * zi;
    sy += yi;
    szy += zi * yi;
  }
  const n = z.length;
  const det = sz2 * n - sz * sz;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  const a = (szy * n - sy * sz) / det;
  const c = (sz2 * sy - sz * szy) / det;
  let sse = 0;
  for (let i = 0; i < z.length; i++) {
    const err = y[i]! - (a * z[i]! + c);
    sse += err * err;
  }
  return { a, c, sse };
}

export function fitExponential(x: number[], y: number[]): FitResult | null {
  const span = Math.max(1e-6, Math.max(...x) - Math.min(...x));
  const bCandidates: number[] = [0];
  const scales = [0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2.5, 4, 6, 9];
  for (const s of scales) bCandidates.push(s / span);
  let best: { A: number; B: number; C: number; sse: number; pred: number[] } | null = null;
  for (const B of bCandidates) {
    const z = x.map((xi) => Math.exp(-B * xi));
    const lin = fitLinearTwoVar(z, y);
    if (!lin) continue;
    const pred = z.map((zi) => lin.a * zi + lin.c);
    if (!best || lin.sse < best.sse) best = { A: lin.a, B, C: lin.c, sse: lin.sse, pred };
  }
  if (!best) return null;
  return {
    name: "Exponential",
    latex: String.raw`y = A\,\mathrm{e}^{-Bx} + C`,
    params: { A: best.A, B: best.B, C: best.C },
    r2: r2Score(y, best.pred),
  };
}

type PowerLawFitState = { A: number; B: number; C: number; sse: number; pred: number[] };

/** SSE-minimizing (A, C) for fixed exponent B in y = A x^B + C. */
function fitPowerLawAtFixedB(x: number[], y: number[], B: number): PowerLawFitState | null {
  const z: number[] = [];
  for (const xi of x) {
    const zi = Math.pow(xi, B);
    if (!Number.isFinite(zi)) return null;
    z.push(zi);
  }
  const lin = fitLinearTwoVar(z, y);
  if (!lin) return null;
  const A = lin.a;
  const C = lin.c;
  if (!Number.isFinite(A) || !Number.isFinite(C)) return null;
  const pred = z.map((zi) => A * zi + C);
  let sse = 0;
  for (let i = 0; i < y.length; i++) {
    const err = y[i]! - pred[i]!;
    sse += err * err;
  }
  return { A, B, C, sse, pred };
}

/**
 * y = A x^B + C. For each B on a grid, z_i = x_i^B is linear in (A, C), so we OLS-fit y = A z + C.
 * Works when x includes 0 (e.g. step 0): Math.pow(0, B) is 0 for B>0, 1 for B=0; B<0 yields non-finite z and is skipped.
 *
 * Coarse search uses ΔB = (bMax−bMin)/nSteps = 0.05; a refinement pass on ± one coarse step around the winner
 * removes the “B is always a multiple of 0.05” artifact in the returned fit.
 */
export function fitPowerLaw(x: number[], y: number[]): FitResult | null {
  if (x.length !== y.length || x.length < 3) return null;
  let best: PowerLawFitState | null = null;
  const bMin = -2.5;
  const bMax = 5;
  const nSteps = 150;
  const dBCoarse = (bMax - bMin) / nSteps;

  const consider = (B: number) => {
    const r = fitPowerLawAtFixedB(x, y, B);
    if (!r) return;
    if (!best || r.sse < best.sse) best = r;
  };

  for (let k = 0; k <= nSteps; k++) {
    consider(bMin + k * dBCoarse);
  }
  if (!best) return null;

  const lo = Math.max(bMin, best.B - dBCoarse);
  const hi = Math.min(bMax, best.B + dBCoarse);
  if (hi > lo + 1e-12) {
    const nRef = 80;
    for (let j = 0; j <= nRef; j++) {
      consider(lo + (j / nRef) * (hi - lo));
    }
  }

  return {
    name: "Power law",
    latex: String.raw`y = A x^{B} + C`,
    params: { A: best.A, B: best.B, C: best.C },
    r2: r2Score(y, best.pred),
  };
}

/** Points whose `steps[i]` lie in `[min(stepMin,stepMax), max(...)]` inclusive (same convention as law windows). */
export function sliceSeriesByStepInclusive(
  steps: number[],
  values: number[],
  stepMin: number,
  stepMax: number,
): { xs: number[]; ys: number[] } | null {
  if (steps.length !== values.length || steps.length < 3) return null;
  const lo = Math.min(stepMin, stepMax);
  const hi = Math.max(stepMin, stepMax);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]!;
    if (s >= lo && s <= hi) {
      xs.push(s);
      ys.push(values[i]!);
    }
  }
  if (xs.length < 3) return null;
  return { xs, ys };
}

/** Re-fit `y = A x^B + C` on the slice of the series inside the step span (for UI when the user edits the region). */
export function refitPowerLawForStepRegion(
  steps: number[],
  values: number[],
  stepMin: number,
  stepMax: number,
): FitResult | null {
  const sl = sliceSeriesByStepInclusive(steps, values, stepMin, stepMax);
  if (!sl) return null;
  return fitPowerLaw(sl.xs, sl.ys);
}
