/** Shared layout and math for Training viz / Observable scalar line charts and Curve annotator mirror. */

export const SL_CHART_W = 232;
export const SL_CHART_H = 122;
export const SL_PAD_L = 36;
export const SL_PAD_R = 8;
export const SL_PAD_T = 10;
export const SL_PAD_B = 32;
export const SL_TICK_LEN = 4;
export const SL_Y_LABEL_X = 9;
export const SL_LOG_Y_FLOOR = 1e-15;

export type SLBounds = { minX: number; maxX: number; minY: number; maxY: number };

/**
 * Expand plot bounds on both sides so markers/line endpoints do not sit on clip borders.
 * `padX`/`padY` are per-side fractions of span.
 */
export function slPadBounds(b: SLBounds, padX = 0.055, padY = 0.055): SLBounds {
  if (!Number.isFinite(b.minX) || !Number.isFinite(b.maxX) || !Number.isFinite(b.minY) || !Number.isFinite(b.maxY)) {
    return b;
  }
  const spanX = b.maxX - b.minX;
  const spanY = b.maxY - b.minY;
  const bumpX = (Number.isFinite(spanX) && spanX > 0 ? spanX : Math.max(Math.abs(b.maxX), Math.abs(b.minX), 1)) * padX;
  const bumpY = (Number.isFinite(spanY) && spanY > 0 ? spanY : Math.max(Math.abs(b.maxY), Math.abs(b.minY), 1)) * padY;
  return {
    minX: b.minX - bumpX,
    maxX: b.maxX + bumpX,
    minY: b.minY - bumpY,
    maxY: b.maxY + bumpY,
  };
}

export function slTransformStep(step: number, logX: boolean): number {
  return logX ? Math.log10(Math.max(0, step) + 1) : step;
}

/** Inverse of `slTransformStep`: plot-x → raw step (>= 0). */
export function slStepFromPlotX(tx: number, logX: boolean): number {
  if (logX) {
    return Math.max(0, 10 ** tx - 1);
  }
  return tx;
}

/** Plot-space x for scalar charts: linear step, log10(step+1), or log10(raw) when ``plainLogX``. */
export function slScalarPlotXFromRaw(raw: number, logX: boolean, plainLogX: boolean): number {
  if (!logX) return raw;
  if (plainLogX) return Math.log10(Math.max(raw, 1e-15));
  return slTransformStep(raw, true);
}

/** Inverse of ``slScalarPlotXFromRaw`` (raw sweep / step on x). */
export function slScalarRawXFromPlotX(tx: number, logX: boolean, plainLogX: boolean): number {
  if (!logX) return tx;
  if (plainLogX) return 10 ** tx;
  return slStepFromPlotX(tx, true);
}

/**
 * Sort (x, y) pairs by x ascending. Sweep / flat-line previews often arrive in arbitrary leaf order;
 * drawing segments in that order produces long diagonal “spikes” across the plot.
 */
export function slPairsSortedByX(steps: number[], values: number[]): { steps: number[]; values: number[] } {
  if (steps.length !== values.length || steps.length < 2) return { steps, values };
  const pairs = steps
    .map((s, i) => ({ s, v: values[i]! }))
    .filter((p) => Number.isFinite(p.s) && Number.isFinite(p.v));
  if (pairs.length < 2) {
    return { steps: pairs.map((p) => p.s), values: pairs.map((p) => p.v) };
  }
  pairs.sort((a, b) => a.s - b.s);
  const sortedSteps: number[] = [];
  const sortedValues: number[] = [];
  let currX = pairs[0]!.s;
  let sumY = pairs[0]!.v;
  let count = 1;
  const flush = () => {
    sortedSteps.push(currX);
    sortedValues.push(sumY / count);
  };
  for (let i = 1; i < pairs.length; i++) {
    const next = pairs[i]!;
    const eps = Math.max(1e-12, Math.abs(currX) * 1e-12, Math.abs(next.s) * 1e-12);
    if (Math.abs(next.s - currX) <= eps) {
      sumY += next.v;
      count += 1;
      continue;
    }
    flush();
    currX = next.s;
    sumY = next.v;
    count = 1;
  }
  flush();
  return { steps: sortedSteps, values: sortedValues };
}

export function slTransformLoss(v: number, logY: boolean): number {
  return logY ? Math.log10(Math.max(v, SL_LOG_Y_FLOOR)) : v;
}

export function slRawLossToPlotY(raw: number, yPlotMetric: "loss" | "perplexity"): number {
  if (yPlotMetric === "perplexity") return Math.exp(raw);
  return raw;
}

export function slNiceStep(span: number): number {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const exp = Math.floor(Math.log10(span));
  const f = span / 10 ** exp;
  let nf = 1;
  if (f < 1.5) nf = 1;
  else if (f < 3) nf = 2;
  else if (f < 7) nf = 5;
  else nf = 10;
  return nf * 10 ** exp;
}

export function slGenerateTicks(min: number, max: number, maxTicks = 5): number[] {
  const span = max - min;
  if (span <= 0 || !Number.isFinite(span)) return [min];
  const step = slNiceStep(span / Math.max(1, maxTicks - 1));
  const start = Math.ceil(min / step - 1e-9) * step;
  const out: number[] = [];
  for (let t = start; t <= max + step * 1e-9; t += step) {
    if (t >= min - 1e-9 && t <= max + 1e-9) out.push(t);
    if (out.length > 12) break;
  }
  if (out.length === 0) return [min, max];
  return out;
}

function slUniqueSortedInRange(values: number[], lo: number, hi: number): number[] {
  return Array.from(
    new Set(values.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9)),
  ).sort((a, b) => a - b);
}

/**
 * Raw step counts for log10(step+1) x-axis: prefer multiples of 10 once the span is large,
 * instead of evenly spacing in transform space (which yields 99, 315, …).
 */
function slGenerateRawStepTicksForLogStepAxis(xMinRaw: number, xMaxRaw: number, maxTicks: number): number[] {
  const span = xMaxRaw - xMinRaw;
  if (span <= 0 || !Number.isFinite(span)) return [xMinRaw];
  let step = slNiceStep(span / Math.max(1, maxTicks - 1));
  if (xMaxRaw >= 100 && step >= 10) {
    step = Math.max(10, Math.round(step / 10) * 10);
    if (step > span) step = slNiceStep(span);
  } else if (xMaxRaw >= 30 && step >= 10) {
    const r = Math.round(step / 10) * 10;
    if (r >= 10 && r <= span) step = r;
  }
  if (step > span) step = slNiceStep(span);
  const start = Math.ceil(xMinRaw / step - 1e-9) * step;
  const out: number[] = [];
  for (let x = start; x <= xMaxRaw + step * 1e-9; x += step) {
    if (x >= xMinRaw - 1e-9 && x <= xMaxRaw + 1e-9) out.push(x);
    if (out.length > 12) break;
  }
  if (out.length === 0) return [xMinRaw, xMaxRaw];
  return out;
}

function slGenerateRawYTicksForLogValueAxis(yMinRaw: number, yMaxRaw: number, maxTicks: number): number[] {
  const ya = Math.max(yMinRaw, SL_LOG_Y_FLOOR);
  const yb = Math.max(ya * (1 + 1e-15), yMaxRaw);
  const lo = Math.log10(ya);
  const hi = Math.log10(yb);
  const kStart = Math.floor(lo - 1e-12);
  const kEnd = Math.ceil(hi + 1e-12);
  const rawCandidates: number[] = [];
  for (let k = kStart; k <= kEnd; k++) {
    for (const m of [1, 2, 5] as const) {
      const v = m * 10 ** k;
      if (v >= ya * (1 - 1e-12) && v <= yb * (1 + 1e-12)) rawCandidates.push(v);
    }
  }
  const candidates = Array.from(new Set(rawCandidates)).sort((a, b) => a - b);
  if (candidates.length > 0) {
    if (candidates.length <= maxTicks + 2) return candidates;
    const target = Math.max(3, Math.min(maxTicks + 2, 10));
    const picked = new Set<number>();
    const loL = Math.log10(candidates[0]!);
    const hiL = Math.log10(candidates[candidates.length - 1]!);
    for (let i = 0; i < target; i++) {
      const t = target <= 1 ? 0.5 : i / (target - 1);
      const logV = loL + t * (hiL - loL);
      let best = candidates[0]!;
      let bestD = Infinity;
      for (const c of candidates) {
        const d = Math.abs(Math.log10(c) - logV);
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      picked.add(best);
    }
    return Array.from(picked).sort((a, b) => a - b);
  }

  const span = yb - ya || SL_LOG_Y_FLOOR;
  let step = slNiceStep(span / Math.max(1, maxTicks - 1));
  if (step > span) step = slNiceStep(span);
  const start = Math.ceil(ya / step - 1e-9) * step;
  const linearOut: number[] = [];
  for (let y = start; y <= yb + step * 1e-9; y += step) {
    if (y >= ya - 1e-9 && y <= yb + 1e-9) linearOut.push(y);
    if (linearOut.length > 12) break;
  }
  if (linearOut.length === 0) return [ya, yb];
  return linearOut;
}

const PLAIN_LOG_TICK_MULTS = [1, 3, 10] as const;

/**
 * Raw x tick values for log10(rawX) axes: {1,3,10}×10^k in range, thinned if there are too many.
 */
function slGenerateRawTicksPlainLog13(xMin: number, xMax: number, maxTicks: number): number[] {
  const xa = Math.max(xMin, 1e-15);
  const xb = Math.max(xa * (1 + 1e-15), xMax);
  const lo = Math.log10(xa);
  const hi = Math.log10(xb);
  const kStart = Math.floor(lo - 1e-12);
  const kEnd = Math.ceil(hi + 1e-12);
  const acc: number[] = [];
  for (let k = kStart; k <= kEnd; k++) {
    for (const m of PLAIN_LOG_TICK_MULTS) {
      const v = m * 10 ** k;
      if (v >= xa * (1 - 1e-12) && v <= xb * (1 + 1e-12)) acc.push(v);
    }
  }
  const sorted = Array.from(new Set(acc)).sort((a, b) => a - b);
  if (sorted.length === 0) return [xa, xb];
  const budget = Math.max(4, Math.min(maxTicks + 2, 14));
  if (sorted.length <= budget) return sorted;
  const loL = Math.log10(sorted[0]!);
  const hiL = Math.log10(sorted[sorted.length - 1]!);
  const target = Math.min(budget, sorted.length);
  const picked = new Set<number>();
  for (let i = 0; i < target; i++) {
    const t = target <= 1 ? 0.5 : i / (target - 1);
    const logV = loL + t * (hiL - loL);
    let best = sorted[0]!;
    let bestD = Infinity;
    for (const s of sorted) {
      const d = Math.abs(Math.log10(s) - logV);
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    }
    picked.add(best);
  }
  return Array.from(picked).sort((a, b) => a - b);
}

/** X-axis tick positions in plot space for Training / scalar charts using log10(step+1). */
export function slGenerateXTicks(minPlotX: number, maxPlotX: number, logX: boolean, maxTicks = 5): number[] {
  if (!logX) return slGenerateTicks(minPlotX, maxPlotX, maxTicks);
  const lo = Math.min(minPlotX, maxPlotX);
  const hi = Math.max(minPlotX, maxPlotX);
  const xMinRaw = Math.max(0, slStepFromPlotX(lo, true));
  const xMaxRaw = Math.max(xMinRaw, slStepFromPlotX(hi, true));
  const raw = slGenerateRawStepTicksForLogStepAxis(xMinRaw, xMaxRaw, maxTicks);
  const plot = slUniqueSortedInRange(
    raw.map((s) => slTransformStep(s, true)),
    lo,
    hi,
  );
  return plot.length ? plot : slGenerateTicks(lo, hi, maxTicks);
}

/** Y-axis tick positions in plot space for log10(value) y-axis. */
export function slGenerateYTicks(minPlotY: number, maxPlotY: number, logY: boolean, maxTicks = 5): number[] {
  if (!logY) return slGenerateTicks(minPlotY, maxPlotY, maxTicks);
  const lo = Math.min(minPlotY, maxPlotY);
  const hi = Math.max(minPlotY, maxPlotY);
  const yMinRaw = Math.max(SL_LOG_Y_FLOOR, 10 ** lo);
  const yMaxRaw = Math.max(yMinRaw * (1 + 1e-15), 10 ** hi);
  const raw = slGenerateRawYTicksForLogValueAxis(yMinRaw, yMaxRaw, maxTicks);
  const plot = slUniqueSortedInRange(
    raw.map((y) => slTransformLoss(y, true)),
    lo,
    hi,
  );
  return plot.length ? plot : slGenerateTicks(lo, hi, maxTicks);
}

/**
 * X-axis ticks in plot space when transform is log10(rawX), rawX > 0 (sweep numeric x).
 * Uses {1,3,10}×10^k style positions in raw space (thinned when crowded), not linear raw spacing.
 */
export function slGenerateXTicksPlainLog(minPlotX: number, maxPlotX: number, maxTicks = 7): number[] {
  const lo = Math.min(minPlotX, maxPlotX);
  const hi = Math.max(minPlotX, maxPlotX);
  const xMin = Math.max(1e-15, 10 ** lo);
  const xMax = Math.max(xMin * (1 + 1e-15), 10 ** hi);
  const raw = slGenerateRawTicksPlainLog13(xMin, xMax, maxTicks);
  const plot = slUniqueSortedInRange(
    raw.map((rx) => Math.log10(rx)),
    lo,
    hi,
  );
  return plot.length ? plot : slGenerateTicks(lo, hi, maxTicks);
}

export function slGenerateXTicksScalar(
  minPlotX: number,
  maxPlotX: number,
  logX: boolean,
  plainLogX: boolean,
  maxTicks?: number,
): number[] {
  const n = maxTicks ?? (logX && plainLogX ? 7 : 5);
  if (!logX) return slGenerateTicks(minPlotX, maxPlotX, n);
  if (plainLogX) return slGenerateXTicksPlainLog(minPlotX, maxPlotX, n);
  return slGenerateXTicks(minPlotX, maxPlotX, true, n);
}

export function slFormatXTickScalar(t: number, logX: boolean, plainLogX: boolean): string {
  if (!logX) return slFormatXTick(t, false);
  if (plainLogX) return slFormatXTickPlainLog(t);
  return slFormatXTick(t, true);
}

/** Tick label when x transform is log10(raw) (no +1). */
export function slFormatXTickPlainLog(t: number): string {
  const raw = 10 ** t;
  if (!Number.isFinite(raw) || raw <= 0) return "0";
  const r = Math.round(raw);
  if (r > 0 && Math.abs(raw - r) / Math.max(r, 1e-12) < 1e-5) return String(r);
  if (raw >= 10_000) return `${Math.round(raw / 1000)}k`;
  if (raw >= 1000) return raw.toFixed(0);
  if (raw >= 1) return raw.toFixed(2).replace(/\.?0+$/, "");
  return raw.toFixed(3).replace(/\.?0+$/, "");
}

export function slFormatXTick(t: number, logX: boolean): string {
  if (logX) {
    const s = 10 ** t - 1;
    if (!Number.isFinite(s) || s < 0) return "0";
    if (Math.abs(s - Math.round(s)) <= 1e-6 + 1e-9 * Math.max(1, s)) return String(Math.round(s));
    if (s >= 10_000) return `${Math.round(s / 1000)}k`;
    return String(Math.round(s));
  }
  if (Math.abs(t - Math.round(t)) < 1e-6) return String(Math.round(t));
  return t.toFixed(1);
}

export function slFormatYTick(t: number, logY: boolean): string {
  if (logY) {
    const v = 10 ** t;
    if (!Number.isFinite(v)) return "";
    const r = Math.round(v);
    if (r !== 0 && Math.abs(v - r) / Math.max(Math.abs(r), 1e-12) < 1e-5) return String(r);
    if (v >= 1000 || (v > 0 && v < 0.01)) return v.toExponential(0);
    if (v >= 1) return v.toFixed(2).replace(/\.?0+$/, "");
    return v.toFixed(3).replace(/\.?0+$/, "");
  }
  if (t === 0) return "0";
  if (Math.abs(t) >= 1000 || (Math.abs(t) < 0.01 && t !== 0)) return t.toExponential(1);
  if (Math.abs(t) < 1) return t.toFixed(3).replace(/\.?0+$/, "");
  return t.toFixed(2).replace(/\.?0+$/, "");
}

export function slComputeBoundsTraining(
  steps: number[],
  trainVals: number[],
  testVals: number[] | null,
  showTrain: boolean,
  showTest: boolean,
  logX: boolean,
  logY: boolean,
  yPlotMetric: "loss" | "perplexity",
): SLBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;

  const consider = (values: number[]) => {
    if (values.length !== steps.length) return;
    for (let i = 0; i < steps.length; i++) {
      const x = slTransformStep(steps[i]!, logX);
      const y = slTransformLoss(slRawLossToPlotY(values[i]!, yPlotMetric), logY);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      any = true;
    }
  };

  if (showTrain) consider(trainVals);
  if (showTest && testVals) consider(testVals);

  if (!any) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 0.5;
    maxY += 0.5;
  }
  // Match matplotlib default margins (~5%) so canvas curves align with notebook ``plt.plot`` views.
  const spanX = maxX - minX || 1e-12;
  const spanY = maxY - minY || 1e-12;
  const mx = spanX * 0.05;
  const my = spanY * 0.05;
  minX -= mx;
  maxX += mx;
  minY -= my;
  maxY += my;
  if (!logX) {
    minX = Math.max(0, minX);
  }
  return { minX, maxX, minY, maxY };
}

export function slComputeBoundsScalar(
  steps: number[],
  values: number[],
  show: boolean,
  logX: boolean,
  logY: boolean,
  plainLogX = false,
): SLBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  if (show && values.length === steps.length) {
    for (let i = 0; i < steps.length; i++) {
      const x = slScalarPlotXFromRaw(steps[i]!, logX, plainLogX);
      const y = slTransformLoss(values[i]!, logY);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      any = true;
    }
  }
  if (!any) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 0.5;
    maxY += 0.5;
  }
  return slPadBounds({ minX, maxX, minY, maxY }, 0.055, 0.055);
}

export function slBuildSeriesPathTraining(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: SLBounds,
  yPlotMetric: "loss" | "perplexity",
): string {
  if (steps.length < 2 || values.length !== steps.length) return "";
  const innerW = SL_CHART_W - SL_PAD_L - SL_PAD_R;
  const innerH = SL_CHART_H - SL_PAD_T - SL_PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return steps
    .map((s, i) => {
      const x = slTransformStep(s, logX);
      const y = slTransformLoss(slRawLossToPlotY(values[i]!, yPlotMetric), logY);
      const px = SL_PAD_L + (innerW * (x - b.minX)) / spanX;
      const py = SL_PAD_T + innerH * (1 - (y - b.minY) / spanY);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

export function slBuildSeriesPathScalar(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: SLBounds,
  plainLogX = false,
): string {
  if (steps.length < 2 || values.length !== steps.length) return "";
  const innerW = SL_CHART_W - SL_PAD_L - SL_PAD_R;
  const innerH = SL_CHART_H - SL_PAD_T - SL_PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  const cmds: string[] = [];
  let penDown = false;
  for (let i = 0; i < steps.length; i++) {
    const x = slScalarPlotXFromRaw(steps[i]!, logX, plainLogX);
    const y = slTransformLoss(values[i]!, logY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      penDown = false;
      continue;
    }
    const px = SL_PAD_L + (innerW * (x - b.minX)) / spanX;
    const py = SL_PAD_T + innerH * (1 - (y - b.minY) / spanY);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      penDown = false;
      continue;
    }
    cmds.push(`${penDown ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`);
    penDown = true;
  }
  return cmds.join(" ");
}

export function slXToPx(x: number, b: SLBounds): number {
  const innerW = SL_CHART_W - SL_PAD_L - SL_PAD_R;
  return SL_PAD_L + (innerW * (x - b.minX)) / (b.maxX - b.minX || 1);
}

export function slYToPx(y: number, b: SLBounds): number {
  const innerH = SL_CHART_H - SL_PAD_T - SL_PAD_B;
  return SL_PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

/** Map SVG pixel x to plot-space x (same space as `slTransformStep` output). */
export function slPxToPlotX(px: number, b: SLBounds): number {
  const innerW = SL_CHART_W - SL_PAD_L - SL_PAD_R;
  const t = (px - SL_PAD_L) / innerW;
  return b.minX + t * (b.maxX - b.minX || 1);
}

export function slInnerRight(): number {
  return SL_CHART_W - SL_PAD_R;
}

export function slInnerBottom(): number {
  return SL_PAD_T + (SL_CHART_H - SL_PAD_T - SL_PAD_B);
}

export function slViewBoundsFromZoom(
  fullBounds: SLBounds,
  zoomXMin: number | undefined,
  zoomXMax: number | undefined,
): SLBounds {
  if (typeof zoomXMin === "number" && typeof zoomXMax === "number") {
    return {
      minX: zoomXMin,
      maxX: zoomXMax,
      minY: fullBounds.minY,
      maxY: fullBounds.maxY,
    };
  }
  return fullBounds;
}
