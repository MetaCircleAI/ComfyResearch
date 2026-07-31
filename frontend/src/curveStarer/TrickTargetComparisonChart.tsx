import { useMemo } from "react";
import type { CurvePoint } from "./observableCurvePayload";
import { slFormatYTick, slGenerateYTicks, slPadBounds } from "../components/nodes/scalarLineChartShared";
import type { TargetObjective } from "./targetPhaseTransition";

const CHART_W = 520;
const CHART_H = 148;
const PAD_L = 40;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 28;

function sortedPoints(points: CurvePoint[]): CurvePoint[] {
  return points
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.loss))
    .slice()
    .sort((a, b) => a.t - b.t);
}

function boundsForCurves(baseline: CurvePoint[], trick: CurvePoint[], threshold?: number) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of [...baseline, ...trick]) {
    minX = Math.min(minX, p.t);
    maxX = Math.max(maxX, p.t);
    minY = Math.min(minY, p.loss);
    maxY = Math.max(maxY, p.loss);
  }
  if (Number.isFinite(threshold)) {
    minY = Math.min(minY, threshold);
    maxY = Math.max(maxY, threshold);
  }
  if (!Number.isFinite(minX)) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 0.5;
    maxY += 0.5;
  }
  return slPadBounds({ minX, maxX, minY, maxY }, 0.06, 0.08);
}

function buildPath(
  points: CurvePoint[],
  b: { minX: number; maxX: number; minY: number; maxY: number },
): string {
  const pts = sortedPoints(points);
  if (pts.length < 2) return "";
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return pts
    .map((p, i) => {
      const px = PAD_L + (innerW * (p.t - b.minX)) / spanX;
      const py = PAD_T + innerH * (1 - (p.loss - b.minY) / spanY);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

function xToPx(t: number, b: { minX: number; maxX: number }): number {
  const innerW = CHART_W - PAD_L - PAD_R;
  return PAD_L + (innerW * (t - b.minX)) / (b.maxX - b.minX || 1);
}

function yToPx(y: number, b: { minY: number; maxY: number }): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

export function TrickTargetComparisonChart({
  baselineCurve,
  trickCurve,
  yAxisLabel = "target",
  threshold,
  objective,
  baselineCrossingStep,
  trickCrossingStep,
}: {
  baselineCurve: CurvePoint[];
  trickCurve: CurvePoint[];
  yAxisLabel?: string;
  threshold?: number;
  objective?: TargetObjective;
  baselineCrossingStep?: number | null;
  trickCrossingStep?: number | null;
}) {
  const baseline = useMemo(() => sortedPoints(baselineCurve), [baselineCurve]);
  const trick = useMemo(() => sortedPoints(trickCurve), [trickCurve]);
  const b = useMemo(
    () => boundsForCurves(baseline, trick, threshold),
    [baseline, trick, threshold],
  );
  const yTicks = useMemo(() => slGenerateYTicks(b.minY, b.maxY, false), [b]);
  const innerBottom = CHART_H - PAD_B;
  const innerRight = CHART_W - PAD_R;

  const baselinePath = useMemo(() => buildPath(baseline, b), [baseline, b]);
  const trickPath = useMemo(() => buildPath(trick, b), [trick, b]);

  if (baseline.length < 2 && trick.length < 2) return null;

  const showThreshold = Number.isFinite(threshold);
  const thresholdY = showThreshold ? yToPx(threshold!, b) : 0;

  return (
    <div className="cr-trick-target-chart">
      <div className="cr-trick-target-chart__legend" aria-hidden>
        <span className="cr-trick-target-chart__legend-item cr-trick-target-chart__legend-item--baseline">
          Original target
        </span>
        {trick.length >= 2 ? (
          <span className="cr-trick-target-chart__legend-item cr-trick-target-chart__legend-item--trick">
            +Trick target
          </span>
        ) : null}
        {showThreshold ? (
          <span className="cr-trick-target-chart__legend-item cr-trick-target-chart__legend-item--threshold">
            Threshold
          </span>
        ) : null}
      </div>
      <svg
        className="cr-trick-target-chart__svg"
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        width="100%"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Original vs trick target curve comparison"
      >
        <rect
          x={PAD_L}
          y={PAD_T}
          width={CHART_W - PAD_L - PAD_R}
          height={CHART_H - PAD_T - PAD_B}
          rx={4}
          className="cr-trick-target-chart__bg"
        />
        {yTicks.map((yt) => (
          <line
            key={`gy-${yt}`}
            x1={PAD_L}
            y1={yToPx(yt, b)}
            x2={innerRight}
            y2={yToPx(yt, b)}
            className="cr-trick-target-chart__grid"
          />
        ))}
        <line x1={PAD_L} y1={innerBottom} x2={innerRight} y2={innerBottom} className="cr-trick-target-chart__axis" />
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={innerBottom} className="cr-trick-target-chart__axis" />
        {showThreshold ? (
          <line
            x1={PAD_L}
            y1={thresholdY}
            x2={innerRight}
            y2={thresholdY}
            className="cr-trick-target-chart__threshold"
          />
        ) : null}
        {baselineCrossingStep != null ? (
          <line
            x1={xToPx(baselineCrossingStep, b)}
            y1={PAD_T}
            x2={xToPx(baselineCrossingStep, b)}
            y2={innerBottom}
            className="cr-trick-target-chart__crossing cr-trick-target-chart__crossing--baseline"
          />
        ) : null}
        {trickCrossingStep != null ? (
          <line
            x1={xToPx(trickCrossingStep, b)}
            y1={PAD_T}
            x2={xToPx(trickCrossingStep, b)}
            y2={innerBottom}
            className="cr-trick-target-chart__crossing cr-trick-target-chart__crossing--trick"
          />
        ) : null}
        {yTicks.map((yt) => (
          <text
            key={`yt-${yt}`}
            x={PAD_L - 6}
            y={yToPx(yt, b)}
            dominantBaseline="middle"
            textAnchor="end"
            className="cr-trick-target-chart__tick"
          >
            {slFormatYTick(yt, false)}
          </text>
        ))}
        {baselinePath ? (
          <path d={baselinePath} fill="none" className="cr-trick-target-chart__line cr-trick-target-chart__line--baseline" />
        ) : null}
        {trickPath ? (
          <path d={trickPath} fill="none" className="cr-trick-target-chart__line cr-trick-target-chart__line--trick" />
        ) : null}
        <text x={PAD_L + (CHART_W - PAD_L - PAD_R) / 2} y={CHART_H - 4} textAnchor="middle" className="cr-trick-target-chart__axis-label">
          step
        </text>
        <text
          x={8}
          y={PAD_T + (CHART_H - PAD_T - PAD_B) / 2}
          textAnchor="middle"
          dominantBaseline="middle"
          transform={`rotate(-90, 8, ${PAD_T + (CHART_H - PAD_T - PAD_B) / 2})`}
          className="cr-trick-target-chart__axis-label"
        >
          {yAxisLabel}
          {showThreshold && objective ? ` (${objective === "higher" ? "≥" : "≤"} ${threshold})` : ""}
        </text>
      </svg>
    </div>
  );
}
