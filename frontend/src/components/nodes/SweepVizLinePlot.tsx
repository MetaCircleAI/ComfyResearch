import { useMemo } from "react";
import { logYForAxisSide, type PlotSeries } from "../../graph/sweepVizPlot";
import {
  slFormatXTickPlainLog,
  slFormatYTick,
  slGenerateTicks,
  slGenerateXTicksPlainLog,
  slGenerateYTicks,
  slPadBounds,
} from "./scalarLineChartShared";

type SweepVizLinePlotProps = {
  chartId: string;
  series: PlotSeries[];
  xKey: string;
  xIsNumeric: boolean;
  yAxisLabel: string;
  yAxisLabelRight?: string;
  legendSummary: string;
  logScaleX: boolean;
  logScaleY: boolean;
  /** Split accuracy-like series onto a right Y axis (curve overlays). */
  dualAxis?: boolean;
  /** Scatter markers on data points (sweep tables); off for time-series overlays. */
  showMarkers?: boolean;
  /** Keep the external legend list when the SVG's in-plot legend is insufficient. */
  showLegendList?: boolean;
};

const CHART_W = 232;
const CHART_H = 122;
const PAD_L = 36;
const PAD_R = 8;
const PAD_R_DUAL = 36;
const PAD_T = 10;
const PAD_B = 32;
const TICK_LEN = 4;
const Y_LABEL_X = 9;
const Y_LABEL_RIGHT_X = CHART_W - 6;

const LOG_Y_FLOOR = 1e-15;

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function transformPlotX(rawX: number, logX: boolean, xIsNumeric: boolean): number {
  if (!xIsNumeric) return rawX;
  return logX ? Math.log10(Math.max(rawX, 1e-15)) : rawX;
}

function transformY(v: number, logY: boolean): number {
  return logY ? Math.log10(Math.max(v, LOG_Y_FLOOR)) : v;
}

function computeBounds(
  series: PlotSeries[],
  logScaleX: boolean,
  logScaleY: boolean,
  xIsNumeric: boolean,
): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;

  for (const s of series) {
    for (const p of s.points) {
      const x = transformPlotX(p.x, logScaleX && xIsNumeric, xIsNumeric);
      const y = transformY(p.y, logScaleY);
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
  // Inset data from the clip rect so markers are not cut off at edges.
  return slPadBounds({ minX, maxX, minY, maxY }, 0.055, 0.055);
}

function buildSeriesPath(
  points: { x: number; y: number }[],
  logScaleX: boolean,
  logScaleY: boolean,
  xIsNumeric: boolean,
  b: Bounds,
  padRight: number,
): string {
  if (points.length < 2) return "";
  const innerW = CHART_W - PAD_L - padRight;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return points
    .map((p, i) => {
      const tx = transformPlotX(p.x, logScaleX && xIsNumeric, xIsNumeric);
      const ty = transformY(p.y, logScaleY);
      const px = PAD_L + (innerW * (tx - b.minX)) / spanX;
      const py = PAD_T + innerH * (1 - (ty - b.minY) / spanY);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

function xToPx(x: number, b: Bounds, padRight: number): number {
  const innerW = CHART_W - PAD_L - padRight;
  return PAD_L + (innerW * (x - b.minX)) / (b.maxX - b.minX || 1);
}

function yToPx(y: number, b: Bounds): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

function seriesSide(s: PlotSeries): "left" | "right" {
  return s.yAxis === "right" ? "right" : "left";
}

export function SweepVizLinePlot({
  chartId,
  series,
  xKey,
  xIsNumeric,
  yAxisLabel,
  yAxisLabelRight = "accuracy",
  legendSummary,
  logScaleX,
  logScaleY,
  dualAxis = false,
  showMarkers = true,
  showLegendList = true,
}: SweepVizLinePlotProps) {
  const padR = dualAxis ? PAD_R_DUAL : PAD_R;
  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - padR;
  const plotMidX = PAD_L + (CHART_W - PAD_L - padR) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${chartId}-tviz-sweep-clip`;

  const effectiveLogX = !!(logScaleX && xIsNumeric);
  const effectiveLogY = !!logScaleY;
  // 双轴时 log-y 仅左轴(loss);右轴(acc/observable)保持线性(Fig-1 形态)。
  const logYLeft = logYForAxisSide("left", dualAxis, effectiveLogY);
  const logYRight = logYForAxisSide("right", dualAxis, effectiveLogY);
  const logYForSeries = (s: PlotSeries) => (dualAxis ? (seriesSide(s) === "right" ? logYRight : logYLeft) : effectiveLogY);

  const boundsAll = useMemo(
    () => computeBounds(series, effectiveLogX, effectiveLogY, xIsNumeric),
    [series, effectiveLogX, effectiveLogY, xIsNumeric],
  );

  const boundsLeft = useMemo(() => {
    const leftSeries = series.filter((s) => seriesSide(s) === "left");
    return leftSeries.length > 0
      ? computeBounds(leftSeries, effectiveLogX, logYLeft, xIsNumeric)
      : boundsAll;
  }, [series, boundsAll, effectiveLogX, logYLeft, xIsNumeric]);

  const boundsRight = useMemo(() => {
    const rightSeries = series.filter((s) => seriesSide(s) === "right");
    return rightSeries.length > 0
      ? computeBounds(rightSeries, effectiveLogX, logYRight, xIsNumeric)
      : boundsAll;
  }, [series, boundsAll, effectiveLogX, logYRight, xIsNumeric]);

  const xBounds = boundsAll;
  const yBoundsFor = (s: PlotSeries) => {
    if (!dualAxis) return boundsAll;
    return seriesSide(s) === "right" ? boundsRight : boundsLeft;
  };

  const yToPxFor = (y: number, s: PlotSeries) => yToPx(y, yBoundsFor(s));

  const yTicksBoundsLeft = dualAxis ? boundsLeft : boundsAll;

  const catLabelByIndex = useMemo(() => {
    const m = new Map<number, string>();
    if (xIsNumeric) return m;
    for (const s of series) {
      for (const p of s.points) {
        m.set(p.x, p.xDisplay);
      }
    }
    return m;
  }, [series, xIsNumeric]);

  const xTicks = useMemo(() => {
    if (!xIsNumeric) {
      const keys = [...catLabelByIndex.keys()].sort((a, b) => a - b);
      if (keys.length) return keys;
    }
    if (effectiveLogX) return slGenerateXTicksPlainLog(xBounds.minX, xBounds.maxX);
    return slGenerateTicks(xBounds.minX, xBounds.maxX);
  }, [xIsNumeric, catLabelByIndex, xBounds, effectiveLogX]);

  const yTicksLeft = useMemo(() => {
    return logYLeft
      ? slGenerateYTicks(yTicksBoundsLeft.minY, yTicksBoundsLeft.maxY, true)
      : slGenerateTicks(yTicksBoundsLeft.minY, yTicksBoundsLeft.maxY);
  }, [yTicksBoundsLeft, logYLeft]);

  const yTicksRight = useMemo(() => {
    if (!dualAxis) return [];
    return logYRight
      ? slGenerateYTicks(boundsRight.minY, boundsRight.maxY, true)
      : slGenerateTicks(boundsRight.minY, boundsRight.maxY);
  }, [boundsRight, dualAxis, logYRight]);

  const seriesPaths = useMemo(() => {
    return series.map((s) => {
      const yb = dualAxis ? yBoundsFor(s) : boundsAll;
      const merged: Bounds = { minX: xBounds.minX, maxX: xBounds.maxX, minY: yb.minY, maxY: yb.maxY };
      return {
        id: s.id,
        color: s.color,
        strokeDasharray: s.strokeDasharray,
        d: buildSeriesPath(
          s.points.map((p) => ({ x: p.x, y: p.y })),
          effectiveLogX,
          logYForSeries(s),
          xIsNumeric,
          merged,
          padR,
        ),
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, xBounds, boundsAll, boundsLeft, boundsRight, dualAxis, effectiveLogX, effectiveLogY, xIsNumeric, padR]);

  const hasSeries = series.some((s) => s.points.length > 0);

  return (
    <div className="cr-sweep-viz__lineplot">
      {legendSummary ? (
        <div className="cr-sweep-viz__lineplot-meta">
          <span className="cr-sweep-viz__lineplot-meta-line">{legendSummary}</span>
        </div>
      ) : null}
      <div className="cr-tviz-chart-wrap">
        <svg
          className="cr-tviz-chart nodrag nopan"
          viewBox={`0 0 ${CHART_W} ${CHART_H}`}
          width={CHART_W}
          height={CHART_H}
          aria-label="Sweep line plot"
        >
          <defs>
            <clipPath id={clipId}>
              <rect
                x={PAD_L}
                y={PAD_T}
                width={CHART_W - PAD_L - padR}
                height={CHART_H - PAD_T - PAD_B}
              />
            </clipPath>
          </defs>
          <rect
            x={PAD_L}
            y={PAD_T}
            width={CHART_W - PAD_L - padR}
            height={CHART_H - PAD_T - PAD_B}
            rx={4}
            className="cr-tviz-chart__plot-bg"
          />

          {yTicksLeft.map((yt) => {
            const py = yToPx(yt, yTicksBoundsLeft);
            return (
              <line
                key={`gy-${yt}`}
                x1={PAD_L}
                y1={py}
                x2={innerRight}
                y2={py}
                className="cr-tviz-chart__grid"
              />
            );
          })}
          {xTicks.map((xt) => {
            const px = xToPx(xt, xBounds, padR);
            return (
              <line
                key={`gx-${xt}`}
                x1={px}
                y1={PAD_T}
                x2={px}
                y2={innerBottom}
                className="cr-tviz-chart__grid"
              />
            );
          })}

          <line
            x1={PAD_L}
            y1={innerBottom}
            x2={innerRight}
            y2={innerBottom}
            className="cr-tviz-chart__axis-line"
          />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={innerBottom} className="cr-tviz-chart__axis-line" />

          {xTicks.map((xt) => {
            const px = xToPx(xt, xBounds, padR);
            return (
              <g key={`xt-${xt}`}>
                <line
                  x1={px}
                  y1={innerBottom}
                  x2={px}
                  y2={innerBottom + TICK_LEN}
                  className="cr-tviz-chart__tick"
                />
                <text
                  x={px}
                  y={innerBottom + 12}
                  textAnchor="middle"
                  className="cr-tviz-chart__tick-label"
                >
                  {xIsNumeric
                    ? effectiveLogX
                      ? slFormatXTickPlainLog(xt)
                      : Math.abs(xt - Math.round(xt)) < 1e-6
                        ? String(Math.round(xt))
                        : xt.toFixed(1)
                    : (() => {
                        const raw = catLabelByIndex.get(xt) ?? String(xt);
                        return raw.length > 10 ? `${raw.slice(0, 9)}…` : raw;
                      })()}
                </text>
              </g>
            );
          })}

          {yTicksLeft.map((yt) => {
            const py = yToPx(yt, yTicksBoundsLeft);
            return (
              <g key={`yt-${yt}`}>
                <line
                  x1={PAD_L - TICK_LEN}
                  y1={py}
                  x2={PAD_L}
                  y2={py}
                  className="cr-tviz-chart__tick"
                />
                <text
                  x={PAD_L - 6}
                  y={py}
                  dominantBaseline="middle"
                  textAnchor="end"
                  className="cr-tviz-chart__tick-label"
                >
                  {slFormatYTick(yt, logYLeft)}
                </text>
              </g>
            );
          })}

          {dualAxis ? (
            <line x1={innerRight} y1={PAD_T} x2={innerRight} y2={innerBottom} className="cr-tviz-chart__axis-line" />
          ) : null}

          {dualAxis
            ? yTicksRight.map((yt) => {
                const py = yToPx(yt, boundsRight);
                return (
                  <g key={`ytr-${yt}`}>
                    <line
                      x1={innerRight}
                      y1={py}
                      x2={innerRight + TICK_LEN}
                      y2={py}
                      className="cr-tviz-chart__tick"
                    />
                    <text
                      x={innerRight + 6}
                      y={py}
                      dominantBaseline="middle"
                      textAnchor="start"
                      className="cr-tviz-chart__tick-label"
                    >
                      {slFormatYTick(yt, logYRight)}
                    </text>
                  </g>
                );
              })
            : null}

          {seriesPaths.map((sp) =>
            sp.d ? (
              <path
                key={sp.id}
                d={sp.d}
                className="cr-tviz-chart__series-line"
                style={{ stroke: sp.color, strokeDasharray: sp.strokeDasharray }}
                strokeWidth={1.6}
                clipPath={`url(#${clipId})`}
              />
            ) : null,
          )}

          {showMarkers
            ? series.map((s) =>
                s.points.map((p) => {
                  const tx = transformPlotX(p.x, effectiveLogX, xIsNumeric);
                  const ty = transformY(p.y, logYForSeries(s));
                  const px = xToPx(tx, xBounds, padR);
                  const py = yToPxFor(ty, s);
                  return (
                    <circle
                      key={`${s.id}-${p.rowId}-${p.x}`}
                      cx={px}
                      cy={py}
                      r={2.5}
                      style={{ fill: s.color, stroke: "#0f0f12" }}
                      strokeWidth={0.5}
                      clipPath={`url(#${clipId})`}
                    />
                  );
                }),
              )
            : null}

          <text
            x={plotMidX}
            y={CHART_H - 2}
            textAnchor="middle"
            className="cr-tviz-chart__axis-title"
          >
            {xKey}
          </text>

          <text
            x={Y_LABEL_X}
            y={plotMidY}
            textAnchor="middle"
            dominantBaseline="middle"
            transform={`rotate(-90, ${Y_LABEL_X}, ${plotMidY})`}
            className="cr-tviz-chart__axis-title"
          >
            {dualAxis ? yAxisLabel || "loss" : yAxisLabel}
          </text>

          {dualAxis ? (
            <text
              x={Y_LABEL_RIGHT_X}
              y={plotMidY}
              textAnchor="middle"
              dominantBaseline="middle"
              transform={`rotate(90, ${Y_LABEL_RIGHT_X}, ${plotMidY})`}
              className="cr-tviz-chart__axis-title"
            >
              {yAxisLabelRight}
            </text>
          ) : null}

          {!dualAxis && showMarkers && hasSeries && series.length > 1 ? (
            <g className="cr-tviz-legend" transform={`translate(${innerRight - 58}, ${PAD_T + 2})`}>
              {series.slice(0, 4).map((s, i) => (
                <g key={s.id} transform={`translate(0, ${i * 12})`}>
                  <line
                    x1={0}
                    y1={5}
                    x2={12}
                    y2={5}
                    className="cr-tviz-chart__series-line"
                    style={{ stroke: s.color }}
                    strokeWidth={1.4}
                  />
                  <text x={14} y={7} className="cr-tviz-chart__legend-text">
                    {s.label.length > 14 ? `${s.label.slice(0, 13)}…` : s.label}
                  </text>
                </g>
              ))}
            </g>
          ) : null}
        </svg>
      </div>
      {showLegendList && series.length > 1 ? (
        <ul className="cr-sweep-viz__lineplot-legend">
          {series.map((s) => (
            <li key={s.id}>
              <span className="cr-sweep-viz__lineplot-legend-swatch" style={{ background: s.color }} />
              <span>{s.label}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
