import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useMemo, useState } from "react";
import {
  defaultObservableVizWeightL1Data,
  type ObservableVizWeightL1NodeData,
} from "./observableVizWeightL1Defaults";
import { useObservableVizHeaderTitle } from "./observableVizTitle";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { VizSocketsBar } from "./VizSocketsBar";
import { slFormatXTick, slFormatYTick, slGenerateXTicks, slGenerateYTicks, slPadBounds } from "./scalarLineChartShared";
import { useLineChartZoom } from "./useLineChartZoom";

const CHART_W = 232;
const CHART_H = 122;
const PAD_L = 36;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 32;
const TICK_LEN = 4;
const Y_LABEL_X = 9;
const LOG_Y_FLOOR = 1e-15;

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function patchData(
  id: string,
  patch: Partial<ObservableVizWeightL1NodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultObservableVizWeightL1Data();
      const cur = (n.data ?? {}) as Partial<ObservableVizWeightL1NodeData>;
      const prev: ObservableVizWeightL1NodeData = {
        pairedObservableId: cur.pairedObservableId,
        pairedTrainerId: cur.pairedTrainerId,
        lastSweepSummary: cur.lastSweepSummary,
        valueHistory: cur.valueHistory,
        stepTicks: cur.stepTicks,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
        showSeries: cur.showSeries ?? def.showSeries ?? true,
        zoomXMin: cur.zoomXMin,
        zoomXMax: cur.zoomXMax,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function transformStep(step: number, logX: boolean): number {
  return logX ? Math.log10(Math.max(0, step) + 1) : step;
}

function transformY(v: number, logY: boolean): number {
  return logY ? Math.log10(Math.max(v, LOG_Y_FLOOR)) : v;
}

function computeBounds(
  steps: number[],
  values: number[],
  show: boolean,
  logX: boolean,
  logY: boolean,
): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  if (show && values.length === steps.length) {
    for (let i = 0; i < steps.length; i++) {
      const x = transformStep(steps[i]!, logX);
      const y = transformY(values[i]!, logY);
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

function buildPath(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: Bounds,
): string {
  if (steps.length < 2 || values.length !== steps.length) return "";
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return steps
    .map((s, i) => {
      const x = transformStep(s, logX);
      const y = transformY(values[i]!, logY);
      const px = PAD_L + (innerW * (x - b.minX)) / spanX;
      const py = PAD_T + innerH * (1 - (y - b.minY) / spanY);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

function xToPx(x: number, b: Bounds): number {
  const innerW = CHART_W - PAD_L - PAD_R;
  return PAD_L + (innerW * (x - b.minX)) / (b.maxX - b.minX || 1);
}

function yToPx(y: number, b: Bounds): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

export function ObservableVizWeightL1Node({ id, data, selected }: NodeProps) {
  const def = defaultObservableVizWeightL1Data();
  const raw = (data ?? {}) as Partial<ObservableVizWeightL1NodeData>;
  const d: ObservableVizWeightL1NodeData = {
    pairedObservableId: raw.pairedObservableId,
    pairedTrainerId: raw.pairedTrainerId,
    lastSweepSummary: raw.lastSweepSummary,
    valueHistory: raw.valueHistory,
    stepTicks: raw.stepTicks,
    logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
    logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
    showSeries: raw.showSeries ?? def.showSeries ?? true,
    zoomXMin: raw.zoomXMin,
    zoomXMax: raw.zoomXMax,
  };
  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<ObservableVizWeightL1NodeData>) => patchData(id, patch, setNodes);
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);

  const isLive = !!(
    d.valueHistory &&
    d.stepTicks &&
    d.valueHistory.length >= 2 &&
    d.stepTicks.length === d.valueHistory.length
  );

  const steps = useMemo(() => {
    if (isLive && d.stepTicks) return d.stepTicks;
    return [];
  }, [isLive, d.stepTicks]);

  const vals = useMemo(() => {
    if (isLive && d.valueHistory) return d.valueHistory;
    return [];
  }, [isLive, d.valueHistory]);

  const bounds = useMemo(
    () => computeBounds(steps, vals, !!d.showSeries, !!d.logScaleX, !!d.logScaleY),
    [steps, vals, d.showSeries, d.logScaleX, d.logScaleY],
  );
  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - PAD_R;
  const hasSeries = !!d.showSeries && vals.length >= 2 && vals.length === steps.length;
  const {
    viewBounds,
    isZoomed,
    selectionRect,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onMouseLeave,
    resetZoom,
  } = useLineChartZoom({
    bounds,
    enabled: hasSeries,
    left: PAD_L,
    right: innerRight,
    top: PAD_T,
    bottom: innerBottom,
    initialZoom:
      typeof d.zoomXMin === "number" && typeof d.zoomXMax === "number"
        ? { minX: d.zoomXMin, maxX: d.zoomXMax, minY: bounds.minY, maxY: bounds.maxY }
        : null,
    onZoomChange: (z) => {
      update({
        zoomXMin: z?.minX,
        zoomXMax: z?.maxX,
      });
    },
  });

  const xTicks = useMemo(
    () => slGenerateXTicks(viewBounds.minX, viewBounds.maxX, !!d.logScaleX),
    [viewBounds, d.logScaleX],
  );
  const yTicks = useMemo(
    () => slGenerateYTicks(viewBounds.minY, viewBounds.maxY, !!d.logScaleY),
    [viewBounds, d.logScaleY],
  );

  const seriesPath = useMemo(() => {
    if (!d.showSeries) return "";
    return buildPath(steps, vals, !!d.logScaleX, !!d.logScaleY, viewBounds);
  }, [d.showSeries, d.logScaleX, d.logScaleY, steps, vals, viewBounds]);

  const plotMidX = PAD_L + (CHART_W - PAD_L - PAD_R) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${id}-ow1-clip`;

  const hint = isLive
    ? "Weight L1 from last training run"
    : "No data yet — connect Trainer “observable”, then Train.";

  return (
    <div
      className={`cr-node cr-node--observable-viz-weight-l1${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar annotatorSource />
        <div className="cr-tviz-chart-divider" aria-hidden />
        <div className="cr-tviz-chart-controls nodrag nopan">
          <label className="cr-tviz-check cr-tviz-check--train">
            <input
              type="checkbox"
              checked={!!d.showSeries}
              onChange={(e) => update({ showSeries: e.target.checked })}
            />
            ‖W‖₁
          </label>
          <label className="cr-tviz-check cr-tviz-check--log-x">
            <input
              type="checkbox"
              checked={!!d.logScaleX}
              onChange={(e) => update({ logScaleX: e.target.checked })}
            />
            log x
          </label>
          <label className="cr-tviz-check cr-tviz-check--log-y">
            <input
              type="checkbox"
              checked={!!d.logScaleY}
              onChange={(e) => update({ logScaleY: e.target.checked })}
            />
            log y
          </label>
          {isZoomed ? (
            <button type="button" className="cr-tviz-reset-zoom" onClick={resetZoom}>
              reset zoom-in
            </button>
          ) : null}
        </div>
        {d.lastSweepSummary?.trim() ? (
          <div className="cr-tviz-sweep-line nodrag nopan" title={d.lastSweepSummary}>
            {d.lastSweepSummary}
          </div>
        ) : null}
        <div className="cr-tviz-chart-wrap">
          <svg
            className="cr-tviz-chart nodrag nopan"
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            width={CHART_W}
            height={CHART_H}
            aria-label="Weight L1 norm over training"
            onMouseDown={(e) => {
              if (!hasSeries) return;
              setIsDraggingZoom(true);
              onMouseDown(e);
            }}
            onMouseMove={onMouseMove}
            onMouseUp={() => {
              setIsDraggingZoom(false);
              onMouseUp();
            }}
            onMouseLeave={() => {
              setIsDraggingZoom(false);
              onMouseLeave();
            }}
            style={{ cursor: hasSeries ? (isDraggingZoom ? "crosshair" : "zoom-in") : "default" }}
          >
            <defs>
              <clipPath id={clipId}>
                <rect
                  x={PAD_L}
                  y={PAD_T}
                  width={CHART_W - PAD_L - PAD_R}
                  height={CHART_H - PAD_T - PAD_B}
                />
              </clipPath>
            </defs>
            <rect
              x={PAD_L}
              y={PAD_T}
              width={CHART_W - PAD_L - PAD_R}
              height={CHART_H - PAD_T - PAD_B}
              rx={4}
              className="cr-tviz-chart__plot-bg"
            />

            {yTicks.map((yt) => (
              <line
                key={`gy-${yt}`}
                x1={PAD_L}
                y1={yToPx(yt, viewBounds)}
                x2={innerRight}
                y2={yToPx(yt, viewBounds)}
                className="cr-tviz-chart__grid"
              />
            ))}
            {xTicks.map((xt) => (
              <line
                key={`gx-${xt}`}
                x1={xToPx(xt, viewBounds)}
                y1={PAD_T}
                x2={xToPx(xt, viewBounds)}
                y2={innerBottom}
                className="cr-tviz-chart__grid"
              />
            ))}

            <line
              x1={PAD_L}
              y1={innerBottom}
              x2={innerRight}
              y2={innerBottom}
              className="cr-tviz-chart__axis-line"
            />
            <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={innerBottom} className="cr-tviz-chart__axis-line" />

            {xTicks.map((xt) => {
              const px = xToPx(xt, viewBounds);
              return (
                <g key={`xt-${xt}`}>
                  <line
                    x1={px}
                    y1={innerBottom}
                    x2={px}
                    y2={innerBottom + TICK_LEN}
                    className="cr-tviz-chart__tick"
                  />
                  <text x={px} y={innerBottom + 12} textAnchor="middle" className="cr-tviz-chart__tick-label">
                    {slFormatXTick(xt, !!d.logScaleX)}
                  </text>
                </g>
              );
            })}

            {yTicks.map((yt) => {
              const py = yToPx(yt, viewBounds);
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
                    {slFormatYTick(yt, !!d.logScaleY)}
                  </text>
                </g>
              );
            })}

            {seriesPath ? (
              <path
                d={seriesPath}
                fill="none"
                className="cr-tviz-chart__line"
                strokeWidth={1.6}
                clipPath={`url(#${clipId})`}
              />
            ) : null}
            {selectionRect ? (
              <rect
                x={selectionRect.x}
                y={selectionRect.y}
                width={selectionRect.w}
                height={selectionRect.h}
                className="cr-tviz-chart__zoom-box"
              />
            ) : null}

            <text x={plotMidX} y={CHART_H - 2} textAnchor="middle" className="cr-tviz-chart__axis-title">
              step
            </text>

            <text
              x={Y_LABEL_X}
              y={plotMidY}
              textAnchor="middle"
              dominantBaseline="middle"
              transform={`rotate(-90, ${Y_LABEL_X}, ${plotMidY})`}
              className="cr-tviz-chart__axis-title"
            >
              ‖W‖₁
            </text>

            {seriesPath ? (
              <g className="cr-tviz-legend" transform={`translate(${innerRight - 48}, ${PAD_T + 4})`}>
                <line x1={0} y1={5} x2={14} y2={5} className="cr-tviz-chart__line" strokeWidth={1.6} />
                <text x={17} y={7} className="cr-tviz-chart__legend-text">
                  ‖W‖₁
                </text>
              </g>
            ) : null}
          </svg>
          <p className="cr-tviz-hint">{hint}</p>
        </div>
      </div>
    </div>
  );
}
