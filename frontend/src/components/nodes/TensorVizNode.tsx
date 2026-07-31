import { useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { ComfyIntField } from "./comfyNumberFields";
import {
  defaultTensorVizData,
  type TensorViz1dLineSort,
  type TensorViz1dPlotStyle,
  type TensorViz2dPlotStyle,
  type TensorViz2dScatterAxis,
  type TensorVizNodeData,
} from "./tensorVizDefaults";
import { formatTensorScalarDisplay } from "./tensorVizScalarFormat";
import { USER_OBSERVABLES_CHANGED } from "../../dnd";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
  type Resolved,
} from "../../graph/resolveUpstreamTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { TensorVizObsAddStrip } from "./TensorVizObsAddStrip";
import { VizSocketsBar } from "./VizSocketsBar";
import { useLineChartZoom } from "./useLineChartZoom";
import { slFormatYTick } from "./scalarLineChartShared";
import { TensorHeatmap } from "./TensorHeatmap";

export type TensorVizMode = "general" | "1d" | "2d";

function patchTensorVizData(
  id: string,
  patch: Partial<TensorVizNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTensorVizData();
      const cur = (n.data ?? {}) as Partial<TensorVizNodeData>;
      const prev: TensorVizNodeData = {
        plot1dStyle: cur.plot1dStyle ?? def.plot1dStyle,
        plot1dLineSort: cur.plot1dLineSort ?? def.plot1dLineSort,
        histBins: cur.histBins ?? def.histBins,
        plot2dStyle: cur.plot2dStyle ?? def.plot2dStyle,
        plot2dScatterAxis: cur.plot2dScatterAxis ?? def.plot2dScatterAxis,
        plot2dScatterI1: cur.plot2dScatterI1 ?? def.plot2dScatterI1,
        plot2dScatterI2: cur.plot2dScatterI2 ?? def.plot2dScatterI2,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function resolveTensor(nodes: Node[], edges: FlowEdge[], vizId: string): Resolved {
  return resolveUpstreamTensor(nodes, edges, vizId, "tensor");
}

function ordered1DValues(values: number[], sort: TensorViz1dLineSort): number[] {
  if (sort === "original") return values;
  const c = [...values];
  if (sort === "ascending") {
    c.sort((a, b) => a - b);
    return c;
  }
  c.sort((a, b) => b - a);
  return c;
}

function histogramCounts(
  values: number[],
  binCount: number,
): { counts: number[]; maxC: number; vmin: number; vmax: number } {
  const nbin = Math.max(1, Math.min(500, Math.floor(binCount)));
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) {
    return { counts: new Array(nbin).fill(0), maxC: 1, vmin: 0, vmax: 1 };
  }
  let vmin = Infinity;
  let vmax = -Infinity;
  for (const v of finite) {
    vmin = Math.min(vmin, v);
    vmax = Math.max(vmax, v);
  }
  const span = vmax - vmin;
  const counts = new Array(nbin).fill(0);
  if (span === 0 || !Number.isFinite(span)) {
    counts[0] = finite.length;
    return { counts, maxC: Math.max(1, finite.length), vmin, vmax };
  }
  const width = span / nbin;
  for (const v of finite) {
    let k = Math.floor((v - vmin) / width);
    if (k >= nbin) k = nbin - 1;
    if (k < 0) k = 0;
    counts[k]++;
  }
  const maxC = Math.max(1, ...counts);
  return { counts, maxC, vmin, vmax };
}

const CHART_W = 228;
const CHART_H_1D = 124;
/** Heatmap shows at most this many rows/columns (full tensor may be larger). */
const HEAT_MAX_DIM = 100;
const SCATTER_CHART_H = 132;

function heatColor(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const r = Math.round(32 + x * 200);
  const g = Math.round(48 + (1 - Math.abs(x - 0.45) * 1.4) * 160);
  const b = Math.round(120 + (1 - x) * 110);
  return `rgb(${r},${g},${b})`;
}

const AXIS_STROKE = "#5c5c6a";
const TICK_LEN = 4;

function niceStep(span: number): number {
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

function generateTicks(min: number, max: number, maxTicks = 4): number[] {
  const span = max - min;
  if (span <= 0 || !Number.isFinite(span)) return [min];
  const step = niceStep(span / Math.max(1, maxTicks - 1));
  const start = Math.ceil(min / step - 1e-9) * step;
  const out: number[] = [];
  for (let t = start; t <= max + step * 1e-9; t += step) {
    if (t >= min - 1e-9 && t <= max + 1e-9) out.push(t);
    if (out.length > 12) break;
  }
  if (out.length === 0) return [min, max];
  return out;
}

function formatNumberTick(t: number): string {
  return slFormatYTick(t, false);
}

function indexXTicks(n: number): number[] {
  if (n <= 0) return [0];
  if (n === 1) return [0];
  const raw = generateTicks(0, n - 1, 5).map((x) => Math.round(Math.min(n - 1, Math.max(0, x))));
  return [...new Set(raw)].sort((a, b) => a - b);
}

type PlotRect = { left: number; top: number; w: number; h: number };

/** Pad [min, max] on both ends so points/lines do not sit on the plot border (fraction of span per side). */
function padLinearDomain(min: number, max: number, fraction: number): { min: number; max: number } {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min, max };
  const span = max - min;
  if (span <= 0 || !Number.isFinite(span)) {
    const bump = (Math.abs(max) || Math.abs(min) || 1) * fraction;
    return { min: min - bump, max: max + bump };
  }
  const pad = span * fraction;
  return { min: min - pad, max: max + pad };
}

/** Horizontal inset for 1D line/scatter (fraction of index span per side). */
const TVIZ_1D_DOMAIN_PAD_X = 0.04;
/** Vertical inset: larger than X so peaks/troughs do not hug the top/bottom. */
const TVIZ_1D_DOMAIN_PAD_Y = 0.1;

function mapXIndex(plot: PlotRect, i: number, n: number): number {
  if (n <= 1) return plot.left + plot.w / 2;
  return plot.left + (i / (n - 1)) * plot.w;
}

function mapXLinear(plot: PlotRect, v: number, vmin: number, vmax: number): number {
  const s = vmax - vmin || 1;
  return plot.left + ((v - vmin) / s) * plot.w;
}

function mapYLinear(plot: PlotRect, v: number, vmin: number, vmax: number): number {
  const s = vmax - vmin || 1;
  return plot.top + plot.h - ((v - vmin) / s) * plot.h;
}

function AxesFrame({
  plot,
  xTicks,
  yTicks,
  yMin,
  yMax,
  formatX,
  formatY,
  mapX,
}: {
  plot: PlotRect;
  xTicks: number[];
  yTicks: number[];
  yMin: number;
  yMax: number;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  mapX: (v: number) => number;
}) {
  const x0 = plot.left;
  const y0 = plot.top + plot.h;
  const x1 = plot.left + plot.w;
  const y1 = plot.top;

  return (
    <g className="cr-tensor-viz__axes" aria-hidden>
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke={AXIS_STROKE} strokeWidth={1} />
      <line x1={x0} y1={y0} x2={x0} y2={y1} stroke={AXIS_STROKE} strokeWidth={1} />
      {yTicks.map((ty, yi) => {
        const py = mapYLinear(plot, ty, yMin, yMax);
        return (
          <g key={`y-${yi}-${ty}`}>
            <line x1={x0 - TICK_LEN} y1={py} x2={x0} y2={py} stroke={AXIS_STROKE} strokeWidth={1} />
            <text
              className="cr-tensor-viz__axis-tick"
              x={x0 - TICK_LEN - 2}
              y={py}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {formatY(ty)}
            </text>
          </g>
        );
      })}
      {xTicks.map((tx, xi) => {
        const px = mapX(tx);
        return (
          <g key={`x-${xi}-${tx}`}>
            <line x1={px} y1={y0} x2={px} y2={y0 + TICK_LEN} stroke={AXIS_STROKE} strokeWidth={1} />
            <text
              className="cr-tensor-viz__axis-tick"
              x={px}
              y={y0 + TICK_LEN + 9}
              textAnchor="middle"
              dominantBaseline="hanging"
            >
              {formatX(tx)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Viz1DBar({ values }: { values: number[] }) {
  const n = values.length;
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  const ySpan = vmax - vmin || 1;
  const plot: PlotRect = { left: 34, top: 6, w: CHART_W - 34 - 10, h: CHART_H_1D - 6 - 24 };
  const barGap = n > 80 ? 0 : n > 40 ? 0.5 : 1;
  const barW = Math.max(1, (plot.w - (n - 1) * barGap) / n);
  const yTicks = generateTicks(vmin, vmax, 4);
  const xTicks = indexXTicks(n);

  return (
    <svg
      className="cr-tensor-viz__svg nodrag nopan"
      width={CHART_W}
      height={CHART_H_1D}
      viewBox={`0 0 ${CHART_W} ${CHART_H_1D}`}
      aria-label="1D bar chart"
    >
      {values.map((v, i) => {
        const t = (v - vmin) / ySpan;
        const h = Math.max(1, t * plot.h);
        const x = plot.left + i * (barW + barGap);
        const y = plot.top + plot.h - h;
        return <rect key={i} x={x} y={y} width={barW} height={h} fill={heatColor(t)} rx={0.5} />;
      })}
      <AxesFrame
        plot={plot}
        xTicks={xTicks}
        yTicks={yTicks}
        yMin={vmin}
        yMax={vmax}
        formatX={(i) => String(i)}
        formatY={formatNumberTick}
        mapX={(i) => mapXIndex(plot, i, n)}
      />
    </svg>
  );
}

function Viz1DLine({ values, scatterOnly = false }: { values: number[]; scatterOnly?: boolean }) {
  const n = values.length;
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  const plot: PlotRect = { left: 34, top: 6, w: CHART_W - 34 - 10, h: CHART_H_1D - 6 - 24 };
  const xRawMin = 0;
  const xRawMax = Math.max(1, n - 1);
  const xPad = padLinearDomain(xRawMin, xRawMax, TVIZ_1D_DOMAIN_PAD_X);
  const yPad = padLinearDomain(vmin, vmax, TVIZ_1D_DOMAIN_PAD_Y);
  const baseBounds = { minX: xPad.min, maxX: xPad.max, minY: yPad.min, maxY: yPad.max };
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);
  const [clipId] = useState(
    () => `tviz-1d-${scatterOnly ? "scatter" : "line"}-clip-${Math.random().toString(36).slice(2, 9)}`,
  );
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
    bounds: baseBounds,
    enabled: n >= 2,
    left: plot.left,
    right: plot.left + plot.w,
    top: plot.top,
    bottom: plot.top + plot.h,
  });

  const coords: { x: number; y: number }[] = [];
  const spanX = viewBounds.maxX - viewBounds.minX || 1;
  const spanY = viewBounds.maxY - viewBounds.minY || 1;
  for (let i = 0; i < n; i++) {
    const x = plot.left + ((i - viewBounds.minX) / spanX) * plot.w;
    const t = (values[i]! - viewBounds.minY) / spanY;
    const y = plot.top + plot.h - t * plot.h;
    coords.push({ x, y });
  }
  const ptsStr = coords.map((p) => `${p.x},${p.y}`).join(" ");
  const pointR = n > 400 ? 1.2 : n > 150 ? 1.8 : 2.3;

  const yTicks = generateTicks(viewBounds.minY, viewBounds.maxY, 4);
  const xTicks = generateTicks(viewBounds.minX, viewBounds.maxX, 4);

  return (
    <div className="cr-tviz-chart-wrap">
      {isZoomed ? (
        <button type="button" className="cr-tviz-reset-zoom" onClick={resetZoom}>
          reset zoom-in
        </button>
      ) : null}
      <svg
        className="cr-tviz-chart cr-tensor-viz__svg nodrag nopan"
        width={CHART_W}
        height={CHART_H_1D}
        viewBox={`0 0 ${CHART_W} ${CHART_H_1D}`}
        aria-label={scatterOnly ? "1D scatter chart" : "1D line chart"}
        onMouseDown={(e) => {
          if (n < 2) return;
          setIsDraggingZoom(true);
          onMouseDown(e);
        }}
        onMouseMove={onMouseMove}
        onMouseUp={(e) => {
          setIsDraggingZoom(false);
          onMouseUp(e);
        }}
        onMouseLeave={(e) => {
          setIsDraggingZoom(false);
          onMouseLeave(e);
        }}
        style={{ cursor: n >= 2 ? (isDraggingZoom ? "crosshair" : "zoom-in") : "default" }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={plot.left} y={plot.top} width={plot.w} height={plot.h} />
          </clipPath>
        </defs>
        <rect
          x={plot.left}
          y={plot.top}
          width={plot.w}
          height={plot.h}
          rx={4}
          className="cr-tviz-chart__plot-bg"
        />
        {yTicks.map((yt) => {
          const py = mapYLinear(plot, yt, viewBounds.minY, viewBounds.maxY);
          return (
            <line
              key={`gy-${yt}`}
              x1={plot.left}
              y1={py}
              x2={plot.left + plot.w}
              y2={py}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        {xTicks.map((xt) => {
          const px = plot.left + ((xt - viewBounds.minX) / (viewBounds.maxX - viewBounds.minX || 1)) * plot.w;
          return (
            <line
              key={`gx-${xt}`}
              x1={px}
              y1={plot.top}
              x2={px}
              y2={plot.top + plot.h}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        {n > 0 && !scatterOnly ? (
          <polyline
            fill="none"
            className="cr-tviz-chart__line"
            strokeWidth={1.6}
            points={ptsStr}
            clipPath={`url(#${clipId})`}
          />
        ) : null}
        {n > 0 && scatterOnly
          ? coords.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={pointR}
                clipPath={`url(#${clipId})`}
                fill="color-mix(in srgb, var(--cr-accent-tensor) 55%, #1a1a22)"
                stroke="color-mix(in srgb, var(--cr-accent-tensor) 88%, #fff)"
                strokeWidth={0.45}
              />
            ))
          : null}
        {selectionRect ? (
          <rect
            x={selectionRect.x}
            y={selectionRect.y}
            width={selectionRect.w}
            height={selectionRect.h}
            className="cr-tviz-chart__zoom-box"
          />
        ) : null}
        <AxesFrame
          plot={plot}
          xTicks={xTicks}
          yTicks={yTicks}
          yMin={viewBounds.minY}
          yMax={viewBounds.maxY}
          formatX={(i) => (Math.abs(i - Math.round(i)) < 1e-6 ? String(Math.round(i)) : i.toFixed(1))}
          formatY={formatNumberTick}
          mapX={(i) => plot.left + ((i - viewBounds.minX) / (viewBounds.maxX - viewBounds.minX || 1)) * plot.w}
        />
      </svg>
    </div>
  );
}

function Viz1DHistogram({ values, bins }: { values: number[]; bins: number }) {
  const { counts, maxC, vmin, vmax } = histogramCounts(values, bins);
  const nbin = counts.length;
  const plot: PlotRect = { left: 34, top: 6, w: CHART_W - 34 - 10, h: CHART_H_1D - 6 - 24 };
  const gap = nbin > 60 ? 0 : 0.5;
  const span = vmax - vmin || 1;
  const binW = span / nbin;

  const yHi = Math.max(1, maxC);
  const yTicks = generateTicks(0, yHi, 4);
  const xTicks = generateTicks(vmin, vmax, 4);

  return (
    <svg
      className="cr-tensor-viz__svg nodrag nopan"
      width={CHART_W}
      height={CHART_H_1D}
      viewBox={`0 0 ${CHART_W} ${CHART_H_1D}`}
      aria-label="1D histogram"
    >
      {counts.map((c, i) => {
        const t = c / yHi;
        const h = Math.max(c > 0 ? 1 : 0, t * plot.h);
        const x0 = mapXLinear(plot, vmin + i * binW, vmin, vmax);
        const x1 = mapXLinear(plot, vmin + (i + 1) * binW, vmin, vmax);
        const bw = Math.max(1, x1 - x0 - gap);
        const x = x0 + gap / 2;
        const y = plot.top + plot.h - h;
        return <rect key={i} x={x} y={y} width={bw} height={h} fill={heatColor(t)} rx={0.5} />;
      })}
      <AxesFrame
        plot={plot}
        xTicks={xTicks}
        yTicks={yTicks}
        yMin={0}
        yMax={yHi}
        formatX={formatNumberTick}
        formatY={(c) => (Math.abs(c - Math.round(c)) < 1e-6 ? String(Math.round(c)) : formatNumberTick(c))}
        mapX={(v) => mapXLinear(plot, v, vmin, vmax)}
      />
    </svg>
  );
}

function extractScatterSeries(
  values: number[],
  rows: number,
  cols: number,
  axis: TensorViz2dScatterAxis,
  i1: number,
  i2: number,
): { xa: number[]; yb: number[]; err?: string } {
  if (axis === 1) {
    if (i1 < 0 || i1 >= cols || i2 < 0 || i2 >= cols) {
      return {
        xa: [],
        yb: [],
        err: `Column indices i1, i2 must be in 0..${cols - 1} (tensor shape [${rows}, ${cols}]).`,
      };
    }
    const xa = Array.from({ length: rows }, (_, r) => values[r * cols + i1]!);
    const yb = Array.from({ length: rows }, (_, r) => values[r * cols + i2]!);
    return { xa, yb };
  }
  if (i1 < 0 || i1 >= rows || i2 < 0 || i2 >= rows) {
    return {
      xa: [],
      yb: [],
      err: `Row indices i1, i2 must be in 0..${rows - 1} (tensor shape [${rows}, ${cols}]).`,
    };
  }
  const xa = Array.from({ length: cols }, (_, c) => values[i1 * cols + c]!);
  const yb = Array.from({ length: cols }, (_, c) => values[i2 * cols + c]!);
  return { xa, yb };
}

function ScatterAxesFrame({
  plot,
  xTicks,
  yTicks,
  xMin,
  xMax,
  yMin,
  yMax,
  formatTick,
}: {
  plot: PlotRect;
  xTicks: number[];
  yTicks: number[];
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  formatTick: (v: number) => string;
}) {
  const mapPx = (v: number) => mapXLinear(plot, v, xMin, xMax);
  const mapPy = (v: number) => mapYLinear(plot, v, yMin, yMax);
  const x0 = plot.left;
  const y0 = plot.top + plot.h;
  const x1 = plot.left + plot.w;
  const y1 = plot.top;

  return (
    <g className="cr-tensor-viz__axes" aria-hidden>
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke={AXIS_STROKE} strokeWidth={1} />
      <line x1={x0} y1={y0} x2={x0} y2={y1} stroke={AXIS_STROKE} strokeWidth={1} />
      {yTicks.map((ty, yi) => {
        const py = mapPy(ty);
        return (
          <g key={`sy-${yi}-${ty}`}>
            <line x1={x0 - TICK_LEN} y1={py} x2={x0} y2={py} stroke={AXIS_STROKE} strokeWidth={1} />
            <text
              className="cr-tensor-viz__axis-tick"
              x={x0 - TICK_LEN - 2}
              y={py}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {formatTick(ty)}
            </text>
          </g>
        );
      })}
      {xTicks.map((tx, xi) => {
        const px = mapPx(tx);
        return (
          <g key={`sx-${xi}-${tx}`}>
            <line x1={px} y1={y0} x2={px} y2={y0 + TICK_LEN} stroke={AXIS_STROKE} strokeWidth={1} />
            <text
              className="cr-tensor-viz__axis-tick"
              x={px}
              y={y0 + TICK_LEN + 9}
              textAnchor="middle"
              dominantBaseline="hanging"
            >
              {formatTick(tx)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Viz2DScatter({
  shape,
  values,
  axis,
  i1,
  i2,
}: {
  shape: number[];
  values: number[];
  axis: TensorViz2dScatterAxis;
  i1: number;
  i2: number;
}) {
  const rows = shape[0] ?? 0;
  const cols = shape[1] ?? 0;
  if (rows <= 0 || cols <= 0) return <p className="cr-tensor-viz__hint">Invalid 2-D shape.</p>;
  const expected = rows * cols;
  if (values.length !== expected) {
    return <p className="cr-tensor-viz__hint">Length mismatch for 2-D tensor.</p>;
  }

  const { xa, yb, err } = extractScatterSeries(values, rows, cols, axis, i1, i2);
  if (err) return <p className="cr-tensor-viz__hint">{err}</p>;
  if (xa.length === 0) return <p className="cr-tensor-viz__hint">No points to plot.</p>;

  const xmin = Math.min(...xa);
  const xmax = Math.max(...xa);
  const ymin = Math.min(...yb);
  const ymax = Math.max(...yb);
  const xPad = padLinearDomain(xmin, xmax, 0.055);
  const yPad = padLinearDomain(ymin, ymax, 0.055);

  const plot: PlotRect = { left: 36, top: 8, w: CHART_W - 36 - 10, h: SCATTER_CHART_H - 8 - 26 };
  const xTicks = generateTicks(xPad.min, xPad.max, 4);
  const yTicks = generateTicks(yPad.min, yPad.max, 4);
  const n = xa.length;
  const pr = n > 400 ? 1.6 : n > 150 ? 2.2 : 2.8;

  const pts = xa.map((xv, i) => {
    const px = mapXLinear(plot, xv, xPad.min, xPad.max);
    const py = mapYLinear(plot, yb[i]!, yPad.min, yPad.max);
    return { px, py, i };
  });

  return (
    <svg
      className="cr-tensor-viz__svg nodrag nopan"
      width={CHART_W}
      height={SCATTER_CHART_H}
      viewBox={`0 0 ${CHART_W} ${SCATTER_CHART_H}`}
      aria-label="2D tensor scatter"
    >
      {pts.map(({ px, py, i }) => (
        <circle
          key={i}
          cx={px}
          cy={py}
          r={pr}
          fill="color-mix(in srgb, var(--cr-accent-tensor) 55%, #1a1a22)"
          stroke="color-mix(in srgb, var(--cr-accent-tensor) 88%, #fff)"
          strokeWidth={0.45}
        />
      ))}
      <ScatterAxesFrame
        plot={plot}
        xTicks={xTicks}
        yTicks={yTicks}
        xMin={xPad.min}
        xMax={xPad.max}
        yMin={yPad.min}
        yMax={yPad.max}
        formatTick={formatNumberTick}
      />
    </svg>
  );
}

function tensorRankHint(resolved: Resolved): number | null {
  if (resolved.kind === "ok") return resolved.rank;
  if (resolved.kind === "lazy_activation") return resolved.shape.length;
  return null;
}

function headerTitle(mode: TensorVizMode, resolved: Resolved, displayRank: number | null): string {
  if (mode === "1d") return "1D tensor viz";
  if (mode === "2d") return "2D tensor viz";
  const r = displayRank ?? tensorRankHint(resolved);
  if (r === 0) return "0D tensor viz";
  if (r === 1) return "1D tensor viz";
  if (r === 2) return "2D tensor viz";
  return "General tensor viz";
}

function PlotStyleRadios({
  nodeId,
  value,
  onChange,
}: {
  nodeId: string;
  value: TensorViz1dPlotStyle;
  onChange: (v: TensorViz1dPlotStyle) => void;
}) {
  const name = `${nodeId}-plot1d-style`;
  const opt = (v: TensorViz1dPlotStyle, label: string) => (
    <label key={v} className="cr-tensor-viz__radio">
      <input type="radio" name={name} value={v} checked={value === v} onChange={() => onChange(v)} />
      <span>{label}</span>
    </label>
  );
  return (
    <div className="cr-tensor-viz__radio-row" role="group" aria-label="1D plot style">
      {opt("line", "line")}
      {opt("scatter", "scatter")}
      {opt("bar", "bar")}
      {opt("dist", "dist")}
    </div>
  );
}

function LineSortRadios({
  nodeId,
  value,
  onChange,
}: {
  nodeId: string;
  value: TensorViz1dLineSort;
  onChange: (v: TensorViz1dLineSort) => void;
}) {
  const name = `${nodeId}-plot1d-sort`;
  const opt = (v: TensorViz1dLineSort, label: string) => (
    <label key={v} className="cr-tensor-viz__radio">
      <input type="radio" name={name} value={v} checked={value === v} onChange={() => onChange(v)} />
      <span>{label}</span>
    </label>
  );
  return (
    <div className="cr-tensor-viz__radio-row" role="group" aria-label="Line plot sort order">
      {opt("original", "original")}
      {opt("descending", "descending")}
      {opt("ascending", "ascending")}
    </div>
  );
}

function Plot2DStyleRadios({
  nodeId,
  value,
  onChange,
}: {
  nodeId: string;
  value: TensorViz2dPlotStyle;
  onChange: (v: TensorViz2dPlotStyle) => void;
}) {
  const name = `${nodeId}-plot2d-style`;
  const opt = (v: TensorViz2dPlotStyle, label: string) => (
    <label key={v} className="cr-tensor-viz__radio">
      <input type="radio" name={name} value={v} checked={value === v} onChange={() => onChange(v)} />
      <span>{label}</span>
    </label>
  );
  return (
    <div className="cr-tensor-viz__radio-row" role="group" aria-label="2D plot style">
      {opt("scatter", "scatter")}
      {opt("heat", "heat")}
    </div>
  );
}

function ScatterSliceAxisRadios({
  nodeId,
  value,
  onChange,
}: {
  nodeId: string;
  value: TensorViz2dScatterAxis;
  onChange: (v: TensorViz2dScatterAxis) => void;
}) {
  const name = `${nodeId}-plot2d-scatter-axis`;
  return (
    <div className="cr-tensor-viz__radio-row" role="group" aria-label="Fixed index axis for scatter">
      <label className="cr-tensor-viz__radio">
        <input type="radio" name={name} checked={value === 0} onChange={() => onChange(0)} />
        <span>dimension 0</span>
      </label>
      <label className="cr-tensor-viz__radio">
        <input type="radio" name={name} checked={value === 1} onChange={() => onChange(1)} />
        <span>dimension 1</span>
      </label>
    </div>
  );
}

export function TensorVizNode({ id, selected, data, vizMode }: NodeProps & { vizMode: TensorVizMode }) {
  const [obsNameDraft, setObsNameDraft] = useState("");
  const [addObsBusy, setAddObsBusy] = useState(false);
  const [addObsError, setAddObsError] = useState<string | null>(null);

  const def = defaultTensorVizData();
  const raw = (data ?? {}) as Partial<TensorVizNodeData>;
  const d: TensorVizNodeData = {
    plot1dStyle: raw.plot1dStyle ?? def.plot1dStyle,
    plot1dLineSort: raw.plot1dLineSort ?? def.plot1dLineSort,
    histBins: raw.histBins ?? def.histBins,
    plot2dStyle: raw.plot2dStyle ?? def.plot2dStyle,
    plot2dScatterAxis: raw.plot2dScatterAxis ?? def.plot2dScatterAxis,
    plot2dScatterI1: raw.plot2dScatterI1 ?? def.plot2dScatterI1,
    plot2dScatterI2: raw.plot2dScatterI2 ?? def.plot2dScatterI2,
  };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const update = useCallback(
    (patch: Partial<TensorVizNodeData>) => patchTensorVizData(id, patch, setNodes),
    [id, setNodes],
  );

  const resolved = useStore(
    useCallback((state) => resolveTensor(state.nodes as Node[], state.edges as FlowEdge[], id), [id]),
    resolvedTensorEqual,
  );

  const { display, loading: tensorLoading } = useHydratedResolved(resolved);

  const displayRank = display.kind === "ok" ? display.rank : null;
  const title = useMemo(
    () => headerTitle(vizMode, resolved, displayRank),
    [vizMode, resolved, displayRank],
  );

  const isGeneralScalar =
    vizMode === "general" && display.kind === "ok" && display.rank === 0;

  const canAddUserObservable =
    isGeneralScalar && display.kind === "ok" && display.values.length > 0;

  const handleAddUserObservable = useCallback(async () => {
    if (!canAddUserObservable) return;
    setAddObsError(null);
    setAddObsBusy(true);
    try {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const res = await fetch("/api/user-observables", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tensor_viz_node_id: id,
          nodes: g.nodes,
          edges: g.edges,
          ...(obsNameDraft.trim() ? { label: obsNameDraft.trim() } : {}),
        }),
      });
      if (res.ok) {
        window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
        return;
      }
      const t = await res.text().catch(() => "");
      setAddObsError(t ? t.slice(0, 220) : `Request failed (${res.status})`);
    } catch {
      setAddObsError("Network error while saving observable.");
    } finally {
      setAddObsBusy(false);
    }
  }, [canAddUserObservable, getEdges, getNodes, id, obsNameDraft]);

  const show1dControls =
    (vizMode === "general" || vizMode === "1d") && display.kind === "ok" && display.rank === 1;

  const show2dControls =
    (vizMode === "general" || vizMode === "2d") && display.kind === "ok" && display.rank === 2;

  const scatterIdxMax =
    display.kind === "ok" && display.rank === 2
      ? d.plot2dScatterAxis === 1
        ? Math.max(0, (display.shape[1] ?? 1) - 1)
        : Math.max(0, (display.shape[0] ?? 1) - 1)
      : 0;

  const body = useMemo(() => {
    if (display.kind === "none") {
      if (tensorLoading) {
        return <p className="cr-tensor-viz__hint">Loading tensor from server…</p>;
      }
      return <p className="cr-tensor-viz__hint">{display.detail}</p>;
    }
    const { rank, shape, values, sourceSummary } = display;

    if (vizMode === "1d" && rank !== 1) {
      return (
        <p className="cr-tensor-viz__hint">
          This node expects a 1-D tensor (e.g. PCA “explained variance ratio”). Connected: rank {rank}
          {shape.length ? `, shape [${shape.join(", ")}]` : ""}.
        </p>
      );
    }
    if (vizMode === "2d" && rank !== 2) {
      return (
        <p className="cr-tensor-viz__hint">
          This node expects a 2-D tensor (e.g. PCA “principal components”). Connected: rank {rank}
          {shape.length ? `, shape [${shape.join(", ")}]` : ""}.
        </p>
      );
    }

    if (vizMode === "general") {
      if (rank > 2) {
        return (
          <p className="cr-tensor-viz__hint">
            Rank {rank} tensor (shape [{shape.join(", ")}]) — only scalar, 1-D, and 2-D views are supported.
          </p>
        );
      }
      if (rank === 0) {
        if (values.length === 0) {
          return <p className="cr-tensor-viz__hint">Scalar tensor has no stored value.</p>;
        }
        const v = values[0]!;
        return (
          <div className="cr-tensor-viz__plot cr-tensor-viz-0d">
            <div className="cr-tensor-viz-0d__value" title={String(v)}>
              {formatTensorScalarDisplay(v)}
            </div>
            <p className="cr-tensor-viz__meta">{sourceSummary} · []</p>
          </div>
        );
      }
    }

    if (rank === 1) {
      const style = d.plot1dStyle;
      const lineSeries = ordered1DValues(values, d.plot1dLineSort);
      let chart: ReactNode;
      if (style === "bar") {
        chart = <Viz1DBar values={values} />;
      } else if (style === "dist") {
        chart = <Viz1DHistogram values={values} bins={d.histBins} />;
      } else if (style === "scatter") {
        chart = <Viz1DLine values={lineSeries} scatterOnly />;
      } else {
        chart = <Viz1DLine values={lineSeries} />;
      }

      return (
        <div className="cr-tensor-viz__plot">
          {chart}
          <p className="cr-tensor-viz__meta">
            {sourceSummary} · [{shape.join(" × ")}]
          </p>
        </div>
      );
    }
    if (rank === 2) {
      const rows = shape[0] ?? 0;
      const cols = shape[1] ?? 0;
      const style2d = d.plot2dStyle;
      let chart: ReactNode;
      let metaExtra: string;
      if (style2d === "scatter") {
        chart = (
          <Viz2DScatter
            shape={shape}
            values={values}
            axis={d.plot2dScatterAxis}
            i1={d.plot2dScatterI1}
            i2={d.plot2dScatterI2}
          />
        );
        const ax = d.plot2dScatterAxis;
        metaExtra =
          ax === 1
            ? `scatter: X[:, ${d.plot2dScatterI1}] vs X[:, ${d.plot2dScatterI2}]`
            : `scatter: X[${d.plot2dScatterI1}, :] vs X[${d.plot2dScatterI2}, :]`;
      } else {
        chart = <TensorHeatmap shape={shape} values={values} />;
        const truncated = rows > HEAT_MAX_DIM || cols > HEAT_MAX_DIM;
        metaExtra = truncated
          ? `heatmap: first ${HEAT_MAX_DIM}×${HEAT_MAX_DIM} of [${rows}×${cols}]`
          : "heatmap";
      }
      return (
        <div className="cr-tensor-viz__plot">
          {chart}
          <p className="cr-tensor-viz__meta">
            {sourceSummary} · [{shape.join(" × ")}] · {metaExtra}
          </p>
        </div>
      );
    }
    return <p className="cr-tensor-viz__hint">Unsupported rank for visualization.</p>;
  }, [
    display,
    tensorLoading,
    vizMode,
    d.plot1dStyle,
    d.plot1dLineSort,
    d.histBins,
    d.plot2dStyle,
    d.plot2dScatterAxis,
    d.plot2dScatterI1,
    d.plot2dScatterI2,
  ]);

  return (
    <div
      className={`cr-node cr-node--tensor-viz${isGeneralScalar ? " cr-node--tensor-viz-0d" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">{title}</div>
      {isGeneralScalar ? (
        <TensorVizObsAddStrip
          nodeId={id}
          nameDraft={obsNameDraft}
          onNameChange={setObsNameDraft}
          canAdd={canAddUserObservable}
          busy={addObsBusy}
          onAdd={handleAddUserObservable}
        />
      ) : null}
      <div className="cr-node__body cr-node__body--compact">
        {isGeneralScalar && addObsError ? (
          <p className="cr-tensor-viz__hint" role="alert">
            {addObsError}
          </p>
        ) : null}
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Tensor viz input and outputs">
          <VizSocketsBar compareSource={vizMode === "1d"} />
        </div>
        {show1dControls ? (
          <div className="cr-tensor-viz__controls nodrag nopan">
            <div className="cr-tensor-viz__control-block">
              <span className="cr-tensor-viz__control-label">plot</span>
              <PlotStyleRadios nodeId={id} value={d.plot1dStyle} onChange={(v) => update({ plot1dStyle: v })} />
            </div>
            {d.plot1dStyle === "line" || d.plot1dStyle === "scatter" ? (
              <div className="cr-tensor-viz__control-block">
                <span className="cr-tensor-viz__control-label">order</span>
                <LineSortRadios
                  nodeId={id}
                  value={d.plot1dLineSort}
                  onChange={(v) => update({ plot1dLineSort: v })}
                />
              </div>
            ) : null}
            {d.plot1dStyle === "dist" ? (
              <ComfyIntField
                label="bins"
                value={d.histBins}
                min={1}
                max={500}
                title="Histogram bin count"
                onCommit={(n) => update({ histBins: n })}
                ariaLabel="Histogram bin count"
              />
            ) : null}
          </div>
        ) : null}
        {show2dControls ? (
          <div className="cr-tensor-viz__controls nodrag nopan">
            <div className="cr-tensor-viz__control-block">
              <span className="cr-tensor-viz__control-label">2D</span>
              <Plot2DStyleRadios
                nodeId={id}
                value={d.plot2dStyle}
                onChange={(v) => update({ plot2dStyle: v })}
              />
            </div>
            {d.plot2dStyle === "scatter" ? (
              <>
                <div className="cr-tensor-viz__control-block">
                  <span className="cr-tensor-viz__control-label">axis</span>
                  <ScatterSliceAxisRadios
                    nodeId={id}
                    value={d.plot2dScatterAxis}
                    onChange={(v) => update({ plot2dScatterAxis: v })}
                  />
                </div>
                <ComfyIntField
                  label="i1"
                  value={d.plot2dScatterI1}
                  min={0}
                  max={scatterIdxMax}
                  title="Index along the varying dimension (first series)"
                  onCommit={(n) => update({ plot2dScatterI1: n })}
                  ariaLabel="Scatter index i1"
                />
                <ComfyIntField
                  label="i2"
                  value={d.plot2dScatterI2}
                  min={0}
                  max={scatterIdxMax}
                  title="Index along the varying dimension (second series)"
                  onCommit={(n) => update({ plot2dScatterI2: n })}
                  ariaLabel="Scatter index i2"
                />
              </>
            ) : null}
          </div>
        ) : null}
        <div className="cr-tensor-viz__body nodrag nopan">{body}</div>
      </div>
    </div>
  );
}

export function makeTensorVizComponent(mode: TensorVizMode) {
  return function BoundTensorVizNode(props: NodeProps) {
    return <TensorVizNode {...props} vizMode={mode} />;
  };
}
