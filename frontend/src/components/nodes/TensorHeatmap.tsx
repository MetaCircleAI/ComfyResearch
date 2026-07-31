import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from "react";

const CHART_W = 228;
const CHART_H = 140;
export const HEAT_MAX_DIM = 100;

type AxisLabels = { x: string[]; y: string[]; xTitle: string; yTitle: string };
type HoverCell = { row: number; col: number; value: number };

function heatColor(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const r = Math.round(32 + x * 200);
  const g = Math.round(48 + (1 - Math.abs(x - 0.45) * 1.4) * 160);
  const b = Math.round(120 + (1 - x) * 110);
  return `rgb(${r},${g},${b})`;
}

/** Evenly spaced ticks that retain the first and last visible index. */
export function heatmapTickIndices(count: number, availablePx: number, labelChars: number): number[] {
  if (count <= 0) return [];
  if (count === 1) return [0];
  const labelPx = Math.max(18, labelChars * 7 + 5);
  const max = Math.max(2, Math.floor(availablePx / labelPx));
  if (count <= max) return Array.from({ length: count }, (_, index) => index);
  const out = new Set<number>();
  for (let i = 0; i < max; i++) out.add(Math.round((i * (count - 1)) / (max - 1)));
  return [...out].sort((a, b) => a - b);
}

/** Shared presentation-only heatmap for rank-2 tensor and attention-map views. */
export function TensorHeatmap({
  shape,
  values,
  axisLabels,
  onHoverCell,
}: {
  shape: number[];
  values: number[];
  axisLabels?: AxisLabels;
  onHoverCell?: (cell: HoverCell | null) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [cssWidth, setCssWidth] = useState(CHART_W);
  useEffect(() => {
    const element = svgRef.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setCssWidth(entry.contentRect.width || CHART_W));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const rows = shape[0] ?? 0;
  const cols = shape[1] ?? 0;
  if (rows <= 0 || cols <= 0) return <p className="cr-tensor-viz__hint">Invalid 2-D shape.</p>;
  if (values.length !== rows * cols) return <p className="cr-tensor-viz__hint">Length mismatch for 2-D tensor.</p>;
  const displayRows = Math.min(HEAT_MAX_DIM, rows);
  const displayCols = Math.min(HEAT_MAX_DIM, cols);
  const xLabels = axisLabels?.x.slice(0, displayCols) ?? [];
  const yLabels = axisLabels?.y.slice(0, displayRows) ?? [];
  const hasAxes = !!axisLabels;
  const longestX = Math.max(1, ...xLabels.map((label) => label.length));
  const rotateX = hasAxes && (displayCols > 8 || longestX > 3);
  const plot = hasAxes
    ? { left: 38, right: 8, top: 8, bottom: rotateX ? 43 : 31 }
    : { left: 0, right: 0, top: 4, bottom: 0 };
  const plotW = CHART_W - plot.left - plot.right;
  const plotH = CHART_H - plot.top - plot.bottom;
  // Keep generic tensor heatmaps pixel-for-pixel compatible with their prior layout.
  const genericCell = Math.max(2, Math.min(Math.min(14, Math.floor(CHART_W / displayCols)), Math.min(14, Math.floor(CHART_H / displayRows))));
  const cellW = hasAxes ? plotW / displayCols : genericCell;
  const cellH = hasAxes ? plotH / displayRows : genericCell;
  const cellLeft = hasAxes ? plot.left : Math.max(0, (CHART_W - cellW * displayCols) / 2);
  const cellTop = hasAxes ? plot.top : 4;
  let min = Infinity;
  let max = -Infinity;
  for (let row = 0; row < displayRows; row++) for (let col = 0; col < displayCols; col++) {
    const value = values[row * cols + col]!;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const span = max - min || 1;
  const cells: ReactNode[] = [];
  for (let row = 0; row < displayRows; row++) for (let col = 0; col < displayCols; col++) {
    const value = values[row * cols + col]!;
    cells.push(<rect key={`${row}-${col}`} x={cellLeft + col * cellW} y={cellTop + row * cellH} width={Math.max(0, cellW - 0.35)} height={Math.max(0, cellH - 0.35)} fill={heatColor((value - min) / span)} />);
  }
  const xTicks = heatmapTickIndices(displayCols, cssWidth * (plotW / CHART_W), longestX);
  const yTicks = heatmapTickIndices(displayRows, cssWidth * (plotH / CHART_W), Math.max(1, ...yLabels.map((label) => label.length)));
  const hoverAt = (event: PointerEvent<SVGSVGElement>) => {
    if (!onHoverCell) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * CHART_W) / rect.width;
    const y = ((event.clientY - rect.top) * CHART_H) / rect.height;
    const col = Math.floor((x - cellLeft) / cellW);
    const row = Math.floor((y - cellTop) / cellH);
    if (row < 0 || row >= displayRows || col < 0 || col >= displayCols) onHoverCell(null);
    else onHoverCell({ row, col, value: values[row * cols + col]! });
  };
  return <svg ref={svgRef} className="cr-tensor-viz__svg nodrag nopan" width={CHART_W} height={CHART_H} viewBox={`0 0 ${CHART_W} ${CHART_H}`} aria-label="2D tensor heatmap" onPointerMove={hoverAt} onPointerLeave={() => onHoverCell?.(null)}>
    {cells}
    {hasAxes ? <>
      <line className="cr-tviz-chart__axis-line" x1={plot.left} y1={plot.top + plotH} x2={plot.left + plotW} y2={plot.top + plotH} />
      <line className="cr-tviz-chart__axis-line" x1={plot.left} y1={plot.top} x2={plot.left} y2={plot.top + plotH} />
      {xTicks.map((index) => { const x = plot.left + (index + 0.5) * cellW; return <g key={`x-${index}`}><line className="cr-tviz-chart__tick" x1={x} y1={plot.top + plotH} x2={x} y2={plot.top + plotH + 3} /><text className="cr-tviz-chart__tick-label" textAnchor={rotateX ? "end" : "middle"} transform={rotateX ? `rotate(-45 ${x} ${plot.top + plotH + 7})` : undefined} x={x} y={plot.top + plotH + 10}>{xLabels[index] ?? String(index)}</text></g>; })}
      {yTicks.map((index) => { const y = plot.top + (index + 0.5) * cellH + 3; return <g key={`y-${index}`}><line className="cr-tviz-chart__tick" x1={plot.left - 3} y1={y - 3} x2={plot.left} y2={y - 3} /><text className="cr-tviz-chart__tick-label" textAnchor="end" x={plot.left - 5} y={y}>{yLabels[index] ?? String(index)}</text></g>; })}
      <text className="cr-tviz-chart__axis-title" textAnchor="middle" x={plot.left + plotW / 2} y={CHART_H - 2}>{axisLabels.xTitle}</text>
      <text className="cr-tviz-chart__axis-title" textAnchor="middle" transform={`rotate(-90 11 ${plot.top + plotH / 2})`} x="11" y={plot.top + plotH / 2}>{axisLabels.yTitle}</text>
    </> : null}
  </svg>;
}
