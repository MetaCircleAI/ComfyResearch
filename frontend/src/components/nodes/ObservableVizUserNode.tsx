import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useMemo, useState } from "react";
import {
  defaultObservableVizUserData,
  type ObservableVizUserNodeData,
} from "./observableVizUserDefaults";
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

/** Algebra / user scalar observables: plain “Value”, not reduction-chain notation (legacy ``mean|·|``). */
function resolveUserObservableVizYAxisLabel(raw: string | undefined): string {
  const t = raw?.trim() ?? "";
  if (!t) return "Value";
  if (t.toLowerCase() === "value") return "Value";
  if (t === "mean|·|" || t === "mean|.|") return "Value";
  if (t.includes("|")) return "Value";
  return t;
}

/** Matches Hessian / gradient viz multi-series strokes. */
/* Theme-aware via tokens.css; classic values pin the exact legacy hex per position. */
const SERIES_COLORS = ["var(--cr-chart-1)", "var(--cr-chart-2)", "var(--cr-chart-3)", "var(--cr-chart-4)", "var(--cr-chart-9)", "var(--cr-chart-6)", "var(--cr-chart-7)", "var(--cr-chart-5)"];

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function isFinitePlotNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function patchData(
  id: string,
  patch: Partial<ObservableVizUserNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultObservableVizUserData();
      const cur = (n.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const prev: ObservableVizUserNodeData = {
        pairedObservableId: cur.pairedObservableId,
        pairedTrainerId: cur.pairedTrainerId,
        observableName: cur.observableName ?? def.observableName,
        vizYAxisLabel: cur.vizYAxisLabel,
        lastSweepSummary: cur.lastSweepSummary,
        valueHistory: cur.valueHistory,
        valueHistories: cur.valueHistories,
        seriesLabels: cur.seriesLabels,
        testValueHistory: cur.testValueHistory,
        stepTicks: cur.stepTicks,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
        multiSeriesVisible: cur.multiSeriesVisible,
        showSeries: cur.showSeries ?? def.showSeries ?? true,
        showTrainCurve: cur.showTrainCurve ?? def.showTrainCurve,
        showTestCurve: cur.showTestCurve ?? def.showTestCurve,
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
      const raw = values[i];
      if (!isFinitePlotNumber(raw)) continue;
      const x = transformStep(steps[i]!, logX);
      const y = transformY(raw, logY);
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

function computeBoundsMulti(
  steps: number[],
  seriesList: number[][],
  seriesVisible: boolean[],
  logX: boolean,
  logY: boolean,
): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  if (steps.length >= 2) {
    for (let si = 0; si < seriesList.length; si++) {
      if (seriesVisible[si] === false) continue;
      const values = seriesList[si]!;
      if (values.length !== steps.length) continue;
      for (let i = 0; i < steps.length; i++) {
        const raw = values[i];
        if (!isFinitePlotNumber(raw)) continue;
        const x = transformStep(steps[i]!, logX);
        const y = transformY(raw, logY);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        any = true;
      }
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

function computeBoundsDual(
  steps: number[],
  train: number[],
  test: number[] | undefined,
  showTrain: boolean,
  showTest: boolean,
  logX: boolean,
  logY: boolean,
): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  const consider = (values: number[]) => {
    if (values.length !== steps.length) return;
    for (let i = 0; i < steps.length; i++) {
      const raw = values[i];
      if (!isFinitePlotNumber(raw)) continue;
      const x = transformStep(steps[i]!, logX);
      const y = transformY(raw, logY);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      any = true;
    }
  };
  if (showTrain) consider(train);
  if (showTest && test) consider(test);
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
  let d = "";
  let penUp = true;
  for (let i = 0; i < steps.length; i++) {
    const raw = values[i];
    if (!isFinitePlotNumber(raw)) {
      penUp = true;
      continue;
    }
    const x = transformStep(steps[i]!, logX);
    const y = transformY(raw, logY);
    const px = PAD_L + (innerW * (x - b.minX)) / spanX;
    const py = PAD_T + innerH * (1 - (y - b.minY) / spanY);
    d += `${penUp ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    penUp = false;
  }
  return d;
}

function xToPx(x: number, b: Bounds): number {
  const innerW = CHART_W - PAD_L - PAD_R;
  return PAD_L + (innerW * (x - b.minX)) / (b.maxX - b.minX || 1);
}

function yToPx(y: number, b: Bounds): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

export function ObservableVizUserNode({ id, data, selected }: NodeProps) {
  const def = defaultObservableVizUserData();
  const raw = (data ?? {}) as Partial<ObservableVizUserNodeData>;
  const d: ObservableVizUserNodeData = {
    pairedObservableId: raw.pairedObservableId,
    pairedTrainerId: raw.pairedTrainerId,
    observableName: raw.observableName ?? def.observableName,
    vizYAxisLabel: raw.vizYAxisLabel,
    lastSweepSummary: raw.lastSweepSummary,
    valueHistory: raw.valueHistory,
    valueHistories: raw.valueHistories,
    seriesLabels: raw.seriesLabels,
    testValueHistory: raw.testValueHistory,
    stepTicks: raw.stepTicks,
    logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
    logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
    multiSeriesVisible: raw.multiSeriesVisible,
    showSeries: raw.showSeries ?? def.showSeries ?? true,
    showTrainCurve: raw.showTrainCurve ?? def.showTrainCurve,
    showTestCurve: raw.showTestCurve ?? def.showTestCurve,
    zoomXMin: raw.zoomXMin,
    zoomXMax: raw.zoomXMax,
  };

  const vizInstanceTitle = (data as Record<string, unknown> | undefined)?.instanceTitle;
  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId, vizInstanceTitle);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<ObservableVizUserNodeData>) => patchData(id, patch, setNodes);
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);

  const seriesList = useMemo(() => {
    const vh = d.valueHistories;
    if (!Array.isArray(vh) || vh.length === 0) return [] as number[][];
    return vh
      .filter((row): row is unknown[] => Array.isArray(row))
      .map((row) =>
        row.map((cell) => {
          const n = Number(cell);
          return Number.isFinite(n) ? n : Number.NaN;
        }),
      );
  }, [d.valueHistories]);

  const seriesLabelsResolved = useMemo(
    () =>
      Array.isArray(d.seriesLabels)
        ? d.seriesLabels.filter((x): x is string => typeof x === "string")
        : [],
    [d.seriesLabels],
  );

  const seriesVisibleEffective = useMemo(() => {
    const n = seriesList.length;
    const out = Array.from({ length: n }, () => true);
    const rawVis = d.multiSeriesVisible;
    if (Array.isArray(rawVis)) {
      for (let i = 0; i < n; i++) out[i] = rawVis[i] !== false;
    } else if (d.showSeries === false) {
      for (let i = 0; i < n; i++) out[i] = false;
    }
    return out;
  }, [seriesList.length, d.multiSeriesVisible, d.showSeries]);

  const stepsMulti = useMemo(() => {
    const st = d.stepTicks;
    if (!Array.isArray(st) || st.length < 2) return [] as number[];
    if (seriesList.length === 0) return [];
    const n = st.length;
    if (!seriesList.every((s) => s.length === n)) return [];
    return st.map(Number);
  }, [d.stepTicks, seriesList]);

  const hasMultiSeries =
    seriesList.length > 0 &&
    stepsMulti.length >= 2 &&
    seriesList.every((s) => s.length === stepsMulti.length);

  const isLiveSingle = !!(
    d.valueHistory &&
    d.stepTicks &&
    d.valueHistory.length >= 2 &&
    d.stepTicks.length === d.valueHistory.length
  );

  const isLive = isLiveSingle || hasMultiSeries;

  const hasDual = !!(
    !hasMultiSeries &&
    isLiveSingle &&
    d.testValueHistory &&
    d.stepTicks &&
    d.testValueHistory.length === d.stepTicks.length &&
    d.testValueHistory.length >= 2
  );

  const steps = useMemo(() => {
    if (hasMultiSeries) return stepsMulti;
    if (isLiveSingle && d.stepTicks) return d.stepTicks.map(Number);
    return [];
  }, [hasMultiSeries, stepsMulti, isLiveSingle, d.stepTicks]);

  const vals = useMemo(() => {
    if (!isLiveSingle || hasMultiSeries) return [];
    return d.valueHistory ?? [];
  }, [isLiveSingle, hasMultiSeries, d.valueHistory]);

  const testVals = useMemo(() => {
    if (hasDual && d.testValueHistory) return d.testValueHistory;
    return undefined;
  }, [hasDual, d.testValueHistory]);

  const showTr = hasDual ? !!d.showTrainCurve : !!d.showSeries;
  const showTe = hasDual && !!d.showTestCurve;

  const anyMultiFinite = useMemo(
    () =>
      seriesList.some(
        (row, si) => seriesVisibleEffective[si] !== false && row.some((v) => Number.isFinite(v)),
      ),
    [seriesList, seriesVisibleEffective],
  );

  const bounds = useMemo(() => {
    if (hasMultiSeries) {
      return computeBoundsMulti(
        stepsMulti,
        seriesList,
        seriesVisibleEffective,
        !!d.logScaleX,
        !!d.logScaleY,
      );
    }
    if (hasDual && d.testValueHistory) {
      return computeBoundsDual(
        steps,
        vals,
        d.testValueHistory,
        !!d.showTrainCurve,
        !!d.showTestCurve,
        !!d.logScaleX,
        !!d.logScaleY,
      );
    }
    return computeBounds(steps, vals, showTr, !!d.logScaleX, !!d.logScaleY);
  }, [
    hasMultiSeries,
    stepsMulti,
    seriesList,
    seriesVisibleEffective,
    hasDual,
    d.testValueHistory,
    d.showTrainCurve,
    d.showTestCurve,
    steps,
    vals,
    showTr,
    d.logScaleX,
    d.logScaleY,
  ]);
  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - PAD_R;
  const anyMultiCurveRequested = hasMultiSeries && seriesList.some((_, si) => seriesVisibleEffective[si] !== false);

  const hasSeries = hasMultiSeries
    ? anyMultiCurveRequested &&
      stepsMulti.length >= 2 &&
      seriesList.length > 0 &&
      seriesList.every((s) => s.length === stepsMulti.length) &&
      anyMultiFinite
    : hasDual
      ? (d.showTrainCurve && vals.length >= 2 && vals.length === steps.length) ||
        (!!d.showTestCurve && !!testVals && testVals.length >= 2 && testVals.length === steps.length)
      : !!d.showSeries && vals.length >= 2 && vals.length === steps.length;
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

  const trainPath = useMemo(() => {
    if (hasMultiSeries) return "";
    if (hasDual) {
      if (!d.showTrainCurve) return "";
    } else if (!d.showSeries) {
      return "";
    }
    return buildPath(steps, vals, !!d.logScaleX, !!d.logScaleY, viewBounds);
  }, [hasMultiSeries, hasDual, d.showSeries, d.showTrainCurve, d.logScaleX, d.logScaleY, steps, vals, viewBounds]);

  const testPath = useMemo(() => {
    if (hasMultiSeries || !hasDual || !d.showTestCurve || !testVals) return "";
    return buildPath(steps, testVals, !!d.logScaleX, !!d.logScaleY, viewBounds);
  }, [hasMultiSeries, hasDual, d.showTestCurve, d.logScaleX, d.logScaleY, steps, testVals, viewBounds]);

  const multiPaths = useMemo(() => {
    if (!hasMultiSeries) return [] as string[];
    return seriesList.map((ser, si) =>
      seriesVisibleEffective[si] === false
        ? ""
        : buildPath(stepsMulti, ser, !!d.logScaleX, !!d.logScaleY, viewBounds),
    );
  }, [hasMultiSeries, seriesList, seriesVisibleEffective, stepsMulti, d.logScaleX, d.logScaleY, viewBounds]);

  const plotMidX = PAD_L + (CHART_W - PAD_L - PAD_R) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${id}-ou-clip`;

  const allMultiHidden =
    hasMultiSeries &&
    seriesList.length > 0 &&
    stepsMulti.length >= 2 &&
    !seriesList.some((_, si) => seriesVisibleEffective[si] !== false);

  const toggleMultiSeriesVisible = (idx: number, vis: boolean) => {
    const n = seriesList.length;
    if (n <= 0) return;
    const next = Array.from({ length: n }, (i) => seriesVisibleEffective[i] !== false);
    next[idx] = vis;
    update({ multiSeriesVisible: next });
  };

  const multiSeriesRowLabel = (i: number) => {
    const lab = seriesLabelsResolved[i] ?? `series ${i + 1}`;
    return lab.length > 18 ? `${lab.slice(0, 17)}…` : lab;
  };

  const hint = isLive
    ? hasMultiSeries
      ? allMultiHidden
        ? "All series are hidden — enable at least one curve above the chart."
        : "Multiple series from last training run; invalid points break strokes."
      : hasDual
        ? "Train / test top-1 accuracy from last training run"
        : "Scalar series from last training run (missing / invalid steps omitted from bounds and path)."
    : "No data yet — connect Trainer “observable”, then Train.";

  const yAxisLabel = hasDual ? "top-1 acc" : resolveUserObservableVizYAxisLabel(d.vizYAxisLabel);

  const multiLegendMax = 10;

  return (
    <div
      className={`cr-node cr-node--observable-viz-user${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar annotatorSource compareSource />
        <div className="cr-tviz-chart-divider" aria-hidden />
        <div className="cr-tviz-chart-controls nodrag nopan">
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
        {hasDual ? (
          <div className="cr-tviz-chart-controls nodrag nopan">
            <label className="cr-tviz-check cr-tviz-check--train">
              <input
                type="checkbox"
                checked={!!d.showTrainCurve}
                onChange={(e) => update({ showTrainCurve: e.target.checked })}
              />
              train
            </label>
            <label className="cr-tviz-check cr-tviz-check--test">
              <input
                type="checkbox"
                checked={!!d.showTestCurve}
                onChange={(e) => update({ showTestCurve: e.target.checked })}
              />
              test
            </label>
          </div>
        ) : hasMultiSeries ? null : (
          <div className="cr-tviz-chart-controls nodrag nopan">
            <label className="cr-tviz-check cr-tviz-check--train">
              <input
                type="checkbox"
                checked={!!d.showSeries}
                onChange={(e) => update({ showSeries: e.target.checked })}
              />
              series
            </label>
          </div>
        )}
        {hasMultiSeries && seriesList.length > 0 ? (
          <div className="cr-tviz-chart-controls cr-tviz-eigen-ranks-row nodrag nopan" aria-label="Series curves">
            {seriesList.map((_, si) => (
              <label key={`msv-${si}`} className="cr-tviz-check cr-tviz-check--eigen-rank">
                <input
                  type="checkbox"
                  checked={seriesVisibleEffective[si] !== false}
                  onChange={(e) => toggleMultiSeriesVisible(si, e.target.checked)}
                />
                {multiSeriesRowLabel(si)}
              </label>
            ))}
          </div>
        ) : null}
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
            aria-label="User observable metric over training"
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

            {testPath ? (
              <path
                d={testPath}
                fill="none"
                className="cr-tviz-chart__line cr-tviz-chart__line--test"
                strokeWidth={1.5}
                clipPath={`url(#${clipId})`}
              />
            ) : null}
            {hasMultiSeries
              ? multiPaths.map((md, i) =>
                  md ? (
                    <path
                      key={`multi-${i}`}
                      d={md}
                      fill="none"
                      className="cr-tviz-chart__series-line"
                      stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                      strokeWidth={1.5}
                      clipPath={`url(#${clipId})`}
                    />
                  ) : null,
                )
              : null}
            {trainPath ? (
              <path
                d={trainPath}
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
              {yAxisLabel}
            </text>

            {hasDual && (trainPath || testPath) ? (
              <g className="cr-tviz-legend" transform={`translate(${innerRight - 52}, ${PAD_T + 4})`}>
                {trainPath ? (
                  <>
                    <line x1={0} y1={5} x2={14} y2={5} className="cr-tviz-chart__line" strokeWidth={1.6} />
                    <text x={17} y={7} className="cr-tviz-chart__legend-text">
                      train
                    </text>
                  </>
                ) : null}
                {testPath ? (
                  <g transform={trainPath ? "translate(0, 11)" : ""}>
                    <line
                      x1={0}
                      y1={5}
                      x2={14}
                      y2={5}
                      className="cr-tviz-chart__line cr-tviz-chart__line--test"
                      strokeWidth={1.5}
                    />
                    <text x={17} y={7} className="cr-tviz-chart__legend-text">
                      test
                    </text>
                  </g>
                ) : null}
              </g>
            ) : hasMultiSeries && multiPaths.some(Boolean) ? (
              <g className="cr-tviz-legend" transform={`translate(${PAD_L + 4}, ${PAD_T + 4})`}>
                {seriesList.map((_, i) => {
                  if (i >= multiLegendMax || seriesVisibleEffective[i] === false) return null;
                  const md = multiPaths[i];
                  if (!md) return null;
                  const lab = seriesLabelsResolved[i] ?? `s${i}`;
                  const short = lab.length > 11 ? `${lab.slice(0, 10)}…` : lab;
                  const row = seriesList
                    .slice(0, i)
                    .filter((_, j) => seriesVisibleEffective[j] !== false && !!multiPaths[j]).length;
                  return (
                    <g key={`mlg-${i}`} transform={`translate(0, ${row * 12})`}>
                      <line
                        x1={0}
                        y1={5}
                        x2={12}
                        y2={5}
                        className="cr-tviz-chart__series-line"
                        stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                        strokeWidth={1.5}
                      />
                      <text x={14} y={7} className="cr-tviz-chart__legend-text" style={{ fontSize: 9 }}>
                        {short}
                      </text>
                    </g>
                  );
                })}
                {seriesList.filter((_, j) => seriesVisibleEffective[j] !== false && !!multiPaths[j]).length >
                multiLegendMax ? (
                  <text
                    x={0}
                    y={multiLegendMax * 12 + 4}
                    className="cr-tviz-chart__legend-text"
                    style={{ fontSize: 8 }}
                  >
                    +
                    {seriesList.filter((_, j) => seriesVisibleEffective[j] !== false && !!multiPaths[j]).length -
                      multiLegendMax}{" "}
                    more
                  </text>
                ) : null}
              </g>
            ) : trainPath ? (
              <g className="cr-tviz-legend" transform={`translate(${innerRight - 52}, ${PAD_T + 4})`}>
                <line x1={0} y1={5} x2={14} y2={5} className="cr-tviz-chart__line" strokeWidth={1.6} />
                <text x={17} y={7} className="cr-tviz-chart__legend-text">
                  obs
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
