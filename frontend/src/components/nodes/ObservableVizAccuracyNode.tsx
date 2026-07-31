import { useReactFlow, useStore, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import {
  defaultObservableVizUserData,
  type ObservableVizUserNodeData,
} from "./observableVizUserDefaults";
import { useObservableVizHeaderTitle } from "./observableVizTitle";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import {
  slComputeBoundsTraining,
  slFormatXTick,
  slFormatYTick,
  slGenerateXTicks,
  slGenerateYTicks,
} from "./scalarLineChartShared";
import type { TrainerNodeData } from "./trainerDefaults";
import { VizSocketsBar } from "./VizSocketsBar";
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

function patchAccVizData(
  id: string,
  patch: Partial<ObservableVizUserNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
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
        lastSweepSummary: cur.lastSweepSummary,
        valueHistory: cur.valueHistory,
        testValueHistory: cur.testValueHistory,
        stepTicks: cur.stepTicks,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
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

function transformLoss(v: number, logY: boolean): number {
  return logY ? Math.log10(Math.max(v, LOG_Y_FLOOR)) : v;
}

function rawLossToPlotY(raw: number, _yPlotMetric: "loss"): number {
  return raw;
}

function buildSeriesPath(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: Bounds,
  yPlotMetric: "loss",
): string {
  if (steps.length < 2 || values.length !== steps.length) return "";
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return steps
    .map((s, i) => {
      const x = transformStep(s, logX);
      const y = transformLoss(rawLossToPlotY(values[i]!, yPlotMetric), logY);
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

type FlowNodeLite = { id: string; type?: string | undefined; data?: unknown };

/**
 * Fallback when ``applyTrainerVizPayload`` did not patch this viz node: series live on the
 * paired trainer’s ``observableMetricHistories`` (keys = viz id and/or observable id; test =
 * ``{observableId}::test``) plus shared ``stepTicks``.
 */
function readTrainerMirrorAccuracy(
  nodes: readonly FlowNodeLite[],
  trainerId: string | undefined,
  vizId: string,
  pairedObsId: string | undefined,
): {
  stepTicks?: number[];
  trainVals?: number[];
  testVals?: number[] | null;
} {
  const tid = (trainerId ?? "").trim();
  if (!tid) return {};
  const tr = nodes.find((n) => n.id === tid && n.type === "trainer");
  if (!tr) return {};
  const td = (tr.data ?? {}) as Partial<TrainerNodeData>;
  const ticks = td.stepTicks;
  const omh = td.observableMetricHistories;
  if (!ticks?.length || !omh) return { stepTicks: ticks };
  const pid = (pairedObsId ?? "").trim();
  let train = omh[vizId];
  if (!Array.isArray(train) || train.length !== ticks.length) {
    train = pid ? omh[pid] : undefined;
  }
  if (!Array.isArray(train) || train.length !== ticks.length) {
    return { stepTicks: ticks };
  }
  const testKey = pid ? `${pid}::test` : "";
  const rawTest = testKey ? omh[testKey] : undefined;
  const testOk =
    Array.isArray(rawTest) && rawTest.length === ticks.length && rawTest.length >= 2 ? rawTest : null;
  return { stepTicks: ticks, trainVals: train, testVals: testOk };
}

/**
 * Accuracy mirror: same panel layout as Training viz (y row, log scales, train/test toggles),
 * wired from trainer stream as ``valueHistory`` / ``testValueHistory`` + ``stepTicks``.
 */
export function ObservableVizAccuracyNode({ id, data, selected }: NodeProps) {
  const def = defaultObservableVizUserData();
  const raw = (data ?? {}) as Partial<ObservableVizUserNodeData>;
  const d: ObservableVizUserNodeData = {
    pairedObservableId: raw.pairedObservableId,
    pairedTrainerId: raw.pairedTrainerId,
    observableName: raw.observableName ?? def.observableName,
    lastSweepSummary: raw.lastSweepSummary,
    valueHistory: raw.valueHistory,
    testValueHistory: raw.testValueHistory,
    stepTicks: raw.stepTicks,
    logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
    logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
    showTrainCurve: raw.showTrainCurve ?? def.showTrainCurve,
    showTestCurve: raw.showTestCurve ?? def.showTestCurve,
    showSeries: raw.showSeries ?? def.showSeries,
    zoomXMin: raw.zoomXMin,
    zoomXMax: raw.zoomXMax,
  };

  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<ObservableVizUserNodeData>) => patchAccVizData(id, patch, setNodes);
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);

  const mirror = useStore(
    useCallback(
      (state) =>
        readTrainerMirrorAccuracy(
          state.nodes as FlowNodeLite[],
          d.pairedTrainerId,
          id,
          d.pairedObservableId,
        ),
      [d.pairedTrainerId, d.pairedObservableId, id],
    ),
  );

  const steps = useMemo(() => {
    const lt = d.stepTicks;
    const lv = d.valueHistory;
    if (lt?.length && lv?.length && lt.length === lv.length && lt.length >= 2) return lt;
    const mt = mirror.stepTicks;
    const mv = mirror.trainVals;
    if (mt?.length && mv?.length && mt.length === mv.length && mt.length >= 2) return mt;
    return [];
  }, [d.stepTicks, d.valueHistory, mirror.stepTicks, mirror.trainVals]);

  const trainVals = useMemo(() => {
    const lt = d.stepTicks;
    const lv = d.valueHistory;
    if (lv?.length && lt?.length && lv.length === lt.length && lv.length >= 2) return lv;
    const mv = mirror.trainVals;
    const mt = mirror.stepTicks;
    if (mv?.length && mt?.length && mv.length === mt.length && mv.length >= 2) return mv;
    return [];
  }, [d.valueHistory, d.stepTicks, mirror.trainVals, mirror.stepTicks]);

  const testVals = useMemo(() => {
    const lt = d.stepTicks;
    const tv = d.testValueHistory;
    if (tv?.length && lt?.length && tv.length === lt.length && tv.length >= 2) return tv;
    const mv = mirror.testVals;
    const mt = mirror.stepTicks;
    if (mv?.length && mt?.length && mv.length === mt.length && mv.length >= 2) return mv;
    return null;
  }, [d.testValueHistory, d.stepTicks, mirror.testVals, mirror.stepTicks]);

  const isLive = steps.length >= 2 && trainVals.length === steps.length && trainVals.length >= 2;

  const yPlotMetricEffective: "loss" = "loss";

  const bounds = useMemo(
    () =>
      slComputeBoundsTraining(
        steps,
        trainVals,
        testVals,
        !!d.showTrainCurve,
        !!d.showTestCurve,
        !!d.logScaleX,
        !!d.logScaleY,
        yPlotMetricEffective,
      ),
    [steps, trainVals, testVals, d.showTrainCurve, d.showTestCurve, d.logScaleX, d.logScaleY],
  );
  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - PAD_R;
  const hasSeries =
    (d.showTrainCurve && trainVals.length >= 2 && trainVals.length === steps.length) ||
    (!!d.showTestCurve && !!testVals && testVals.length >= 2 && testVals.length === steps.length);
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
    if (!d.showTrainCurve) return "";
    return buildSeriesPath(steps, trainVals, !!d.logScaleX, !!d.logScaleY, viewBounds, yPlotMetricEffective);
  }, [d.showTrainCurve, d.logScaleX, d.logScaleY, steps, trainVals, viewBounds]);

  const testPath = useMemo(() => {
    if (!d.showTestCurve || !testVals) return "";
    return buildSeriesPath(steps, testVals, !!d.logScaleX, !!d.logScaleY, viewBounds, yPlotMetricEffective);
  }, [d.showTestCurve, d.logScaleX, d.logScaleY, steps, testVals, viewBounds]);

  const hasTestSeries = testVals !== null && testVals.length >= 2;

  const plotMidX = PAD_L + (CHART_W - PAD_L - PAD_R) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${id}-acc-tviz-clip`;

  const hint = isLive
    ? hasTestSeries
      ? "Train / test top-1 accuracy from last run (adjust scales below)"
      : "Train accuracy from last run (no test set in dataset)"
    : "No data yet — connect Trainer “observable”, then Train.";

  return (
    <div
      className={`cr-node cr-node--training-visualization${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar mode="tensor" annotatorSource compareSource />
        <div className="cr-tviz-chart-divider" aria-hidden />
        <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked nodrag nopan">
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--y-metric">
            <div className="cr-tviz-metric-static nodrag nopan" aria-label="Y-axis metric (accuracy)">
              <span className="cr-tviz-metric-select__lbl">y</span>
              <span className="cr-tviz-metric-static__val">acc</span>
            </div>
          </div>
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--scales">
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
          </div>
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--series">
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
                disabled={isLive && !hasTestSeries}
              />
              test
            </label>
            {isZoomed ? (
              <button type="button" className="cr-tviz-reset-zoom" onClick={resetZoom}>
                reset zoom-in
              </button>
            ) : null}
          </div>
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
            aria-label="Train and test accuracy"
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

            {yTicks.map((yt) => {
              const py = yToPx(yt, viewBounds);
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
              const px = xToPx(xt, viewBounds);
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
              acc
            </text>

            {trainPath || testPath ? (
              <g className="cr-tviz-legend" transform={`translate(${innerRight - 52}, ${PAD_T + 4})`}>
                {trainPath ? (
                  <g>
                    <line x1={0} y1={5} x2={14} y2={5} className="cr-tviz-chart__line" strokeWidth={1.6} />
                    <text x={17} y={7} className="cr-tviz-chart__legend-text">
                      train
                    </text>
                  </g>
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
            ) : null}
          </svg>
          <p className="cr-tviz-hint">{hint}</p>
        </div>
      </div>
    </div>
  );
}
