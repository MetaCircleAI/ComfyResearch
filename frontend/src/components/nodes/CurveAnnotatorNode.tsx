import { Handle, Position, useReactFlow, useStore, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CURVE_ANNOTATOR_LABEL_GROUPS,
  CURVE_ANNOTATOR_LABEL_OPTION_GROUPS,
  defaultCurveAnnotatorData,
  type CurveAnnotatorLabel,
  type CurveAnnotatorNodeData,
  type CurveAnnotatorRegion,
} from "./curveAnnotatorDefaults";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { defaultTrainingVisualizationData, type TrainingVisualizationNodeData } from "./trainingVisualizationDefaults";
import type { ObservableScalarVizData } from "../../graph/resolveCurveAnnotatorSource";
import { resolveCurveAnnotatorSource } from "../../graph/resolveCurveAnnotatorSource";
import {
  SL_CHART_H,
  SL_CHART_W,
  SL_PAD_B,
  SL_PAD_L,
  SL_PAD_T,
  SL_TICK_LEN,
  SL_Y_LABEL_X,
  slBuildSeriesPathScalar,
  slBuildSeriesPathTraining,
  slComputeBoundsScalar,
  slComputeBoundsTraining,
  slFormatXTick,
  slFormatYTick,
  slGenerateXTicks,
  slGenerateYTicks,
  SL_PAD_R,
  slInnerBottom,
  slInnerRight,
  slTransformStep,
  slViewBoundsFromZoom,
  slXToPx,
  slYToPx,
} from "./scalarLineChartShared";
import { slPlotInnerRect as plotRect, useXSpanDrag } from "./useXSpanDrag";
import { defaultObservableVizUserData } from "./observableVizUserDefaults";
import { trainingVisualizationSupportsPerplexityYAxis } from "../../graph/trainingVizPerplexitySupport";

const TICK_LEN = SL_TICK_LEN;

const TRAINING_Y_METRIC_OPTIONS = [
  { id: "loss" as const, label: "loss" },
  { id: "perplexity" as const, label: "perplexity" },
];

function patchAnnotatorData(
  id: string,
  patch: Partial<CurveAnnotatorNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const prev = { ...defaultCurveAnnotatorData(), ...(n.data as Partial<CurveAnnotatorNodeData>) };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function patchTrainingSource(
  sourceId: string,
  patch: Partial<TrainingVisualizationNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  const def = defaultTrainingVisualizationData();
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== sourceId || n.type !== "training_visualization") return n;
      const cur = (n.data ?? {}) as Partial<TrainingVisualizationNodeData>;
      const prev: TrainingVisualizationNodeData = {
        lossHistory: cur.lossHistory,
        testLossHistory: cur.testLossHistory,
        stepTicks: cur.stepTicks,
        lastSweepSummary: cur.lastSweepSummary,
        plotPngBase64: cur.plotPngBase64,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
        showTrainCurve: cur.showTrainCurve ?? def.showTrainCurve,
        showTestCurve: cur.showTestCurve ?? def.showTestCurve,
        yPlotMetric: cur.yPlotMetric ?? def.yPlotMetric,
        zoomXMin: cur.zoomXMin,
        zoomXMax: cur.zoomXMax,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function patchObservableScalarSource(
  sourceId: string,
  patch: Partial<ObservableScalarVizData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== sourceId || n.type !== "observable_viz") return n;
      const cur = (n.data ?? {}) as Record<string, unknown>;
      return { ...n, data: { ...cur, ...patch } };
    }),
  );
}

function newRegionId(): string {
  return `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function CurveAnnotatorNode({ id, data, selected }: NodeProps) {
  const { setNodes } = useReactFlow();
  const d = { ...defaultCurveAnnotatorData(), ...(data as Partial<CurveAnnotatorNodeData>) };
  const regions = d.regions ?? [];

  const resolved = useStore(
    useCallback(
      (state) => resolveCurveAnnotatorSource(state.nodes, state.edges, id),
      [id],
    ),
  );

  const trainingPerplexityOk = useStore(
    useCallback(
      (state) =>
        resolved.kind === "training"
          ? trainingVisualizationSupportsPerplexityYAxis(state.nodes, state.edges, resolved.sourceId)
          : false,
      [resolved],
    ),
  );

  useEffect(() => {
    if (resolved.kind !== "training") return;
    if (trainingPerplexityOk) return;
    const y = (resolved.data.yPlotMetric ?? "loss") as "loss" | "perplexity";
    if (y !== "perplexity") return;
    patchTrainingSource(resolved.sourceId, { yPlotMetric: "loss" }, setNodes);
  }, [resolved, trainingPerplexityOk, setNodes]);

  const updateAnnotator = useCallback(
    (patch: Partial<CurveAnnotatorNodeData>) => patchAnnotatorData(id, patch, setNodes),
    [id, setNodes],
  );

  const [pendingSpan, setPendingSpan] = useState<{ stepMin: number; stepMax: number } | null>(null);
  const [pendingLabel, setPendingLabel] = useState<CurveAnnotatorLabel>(CURVE_ANNOTATOR_LABEL_GROUPS.regional[0]);

  const onSpanCommit = useCallback(
    (span: { stepMin: number; stepMax: number } | null) => {
      if (span) {
        setPendingSpan(span);
        setPendingLabel(CURVE_ANNOTATOR_LABEL_GROUPS.regional[0]);
      } else {
        setPendingSpan(null);
      }
    },
    [],
  );

  const trainingModel = useMemo(() => {
    if (resolved.kind !== "training") return null;
    const raw = resolved.data;
    const def = defaultTrainingVisualizationData();
    return {
      sourceId: resolved.sourceId,
      lossHistory: raw.lossHistory,
      testLossHistory: raw.testLossHistory,
      stepTicks: raw.stepTicks,
      lastSweepSummary: raw.lastSweepSummary,
      logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
      logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
      showTrainCurve: raw.showTrainCurve ?? def.showTrainCurve ?? true,
      showTestCurve: raw.showTestCurve ?? def.showTestCurve ?? true,
      yPlotMetric: raw.yPlotMetric ?? def.yPlotMetric ?? "loss",
      zoomXMin: raw.zoomXMin,
      zoomXMax: raw.zoomXMax,
    };
  }, [resolved]);

  const scalarModel = useMemo(() => {
    if (resolved.kind !== "observable_scalar") return null;
    const raw = resolved.data;
    const def = defaultObservableVizUserData();
    return {
      sourceId: resolved.sourceId,
      stepTicks: raw.stepTicks,
      valueHistory: raw.valueHistory,
      logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
      logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
      showSeries: raw.showSeries ?? def.showSeries ?? true,
      zoomXMin: raw.zoomXMin,
      zoomXMax: raw.zoomXMax,
      lastSweepSummary: raw.lastSweepSummary,
    };
  }, [resolved]);

  const innerBottom = slInnerBottom();
  const innerRight = slInnerRight();

  const chart = useMemo(() => {
    if (trainingModel) {
      const steps = trainingModel.stepTicks ?? [];
      const trainVals = trainingModel.lossHistory ?? [];
      const testVals =
        trainingModel.testLossHistory &&
        trainingModel.stepTicks &&
        trainingModel.testLossHistory.length === trainingModel.stepTicks.length
          ? trainingModel.testLossHistory
          : null;
      const isLive =
        !!(
          trainingModel.lossHistory &&
          trainingModel.stepTicks &&
          trainingModel.lossHistory.length >= 2 &&
          trainingModel.stepTicks.length === trainingModel.lossHistory.length
        );
      const yPlotMetric = trainingModel.yPlotMetric;
      const yPlotMetricEffective: "loss" | "perplexity" = trainingPerplexityOk ? yPlotMetric : "loss";
      const fullBounds = slComputeBoundsTraining(
        steps,
        trainVals,
        testVals,
        !!trainingModel.showTrainCurve,
        !!trainingModel.showTestCurve,
        !!trainingModel.logScaleX,
        !!trainingModel.logScaleY,
        yPlotMetricEffective,
      );
      const viewBounds = slViewBoundsFromZoom(fullBounds, trainingModel.zoomXMin, trainingModel.zoomXMax);
      const trainPath =
        trainingModel.showTrainCurve && trainVals.length >= 2
          ? slBuildSeriesPathTraining(
              steps,
              trainVals,
              !!trainingModel.logScaleX,
              !!trainingModel.logScaleY,
              viewBounds,
              yPlotMetricEffective,
            )
          : "";
      const testPath =
        trainingModel.showTestCurve && testVals && testVals.length >= 2
          ? slBuildSeriesPathTraining(
              steps,
              testVals,
              !!trainingModel.logScaleX,
              !!trainingModel.logScaleY,
              viewBounds,
              yPlotMetricEffective,
            )
          : "";
      const hasSeries =
        (trainingModel.showTrainCurve && trainVals.length >= 2 && trainVals.length === steps.length) ||
        (!!trainingModel.showTestCurve &&
          !!testVals &&
          testVals.length >= 2 &&
          testVals.length === steps.length);
      let stepMin = Infinity;
      let stepMax = -Infinity;
      for (const s of steps) {
        stepMin = Math.min(stepMin, s);
        stepMax = Math.max(stepMax, s);
      }
      if (!Number.isFinite(stepMin)) stepMin = 0;
      if (!Number.isFinite(stepMax)) stepMax = 1;
      return {
        kind: "training" as const,
        sourceId: trainingModel.sourceId,
        isLive,
        steps,
        trainPath,
        testPath,
        hasTest: testVals !== null && testVals.length >= 2,
        viewBounds,
        fullBounds,
        logScaleX: !!trainingModel.logScaleX,
        logScaleY: !!trainingModel.logScaleY,
        showTrainCurve: !!trainingModel.showTrainCurve,
        showTestCurve: !!trainingModel.showTestCurve,
        yPlotMetric,
        supportsPerplexityY: trainingPerplexityOk,
        lastSweepSummary: trainingModel.lastSweepSummary,
        hasSeries,
        stepDomain: { min: stepMin, max: stepMax },
        yLabel: yPlotMetricEffective === "perplexity" ? "perplexity" : "loss",
      };
    }
    if (scalarModel) {
      const steps = scalarModel.stepTicks ?? [];
      const vals = scalarModel.valueHistory ?? [];
      const isLive =
        !!(scalarModel.valueHistory && scalarModel.stepTicks && scalarModel.valueHistory.length >= 2);
      const fullBounds = slComputeBoundsScalar(
        steps,
        vals,
        !!scalarModel.showSeries,
        !!scalarModel.logScaleX,
        !!scalarModel.logScaleY,
      );
      const viewBounds = slViewBoundsFromZoom(fullBounds, scalarModel.zoomXMin, scalarModel.zoomXMax);
      const seriesPath =
        scalarModel.showSeries && vals.length >= 2 && vals.length === steps.length
          ? slBuildSeriesPathScalar(
              steps,
              vals,
              !!scalarModel.logScaleX,
              !!scalarModel.logScaleY,
              viewBounds,
            )
          : "";
      const hasSeries = !!scalarModel.showSeries && vals.length >= 2 && vals.length === steps.length;
      let stepMin = Infinity;
      let stepMax = -Infinity;
      for (const s of steps) {
        stepMin = Math.min(stepMin, s);
        stepMax = Math.max(stepMax, s);
      }
      if (!Number.isFinite(stepMin)) stepMin = 0;
      if (!Number.isFinite(stepMax)) stepMax = 1;
      return {
        kind: "observable_scalar" as const,
        sourceId: scalarModel.sourceId,
        isLive,
        steps,
        seriesPath,
        viewBounds,
        fullBounds,
        logScaleX: !!scalarModel.logScaleX,
        logScaleY: !!scalarModel.logScaleY,
        showSeries: !!scalarModel.showSeries,
        lastSweepSummary: scalarModel.lastSweepSummary,
        hasSeries,
        stepDomain: { min: stepMin, max: stepMax },
        yLabel: "value",
      };
    }
    return null;
  }, [trainingModel, scalarModel, trainingPerplexityOk]);

  const plotInner = plotRect();
  const xSpan = useXSpanDrag({
    viewBounds: chart?.viewBounds ?? { minX: 0, maxX: 1, minY: 0, maxY: 1 },
    logX: chart?.logScaleX ?? false,
    plainLogX: false,
    stepDomain: chart?.stepDomain ?? { min: 0, max: 1 },
    enabled: !!chart?.hasSeries && !pendingSpan,
    ...plotInner,
    onCommit: onSpanCommit,
  });

  const regionRects = useMemo(() => {
    if (!chart) return [];
    const vb = chart.viewBounds;
    const logX = chart.logScaleX;
    return regions.map((r) => {
      const tx0 = slTransformStep(r.stepMin, logX);
      const tx1 = slTransformStep(r.stepMax, logX);
      const xa = Math.min(tx0, tx1);
      const xb = Math.max(tx0, tx1);
      const x1 = slXToPx(xa, vb);
      const x2 = slXToPx(xb, vb);
      return { id: r.id, x: x1, w: Math.max(1, x2 - x1) };
    });
  }, [chart, regions]);

  const clipId = `${id}-ca-clip`;

  const hintDisconnected = "Connect annotate out from Training viz or Observable viz (scalar).";
  const hintNoData = "No series yet — run training or check the source viz.";

  return (
    <div
      className={`cr-node cr-node--curve-annotator${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">Curve annotator</div>
      <div className="cr-node__body cr-node__body--curve-annotator">
        <div className="cr-ca-socket-row">
          <div className="cr-ca-socket-row__left">
            <Handle
              type="target"
              position={Position.Left}
              id="from_viz"
              className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
            />
            <span className="cr-tviz-socket-label">from viz</span>
          </div>
        </div>
        <div className="cr-tviz-chart-divider" aria-hidden />

        {resolved.kind === "disconnected" ? (
          <p className="cr-ca-hint">{hintDisconnected}</p>
        ) : resolved.kind === "unsupported" ? (
          <p className="cr-ca-hint cr-ca-hint--warn">{resolved.reason}</p>
        ) : !chart ? (
          <p className="cr-ca-hint">{hintNoData}</p>
        ) : (
          <>
            <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked nodrag nopan cr-ca-controls">
              {chart.kind === "training" ? (
                <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--y-metric">
                  {chart.supportsPerplexityY ? (
                    <div className="cr-tviz-metric-select cr-tviz-metric-select--discrete nodrag nopan">
                      <DiscreteMultiSelect
                        label="y"
                        options={TRAINING_Y_METRIC_OPTIONS}
                        value={chart.yPlotMetric}
                        singleSelect
                        onCommit={(next) => {
                          const id = Array.isArray(next) ? next[0] : next;
                          patchTrainingSource(
                            chart.sourceId,
                            { yPlotMetric: id === "perplexity" ? "perplexity" : "loss" },
                            setNodes,
                          );
                        }}
                        ariaLabel="Y-axis metric"
                      />
                    </div>
                  ) : (
                    <div className="cr-tviz-metric-static nodrag nopan" aria-label="Y-axis metric (loss)">
                      <span className="cr-tviz-metric-select__lbl">y</span>
                      <span className="cr-tviz-metric-static__val">loss</span>
                    </div>
                  )}
                </div>
              ) : null}
              <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--scales">
                <label className="cr-tviz-check cr-tviz-check--log-x">
                  <input
                    type="checkbox"
                    checked={chart.logScaleX}
                    onChange={(e) =>
                      chart.kind === "training"
                        ? patchTrainingSource(chart.sourceId, { logScaleX: e.target.checked }, setNodes)
                        : patchObservableScalarSource(chart.sourceId, { logScaleX: e.target.checked }, setNodes)
                    }
                  />
                  log x
                </label>
                <label className="cr-tviz-check cr-tviz-check--log-y">
                  <input
                    type="checkbox"
                    checked={chart.logScaleY}
                    onChange={(e) =>
                      chart.kind === "training"
                        ? patchTrainingSource(chart.sourceId, { logScaleY: e.target.checked }, setNodes)
                        : patchObservableScalarSource(chart.sourceId, { logScaleY: e.target.checked }, setNodes)
                    }
                  />
                  log y
                </label>
              </div>
              <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--series">
                {chart.kind === "training" ? (
                  <>
                    <label className="cr-tviz-check cr-tviz-check--train">
                      <input
                        type="checkbox"
                        checked={chart.showTrainCurve}
                        onChange={(e) =>
                          patchTrainingSource(chart.sourceId, { showTrainCurve: e.target.checked }, setNodes)
                        }
                      />
                      train
                    </label>
                    <label className="cr-tviz-check cr-tviz-check--test">
                      <input
                        type="checkbox"
                        checked={chart.showTestCurve}
                        onChange={(e) =>
                          patchTrainingSource(chart.sourceId, { showTestCurve: e.target.checked }, setNodes)
                        }
                        disabled={chart.isLive && !chart.hasTest}
                      />
                      test
                    </label>
                  </>
                ) : (
                  <label className="cr-tviz-check cr-tviz-check--train">
                    <input
                      type="checkbox"
                      checked={chart.showSeries}
                      onChange={(e) =>
                        patchObservableScalarSource(chart.sourceId, { showSeries: e.target.checked }, setNodes)
                      }
                    />
                    series
                  </label>
                )}
              </div>
            </div>
            {chart.lastSweepSummary?.trim() ? (
              <div className="cr-tviz-sweep-line nodrag nopan" title={chart.lastSweepSummary}>
                {chart.lastSweepSummary}
              </div>
            ) : null}

            <div className="cr-tviz-chart-wrap">
              <svg
                className="cr-tviz-chart cr-ca-chart nodrag nopan"
                viewBox={`0 0 ${SL_CHART_W} ${SL_CHART_H}`}
                width={SL_CHART_W}
                height={SL_CHART_H}
                aria-label="Curve annotation chart"
                onMouseDown={xSpan.onMouseDown}
                onMouseMove={xSpan.onMouseMove}
                onMouseUp={xSpan.onMouseUp}
                onMouseLeave={xSpan.onMouseLeave}
                style={{
                  cursor: chart.hasSeries ? (xSpan.isDragging ? "crosshair" : "cell") : "default",
                }}
              >
                <defs>
                  <clipPath id={clipId}>
                    <rect
                      x={SL_PAD_L}
                      y={SL_PAD_T}
                      width={SL_CHART_W - SL_PAD_L - SL_PAD_R}
                      height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
                    />
                  </clipPath>
                </defs>
                <rect
                  x={SL_PAD_L}
                  y={SL_PAD_T}
                  width={SL_CHART_W - SL_PAD_L - SL_PAD_R}
                  height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
                  rx={4}
                  className="cr-tviz-chart__plot-bg"
                />

                {slGenerateYTicks(chart.viewBounds.minY, chart.viewBounds.maxY, chart.logScaleY).map((yt) => (
                  <line
                    key={`gy-${yt}`}
                    x1={SL_PAD_L}
                    y1={slYToPx(yt, chart.viewBounds)}
                    x2={innerRight}
                    y2={slYToPx(yt, chart.viewBounds)}
                    className="cr-tviz-chart__grid"
                  />
                ))}
                {slGenerateXTicks(chart.viewBounds.minX, chart.viewBounds.maxX, chart.logScaleX).map((xt) => (
                  <line
                    key={`gx-${xt}`}
                    x1={slXToPx(xt, chart.viewBounds)}
                    y1={SL_PAD_T}
                    x2={slXToPx(xt, chart.viewBounds)}
                    y2={innerBottom}
                    className="cr-tviz-chart__grid"
                  />
                ))}

                <line
                  x1={SL_PAD_L}
                  y1={innerBottom}
                  x2={innerRight}
                  y2={innerBottom}
                  className="cr-tviz-chart__axis-line"
                />
                <line
                  x1={SL_PAD_L}
                  y1={SL_PAD_T}
                  x2={SL_PAD_L}
                  y2={innerBottom}
                  className="cr-tviz-chart__axis-line"
                />

                {slGenerateXTicks(chart.viewBounds.minX, chart.viewBounds.maxX, chart.logScaleX).map((xt) => {
                  const px = slXToPx(xt, chart.viewBounds);
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
                        {slFormatXTick(xt, chart.logScaleX)}
                      </text>
                    </g>
                  );
                })}

                {slGenerateYTicks(chart.viewBounds.minY, chart.viewBounds.maxY, chart.logScaleY).map((yt) => {
                  const py = slYToPx(yt, chart.viewBounds);
                  return (
                    <g key={`yt-${yt}`}>
                      <line
                        x1={SL_PAD_L - TICK_LEN}
                        y1={py}
                        x2={SL_PAD_L}
                        y2={py}
                        className="cr-tviz-chart__tick"
                      />
                      <text
                        x={SL_PAD_L - 6}
                        y={py}
                        dominantBaseline="middle"
                        textAnchor="end"
                        className="cr-tviz-chart__tick-label"
                      >
                        {slFormatYTick(yt, chart.logScaleY)}
                      </text>
                    </g>
                  );
                })}

                {chart.kind === "training" ? (
                  <>
                    {chart.testPath ? (
                      <path
                        d={chart.testPath}
                        fill="none"
                        className="cr-tviz-chart__line cr-tviz-chart__line--test"
                        strokeWidth={1.5}
                        clipPath={`url(#${clipId})`}
                      />
                    ) : null}
                    {chart.trainPath ? (
                      <path
                        d={chart.trainPath}
                        fill="none"
                        className="cr-tviz-chart__line"
                        strokeWidth={1.6}
                        clipPath={`url(#${clipId})`}
                      />
                    ) : null}
                  </>
                ) : chart.seriesPath ? (
                  <path
                    d={chart.seriesPath}
                    fill="none"
                    className="cr-tviz-chart__line"
                    strokeWidth={1.6}
                    clipPath={`url(#${clipId})`}
                  />
                ) : null}

                {regionRects.map((rr) => (
                  <rect
                    key={rr.id}
                    x={rr.x}
                    y={SL_PAD_T}
                    width={rr.w}
                    height={innerBottom - SL_PAD_T}
                    className="cr-ca-region-fill"
                    clipPath={`url(#${clipId})`}
                  />
                ))}

                {xSpan.selectionRect ? (
                  <rect
                    x={xSpan.selectionRect.x}
                    y={xSpan.selectionRect.y}
                    width={xSpan.selectionRect.w}
                    height={xSpan.selectionRect.h}
                    className="cr-ca-span-box"
                  />
                ) : null}

                <text
                  x={SL_PAD_L + (SL_CHART_W - SL_PAD_L - SL_PAD_R) / 2}
                  y={SL_CHART_H - 2}
                  textAnchor="middle"
                  className="cr-tviz-chart__axis-title"
                >
                  step
                </text>
                <text
                  x={SL_Y_LABEL_X}
                  y={SL_PAD_T + (SL_CHART_H - SL_PAD_T - SL_PAD_B) / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  transform={`rotate(-90, ${SL_Y_LABEL_X}, ${SL_PAD_T + (SL_CHART_H - SL_PAD_T - SL_PAD_B) / 2})`}
                  className="cr-tviz-chart__axis-title"
                >
                  {chart.yLabel}
                </text>
              </svg>
              <p className="cr-tviz-hint">
                {chart.hasSeries
                  ? chart.isLive
                    ? "Drag horizontally on the plot to select an x-range, then choose a label."
                    : hintNoData
                  : hintNoData}
              </p>
            </div>

            {pendingSpan ? (
              <div className="cr-ca-pending nodrag nopan">
                <span className="cr-ca-pending__range">
                  steps {pendingSpan.stepMin.toFixed(0)}–{pendingSpan.stepMax.toFixed(0)}
                </span>
                <div className="cr-ca-pending__label-picker nodrag nopan">
                  <DiscreteMultiSelect
                    label="Label"
                    optionGroups={CURVE_ANNOTATOR_LABEL_OPTION_GROUPS}
                    value={pendingLabel}
                    singleSelect
                    onCommit={(next) => {
                      const id = (Array.isArray(next) ? next[0] : next) as CurveAnnotatorLabel | undefined;
                      if (id) setPendingLabel(id);
                    }}
                    ariaLabel="Region label"
                  />
                </div>
                <div className="cr-ca-pending__actions">
                  <button
                    type="button"
                    className="cr-ca-btn"
                    onClick={() => {
                      if (!pendingSpan) return;
                      const next: CurveAnnotatorRegion = {
                        id: newRegionId(),
                        stepMin: pendingSpan.stepMin,
                        stepMax: pendingSpan.stepMax,
                        label: pendingLabel,
                      };
                      updateAnnotator({ regions: [...regions, next] });
                      setPendingSpan(null);
                    }}
                  >
                    Add region
                  </button>
                  <button type="button" className="cr-ca-btn cr-ca-btn--ghost" onClick={() => setPendingSpan(null)}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : null}

            {regions.length > 0 ? (
              <ul className="cr-ca-region-list nodrag nopan" aria-label="Annotated regions">
                {regions.map((r) => (
                  <li key={r.id} className="cr-ca-region-list__item">
                    <span className="cr-ca-region-list__text">
                      {r.stepMin.toFixed(0)}–{r.stepMax.toFixed(0)}: {r.label}
                    </span>
                    <button
                      type="button"
                      className="cr-ca-region-list__rm"
                      aria-label="Remove region"
                      onClick={() => updateAnnotator({ regions: regions.filter((x) => x.id !== r.id) })}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
