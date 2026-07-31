import { addEdge, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { appendResearchNode } from "../../graph/nodeInstanceTitle";
import { TRAINER_AUTO_VIZ_DX } from "../../graph/trainerAutoVizSpawn";
import {
  buildTrainPlotYSeries,
  detectPhaseTransitionRegions,
  detectPlateauRegions,
  detectSpikeRegions,
} from "../../graph/trainingCurvePlateauSpike";
import { trainingVisualizationSupportsPerplexityYAxis } from "../../graph/trainingVizPerplexitySupport";
import {
  resolveTrainingVizTrainSeries,
  trainingVisualizationHasWeightRegLoss,
  trainingVizShowsTestCurve,
  trainingVizWeightRegLabels,
  type TrainingVizLossView,
} from "../../graph/trainingVizLossView";
import {
  defaultCurveAnnotatorData,
  type CurveAnnotatorNodeData,
  type CurveAnnotatorRegion,
} from "./curveAnnotatorDefaults";
import {
  defaultTrainingVisualizationData,
  type TrainingVisualizationNodeData,
} from "./trainingVisualizationDefaults";
import {
  slComputeBoundsTraining,
  slFormatXTick,
  slFormatYTick,
  slGenerateXTicks,
  slGenerateYTicks,
} from "./scalarLineChartShared";
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

function newAnnotatorRegionId(): string {
  return `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function findWiredCurveAnnotatorId(edges: Edge[], nodes: Node[], trainingVizId: string): string | null {
  const e = edges.find(
    (ed) =>
      ed.source === trainingVizId &&
      (ed.sourceHandle ?? "") === "annotator" &&
      (ed.targetHandle ?? "") === "from_viz",
  );
  if (!e) return null;
  const target = nodes.find((n) => n.id === e.target);
  return target?.type === "curve_annotator" ? e.target : null;
}

function patchTvizData(
  id: string,
  patch: Partial<TrainingVisualizationNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTrainingVisualizationData();
      const cur = (n.data ?? {}) as Partial<TrainingVisualizationNodeData>;
      const prev: TrainingVisualizationNodeData = {
        lossHistory: cur.lossHistory,
        testLossHistory: cur.testLossHistory,
        regLossHistory: cur.regLossHistory,
        stepTicks: cur.stepTicks,
        lastSweepSummary: cur.lastSweepSummary,
        plotPngBase64: cur.plotPngBase64,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
        showTrainCurve: cur.showTrainCurve ?? def.showTrainCurve,
        showTestCurve: cur.showTestCurve ?? def.showTestCurve,
        yPlotMetric: cur.yPlotMetric ?? def.yPlotMetric,
        lossView: cur.lossView ?? def.lossView,
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

function rawLossToPlotY(raw: number, yPlotMetric: "loss" | "perplexity"): number {
  if (yPlotMetric === "perplexity") {
    return Math.exp(raw);
  }
  return raw;
}

function buildSeriesPath(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: Bounds,
  yPlotMetric: "loss" | "perplexity",
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
  return PAD_L + ((innerW * (x - b.minX)) / (b.maxX - b.minX || 1));
}

function yToPx(y: number, b: Bounds): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

export function TrainingVisualizationNode({ id, data, selected }: NodeProps) {
  const def = defaultTrainingVisualizationData();
  const raw = (data ?? {}) as Partial<TrainingVisualizationNodeData>;
  const d: TrainingVisualizationNodeData = {
    lossHistory: raw.lossHistory,
    testLossHistory: raw.testLossHistory,
    regLossHistory: raw.regLossHistory,
    stepTicks: raw.stepTicks,
    lastSweepSummary: raw.lastSweepSummary,
    plotPngBase64: raw.plotPngBase64,
    logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
    logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
    showTrainCurve: raw.showTrainCurve ?? def.showTrainCurve ?? true,
    showTestCurve: raw.showTestCurve ?? def.showTestCurve ?? true,
    yPlotMetric: raw.yPlotMetric ?? def.yPlotMetric ?? "loss",
    lossView: raw.lossView ?? def.lossView ?? "loss",
    zoomXMin: raw.zoomXMin,
    zoomXMax: raw.zoomXMax,
  };
  const { setNodes, setEdges, getNodes, getEdges } = useReactFlow();
  const update = (patch: Partial<TrainingVisualizationNodeData>) => patchTvizData(id, patch, setNodes);
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);
  const [plateauThresholdStr, setPlateauThresholdStr] = useState("1e-4");
  const [spikeThresholdStr, setSpikeThresholdStr] = useState("2");
  const [phaseThresholdStr, setPhaseThresholdStr] = useState("2");
  const [detectNotice, setDetectNotice] = useState<string | null>(null);
  const detectNoticeClearRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isLive = !!(
    d.lossHistory &&
    d.stepTicks &&
    d.lossHistory.length >= 2 &&
    d.stepTicks.length === d.lossHistory.length
  );

  const hasWeightRegLoss = useStore(
    useCallback(
      (state) => trainingVisualizationHasWeightRegLoss(state.nodes, state.edges, id),
      [id],
    ),
  );
  const weightRegLabels = useStore(
    useCallback(
      (state) => trainingVizWeightRegLabels(state.nodes, state.edges, id),
      [id],
    ),
  );
  const lossView: TrainingVizLossView = hasWeightRegLoss ? (d.lossView ?? "loss") : "loss";
  const testCurveAvailable = trainingVizShowsTestCurve({ ...d, lossView });

  const steps = useMemo(() => {
    if (isLive && d.stepTicks) return d.stepTicks;
    return [];
  }, [isLive, d.stepTicks]);

  const trainVals = useMemo(() => {
    if (!isLive) return [];
    const resolved = resolveTrainingVizTrainSeries({ ...d, lossView });
    return resolved.length >= 2 ? resolved : [];
  }, [isLive, d, lossView]);

  const testVals = useMemo(() => {
    if (!isLive || !testCurveAvailable) return null;
    if (
      d.testLossHistory &&
      d.stepTicks &&
      d.testLossHistory.length === d.stepTicks.length &&
      d.testLossHistory.length >= 2
    ) {
      return d.testLossHistory;
    }
    return null;
  }, [isLive, testCurveAvailable, d.testLossHistory, d.stepTicks]);

  const yPlotMetric = d.yPlotMetric ?? "loss";
  const supportsPerplexityY = useStore(
    useCallback(
      (state) => trainingVisualizationSupportsPerplexityYAxis(state.nodes, state.edges, id),
      [id],
    ),
  );
  const yPlotMetricEffective: "loss" | "perplexity" =
    supportsPerplexityY && lossView === "loss" ? yPlotMetric : "loss";

  const trainPlotY = useMemo(() => {
    if (!isLive || trainVals.length < 2) return [];
    return buildTrainPlotYSeries(trainVals, yPlotMetricEffective);
  }, [isLive, trainVals, yPlotMetricEffective]);

  const trainDetectY = useMemo(() => {
    if (!isLive || trainVals.length < 2) return [];
    // Keep automatic region detection in the same y-space as the chart view.
    return trainVals.map((v) => transformLoss(rawLossToPlotY(v, yPlotMetricEffective), !!d.logScaleY));
  }, [isLive, trainVals, yPlotMetricEffective, d.logScaleY]);

  const canDetectTrain = isLive && !!d.showTrainCurve && trainVals.length >= 3 && steps.length === trainVals.length;

  const showDetectNotice = useCallback((msg: string) => {
    if (detectNoticeClearRef.current) clearTimeout(detectNoticeClearRef.current);
    setDetectNotice(msg);
    detectNoticeClearRef.current = setTimeout(() => {
      setDetectNotice(null);
      detectNoticeClearRef.current = null;
    }, 4500);
  }, []);

  const upsertCurveAnnotatorRegions = useCallback(
    (newRegions: CurveAnnotatorRegion[], replaceLabel: "plateau" | "spike" | "phase transition") => {
      const edges = getEdges();
      const nodes = getNodes();
      const existingId = findWiredCurveAnnotatorId(edges, nodes, id);

      if (existingId) {
        setNodes((nds) =>
          nds.map((n) => {
            if (n.id !== existingId) return n;
            const prev = { ...defaultCurveAnnotatorData(), ...(n.data as Partial<CurveAnnotatorNodeData>) };
            const kept = (prev.regions ?? []).filter((r) => r.label !== replaceLabel);
            return { ...n, data: { ...prev, regions: [...kept, ...newRegions] } };
          }),
        );
        return;
      }

      if (newRegions.length === 0) return;

      const caId = `curve_annotator-${Math.random().toString(36).slice(2, 10)}`;
      setNodes((nds) => {
        const selfN = nds.find((n) => n.id === id);
        const pos = selfN
          ? { x: selfN.position.x + TRAINER_AUTO_VIZ_DX, y: selfN.position.y }
          : { x: 200, y: 200 };
        const data = { ...defaultCurveAnnotatorData(), regions: newRegions } as Record<string, unknown>;
        let next = appendResearchNode(nds, "curve_annotator", pos, data, caId);
        if (selfN?.parentId) {
          next = { ...next, parentId: selfN.parentId, extent: "parent" as const };
        }
        return [...nds, next];
      });
      setEdges((eds) =>
        addEdge(
          {
            source: id,
            target: caId,
            sourceHandle: "annotator",
            targetHandle: "from_viz",
          },
          eds,
        ),
      );
    },
    [getEdges, getNodes, id, setEdges, setNodes],
  );

  const onDetectPlateaus = useCallback(() => {
    if (!canDetectTrain) return;
    const maxAbsSlope = Number.parseFloat(plateauThresholdStr.trim());
    if (!Number.isFinite(maxAbsSlope) || maxAbsSlope < 0) {
      showDetectNotice("Enter a valid plateau delta (max |Δy/Δstep|, >= 0).");
      return;
    }
    const { regions: spans } = detectPlateauRegions(steps, trainDetectY, maxAbsSlope);
    const regions: CurveAnnotatorRegion[] = spans.map((s) => ({
      id: newAnnotatorRegionId(),
      stepMin: s.stepMin,
      stepMax: s.stepMax,
      label: "plateau",
    }));
    upsertCurveAnnotatorRegions(regions, "plateau");
    if (regions.length === 0) {
      showDetectNotice("No plateaus matched this threshold.");
    }
  }, [canDetectTrain, plateauThresholdStr, showDetectNotice, steps, trainDetectY, upsertCurveAnnotatorRegions]);

  const onDetectSpikes = useCallback(() => {
    if (!canDetectTrain) return;
    const minRelativeSlope = Number.parseFloat(spikeThresholdStr.trim());
    if (!Number.isFinite(minRelativeSlope) || minRelativeSlope <= 1) {
      showDetectNotice("Enter a valid spike delta (relative slope ratio: delta > 1).");
      return;
    }
    const { regions: spans } = detectSpikeRegions(steps, trainDetectY, minRelativeSlope);
    const regions: CurveAnnotatorRegion[] = spans.map((s) => ({
      id: newAnnotatorRegionId(),
      stepMin: s.stepMin,
      stepMax: s.stepMax,
      label: "spike",
    }));
    upsertCurveAnnotatorRegions(regions, "spike");
    if (regions.length === 0) {
      showDetectNotice("No spikes matched this threshold.");
    }
  }, [canDetectTrain, showDetectNotice, spikeThresholdStr, steps, trainDetectY, upsertCurveAnnotatorRegions]);

  const onDetectPhaseTransitions = useCallback(() => {
    if (!canDetectTrain) return;
    const minRelativeSlope = Number.parseFloat(phaseThresholdStr.trim());
    if (!Number.isFinite(minRelativeSlope) || minRelativeSlope <= 1) {
      showDetectNotice("Enter a valid phase Δ (relative slope ratio: delta > 1).");
      return;
    }
    const { regions: spans } = detectPhaseTransitionRegions(steps, trainDetectY, minRelativeSlope);
    const regions: CurveAnnotatorRegion[] = spans.map((s) => ({
      id: newAnnotatorRegionId(),
      stepMin: s.stepMin,
      stepMax: s.stepMax,
      label: "phase transition",
    }));
    upsertCurveAnnotatorRegions(regions, "phase transition");
    if (regions.length === 0) {
      showDetectNotice("No phase transitions matched this threshold.");
    }
  }, [canDetectTrain, phaseThresholdStr, showDetectNotice, steps, trainDetectY, upsertCurveAnnotatorRegions]);

  useEffect(() => {
    if (supportsPerplexityY && lossView === "loss") return;
    if (yPlotMetric !== "perplexity") return;
    update({ yPlotMetric: "loss" });
  }, [supportsPerplexityY, lossView, yPlotMetric, update]);

  useEffect(() => {
    if (lossView === "reg" && d.showTestCurve) {
      update({ showTestCurve: false });
    }
  }, [lossView, d.showTestCurve, update]);

  useEffect(() => {
    return () => {
      if (detectNoticeClearRef.current) clearTimeout(detectNoticeClearRef.current);
    };
  }, []);

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
    [
      steps,
      trainVals,
      testVals,
      d.showTrainCurve,
      d.showTestCurve,
      d.logScaleX,
      d.logScaleY,
      yPlotMetricEffective,
    ],
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
  }, [d.showTrainCurve, d.logScaleX, d.logScaleY, steps, trainVals, viewBounds, yPlotMetricEffective]);

  const testPath = useMemo(() => {
    if (!d.showTestCurve || !testVals) return "";
    return buildSeriesPath(steps, testVals, !!d.logScaleX, !!d.logScaleY, viewBounds, yPlotMetricEffective);
  }, [d.showTestCurve, d.logScaleX, d.logScaleY, steps, testVals, viewBounds, yPlotMetricEffective]);

  const hasTestSeries = testVals !== null && testVals.length >= 2;

  const plotMidX = PAD_L + (CHART_W - PAD_L - PAD_R) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${id}-tviz-clip`;

  const hint = isLive
    ? lossView === "reg"
      ? `${weightRegLabels.regLabel} (train only — no test split)`
      : hasTestSeries
        ? lossView === "loss_plus_reg"
          ? `Train = ${weightRegLabels.lossPlusRegLabel}; test = task loss only`
          : "Train / test task loss from last run"
        : "Train loss from last run (no test set in dataset)"
    : "No data yet — connect Trainer “loss”, then Train.";

  return (
    <div
      className={`cr-node cr-node--training-visualization${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <div className="cr-node__header">Training Viz</div>
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar mode="tensor_list" annotatorSource streamSource />
        <div className="cr-tviz-chart-divider" aria-hidden />
        <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked nodrag nopan">
          {hasWeightRegLoss ? (
            <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--y-metric">
              <label className="cr-tviz-metric-select nodrag nopan">
                <span className="cr-tviz-metric-select__lbl">curve</span>
                <select
                  value={lossView}
                  onChange={(e) => {
                    const v = e.target.value;
                    const next: TrainingVizLossView =
                      v === "reg" ? "reg" : v === "loss_plus_reg" ? "loss_plus_reg" : "loss";
                    update({ lossView: next });
                  }}
                  aria-label="Loss curve view"
                >
                  <option value="loss">loss</option>
                  <option value="reg">{weightRegLabels.regLabel}</option>
                  <option value="loss_plus_reg">{weightRegLabels.lossPlusRegLabel}</option>
                </select>
              </label>
            </div>
          ) : null}
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--y-metric">
            {supportsPerplexityY && lossView === "loss" ? (
              <label className="cr-tviz-metric-select nodrag nopan">
                <span className="cr-tviz-metric-select__lbl">y</span>
                <select
                  value={yPlotMetric}
                  onChange={(e) =>
                    update({ yPlotMetric: e.target.value === "perplexity" ? "perplexity" : "loss" })
                  }
                  aria-label="Y-axis metric"
                >
                  <option value="loss">loss</option>
                  <option value="perplexity">perplexity</option>
                </select>
              </label>
            ) : (
              <div className="cr-tviz-metric-static nodrag nopan" aria-label="Y-axis metric (loss)">
                <span className="cr-tviz-metric-select__lbl">y</span>
                <span className="cr-tviz-metric-static__val">loss</span>
              </div>
            )}
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
                disabled={isLive && (!hasTestSeries || lossView === "reg" || !testCurveAvailable)}
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
            aria-label="Training and test loss"
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
                  <text
                    x={px}
                    y={innerBottom + 12}
                    textAnchor="middle"
                    className="cr-tviz-chart__tick-label"
                  >
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

            <text
              x={plotMidX}
              y={CHART_H - 2}
              textAnchor="middle"
              className="cr-tviz-chart__axis-title"
            >
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
              {lossView === "reg"
                ? weightRegLabels.regLabel
                : lossView === "loss_plus_reg"
                  ? weightRegLabels.lossPlusRegLabel
                  : yPlotMetricEffective === "perplexity"
                    ? "perplexity"
                    : "loss"}
            </text>

            {(trainPath || testPath) ? (
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
          {canDetectTrain ? (
            <div
              className="cr-tviz-detect-panel nodrag nopan"
              aria-label="Automatic plateau, spike, and phase transition detection"
            >
              <div className="cr-tviz-detect-row">
                <label className="cr-tviz-detect-field cr-tviz-detect-field--inline">
                  <span
                    className="cr-tviz-detect-field__lbl cr-tviz-detect-field__lbl--help"
                    title="Plateau delta = maximum allowed absolute slope |Δy/Δstep| on adjacent points (using the currently plotted y metric)."
                  >
                    plateau Δ
                  </span>
                  <input
                    className="cr-tviz-detect-input"
                    type="text"
                    inputMode="decimal"
                    value={plateauThresholdStr}
                    onChange={(e) => setPlateauThresholdStr(e.target.value)}
                    aria-label="Plateau threshold: max absolute slope per step"
                  />
                </label>
                <button type="button" className="cr-tviz-detect-btn" onClick={onDetectPlateaus}>
                  detect plateaus
                </button>
              </div>
              <div className="cr-tviz-detect-row">
                <label className="cr-tviz-detect-field cr-tviz-detect-field--inline">
                  <span
                    className="cr-tviz-detect-field__lbl cr-tviz-detect-field__lbl--help"
                    title="Spike delta is a relative slope ratio: |local rise slope| / |baseline slope|, where baseline slope = (initial loss - final loss) / (final step - initial step). Spike is detected when this ratio is above delta and then recovers."
                  >
                    spike Δ
                  </span>
                  <input
                    className="cr-tviz-detect-input"
                    type="text"
                    inputMode="decimal"
                    value={spikeThresholdStr}
                    onChange={(e) => setSpikeThresholdStr(e.target.value)}
                    aria-label="Spike threshold: minimum upward jump before recovery"
                  />
                </label>
                <button type="button" className="cr-tviz-detect-btn" onClick={onDetectSpikes}>
                  detect spikes
                </button>
              </div>
              <div className="cr-tviz-detect-row">
                <label className="cr-tviz-detect-field cr-tviz-detect-field--inline">
                  <span
                    className="cr-tviz-detect-field__lbl cr-tviz-detect-field__lbl--help"
                    title="Phase transition Δ is a relative slope ratio: (sharp downward |Δy/Δstep|) / |baseline slope|, where baseline slope = |initial − final| / (final step − initial step). A transition ends when loss rebounds from the running minimum by 15% of the initial drop or after 14 steps."
                  >
                    phase Δ
                  </span>
                  <input
                    className="cr-tviz-detect-input"
                    type="text"
                    inputMode="decimal"
                    value={phaseThresholdStr}
                    onChange={(e) => setPhaseThresholdStr(e.target.value)}
                    aria-label="Phase transition threshold: minimum relative downward slope"
                  />
                </label>
                <button type="button" className="cr-tviz-detect-btn" onClick={onDetectPhaseTransitions}>
                  detect phase transitions
                </button>
              </div>
            </div>
          ) : null}
          {detectNotice ? (
            <p className="cr-tviz-hint cr-tviz-hint--detect" role="status" aria-live="polite">
              {detectNotice}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
