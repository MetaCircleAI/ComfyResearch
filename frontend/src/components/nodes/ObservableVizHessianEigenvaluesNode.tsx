import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import {
  defaultObservableVizHessianEigenvaluesData,
  type ObservableVizHessianEigenvaluesNodeData,
} from "./observableVizHessianEigenvaluesDefaults";
import { useObservableVizHeaderTitle } from "./observableVizTitle";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { VizSocketsBar } from "./VizSocketsBar";
import { slFormatXTick, slFormatYTick, slGenerateXTicks, slGenerateYTicks, slPadBounds } from "./scalarLineChartShared";
import { useLineChartZoom } from "./useLineChartZoom";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  isPerTensorLayerSeriesLabels,
  sortedTensorFamiliesFromLabels,
  tensorCurveShortLabel,
  tensorFamilyFromParameterLabel,
} from "../../graph/observableTensorFamily";

const CHART_W = 232;
const CHART_H = 132;
const PAD_L = 36;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 36;
const TICK_LEN = 4;
const Y_LABEL_X = 9;
const LOG_Y_FLOOR = 1e-15;

/* Theme-aware via tokens.css; classic values pin the exact legacy hex per position. */
const SERIES_COLORS = ["var(--cr-chart-1)", "var(--cr-chart-2)", "var(--cr-chart-3)", "var(--cr-chart-4)", "var(--cr-chart-9)", "var(--cr-chart-6)", "var(--cr-chart-7)", "var(--cr-chart-5)"];

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function patchData(
  id: string,
  patch: Partial<ObservableVizHessianEigenvaluesNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultObservableVizHessianEigenvaluesData();
      const curAll = (n.data ?? {}) as Record<string, unknown>;
      const cur = curAll as Partial<ObservableVizHessianEigenvaluesNodeData>;
      const prev: ObservableVizHessianEigenvaluesNodeData = {
        pairedObservableId: cur.pairedObservableId,
        pairedTrainerId: cur.pairedTrainerId,
        lastSweepSummary: cur.lastSweepSummary,
        vizVariant: cur.vizVariant,
        seriesLabels: cur.seriesLabels,
        topK: cur.topK ?? def.topK,
        order: cur.order ?? def.order,
        valueHistories: cur.valueHistories,
        stepTicks: cur.stepTicks,
        logScaleX: cur.logScaleX ?? def.logScaleX,
        logScaleY: cur.logScaleY ?? def.logScaleY,
        eigenSeriesVisible: cur.eigenSeriesVisible,
        showSeries: cur.showSeries,
        zoomXMin: cur.zoomXMin,
        zoomXMax: cur.zoomXMax,
        l2TensorFamily: cur.l2TensorFamily,
      };
      // Preserve `vizVariant`, `instanceTitle`, etc. on `observable_viz` nodes (do not replace entire `data`).
      return { ...n, data: { ...curAll, ...prev, ...patch } };
    }),
  );
}

function transformStep(step: number, logX: boolean): number {
  return logX ? Math.log10(Math.max(0, step) + 1) : step;
}

function transformY(v: number, logY: boolean): number {
  return logY ? Math.log10(Math.max(v, LOG_Y_FLOOR)) : v;
}

function normalizeScalarRow(row: unknown[]): number[] {
  return row.map((cell) => {
    if (cell === null || cell === undefined) return Number.NaN;
    const n = Number(cell);
    return Number.isFinite(n) ? n : Number.NaN;
  });
}

function computeBoundsMulti(
  steps: number[],
  seriesList: number[][],
  eigenVisible: boolean[],
  logX: boolean,
  logY: boolean,
): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  if (steps.length >= 1) {
    for (let si = 0; si < seriesList.length; si++) {
      if (eigenVisible[si] === false) continue;
      const values = seriesList[si]!;
      if (values.length !== steps.length) continue;
      for (let i = 0; i < steps.length; i++) {
        const rawY = values[i]!;
        if (!Number.isFinite(rawY)) continue;
        const x = transformStep(steps[i]!, logX);
        const y = transformY(rawY, logY);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
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

/** Skip non-finite points (JSON null / failed Hessian); break stroke with a new ``M`` after gaps. */
function buildPathFinite(
  steps: number[],
  values: number[],
  logX: boolean,
  logY: boolean,
  b: Bounds,
): string {
  if (values.length !== steps.length || steps.length < 1) return "";
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  if (steps.length === 1) {
    const rawY = values[0]!;
    if (!Number.isFinite(rawY)) return "";
    const x = transformStep(steps[0]!, logX);
    const y = transformY(rawY, logY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return "";
    const px = PAD_L + (innerW * (x - b.minX)) / spanX;
    const py = PAD_T + innerH * (1 - (y - b.minY) / spanY);
    const x2 = Math.min(b.maxX, x + Math.max((b.maxX - b.minX) * 0.02, 1e-6));
    const px2 = PAD_L + (innerW * (x2 - b.minX)) / spanX;
    return `M${px.toFixed(2)},${py.toFixed(2)}L${px2.toFixed(2)},${py.toFixed(2)}`;
  }
  const parts: string[] = [];
  let penUp = true;
  for (let i = 0; i < steps.length; i++) {
    const rawY = values[i]!;
    if (!Number.isFinite(rawY)) {
      penUp = true;
      continue;
    }
    const x = transformStep(steps[i]!, logX);
    const y = transformY(rawY, logY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      penUp = true;
      continue;
    }
    const px = PAD_L + (innerW * (x - b.minX)) / spanX;
    const py = PAD_T + innerH * (1 - (y - b.minY) / spanY);
    parts.push(`${penUp ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`);
    penUp = false;
  }
  return parts.join(" ");
}

function xToPx(x: number, b: Bounds): number {
  const innerW = CHART_W - PAD_L - PAD_R;
  return PAD_L + (innerW * (x - b.minX)) / (b.maxX - b.minX || 1);
}

function yToPx(y: number, b: Bounds): number {
  const innerH = CHART_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

export function ObservableVizHessianEigenvaluesNode({ id, data, selected }: NodeProps) {
  const def = defaultObservableVizHessianEigenvaluesData();
  const raw = (data ?? {}) as Partial<ObservableVizHessianEigenvaluesNodeData>;
  const vizVariant = typeof raw.vizVariant === "string" ? raw.vizVariant.trim() : "";
  const seriesLabels = Array.isArray(raw.seriesLabels)
    ? raw.seriesLabels.filter((x): x is string => typeof x === "string")
    : undefined;
  const d: ObservableVizHessianEigenvaluesNodeData = {
    pairedObservableId: raw.pairedObservableId,
    pairedTrainerId: raw.pairedTrainerId,
    lastSweepSummary: raw.lastSweepSummary,
    vizVariant: raw.vizVariant,
    seriesLabels: raw.seriesLabels,
    topK: raw.topK ?? def.topK,
    order: raw.order === "ascending" ? "ascending" : "descending",
    valueHistories: raw.valueHistories,
    stepTicks: raw.stepTicks,
    logScaleX: raw.logScaleX ?? def.logScaleX ?? false,
    logScaleY: raw.logScaleY ?? def.logScaleY ?? false,
    eigenSeriesVisible: raw.eigenSeriesVisible,
    showSeries: raw.showSeries,
    zoomXMin: raw.zoomXMin,
    zoomXMax: raw.zoomXMax,
    l2TensorFamily: typeof raw.l2TensorFamily === "string" ? raw.l2TensorFamily : undefined,
  };
  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<ObservableVizHessianEigenvaluesNodeData>) => patchData(id, patch, setNodes);
  const [isDraggingZoom, setIsDraggingZoom] = useState(false);

  const rawScalarHistory = (raw as { valueHistory?: unknown[] }).valueHistory;
  const seriesList = useMemo(() => {
    const vh = d.valueHistories;
    if (Array.isArray(vh) && vh.length > 0) {
      const rows = vh
        .filter((row): row is unknown[] => Array.isArray(row) && row.length >= 1)
        .map((row) => normalizeScalarRow(row));
      if (rows.length > 0) return rows;
    }
    if (Array.isArray(rawScalarHistory) && rawScalarHistory.length >= 1) {
      return [normalizeScalarRow(rawScalarHistory)];
    }
    return [];
  }, [d.valueHistories, rawScalarHistory]);

  /** Hessian UI reserves ``topK`` λ slots before/without data; other line-metric variants use one toggle per row. */
  const isHessianEigenViz = vizVariant === "hessian_eigenvalues";
  const rankCount = useMemo(() => {
    if (isHessianEigenViz) {
      return Math.max(1, d.topK ?? def.topK ?? 5, seriesList.length);
    }
    return Math.max(0, seriesList.length);
  }, [isHessianEigenViz, d.topK, def.topK, seriesList.length]);

  const seriesLabelsResolved = seriesLabels ?? [];
  const tensorFamilyOptions = useMemo(() => {
    // Tensor-family picker is tuned for gradient / weight L2 legends.
    if (vizVariant !== "gradient_norm" && vizVariant !== "weight_l2") return [];
    if (!isPerTensorLayerSeriesLabels(seriesLabelsResolved)) return [];
    return sortedTensorFamiliesFromLabels(seriesLabelsResolved);
  }, [vizVariant, seriesLabelsResolved]);

  const isTensorFamilyL2Viz =
    !isHessianEigenViz &&
    (vizVariant === "gradient_norm" || vizVariant === "weight_l2") &&
    tensorFamilyOptions.length > 0;

  const selectedTensorFamily =
    typeof raw.l2TensorFamily === "string" && tensorFamilyOptions.includes(raw.l2TensorFamily)
      ? raw.l2TensorFamily
      : tensorFamilyOptions[0] ?? "";

  const l2DisplayIndices = useMemo(() => {
    if (!isTensorFamilyL2Viz || seriesLabelsResolved.length === 0) {
      return Array.from({ length: seriesList.length }, (_, i) => i);
    }
    const fam = selectedTensorFamily;
    const out: number[] = [];
    if (seriesList.length > 0) out.push(0);
    for (let i = 1; i < seriesList.length; i++) {
      const lab = seriesLabelsResolved[i];
      if (typeof lab === "string" && tensorFamilyFromParameterLabel(lab) === fam) out.push(i);
    }
    return out;
  }, [isTensorFamilyL2Viz, seriesLabelsResolved, selectedTensorFamily, seriesList.length]);

  const tensorFamilyOptsKey = tensorFamilyOptions.join("\u0001");
  useEffect(() => {
    if (!isTensorFamilyL2Viz || tensorFamilyOptions.length === 0) return;
    const cur = typeof raw.l2TensorFamily === "string" ? raw.l2TensorFamily : "";
    if (!cur || !tensorFamilyOptions.includes(cur)) {
      update({ l2TensorFamily: tensorFamilyOptions[0] });
    }
  }, [isTensorFamilyL2Viz, tensorFamilyOptsKey, raw.l2TensorFamily]);

  const checkboxCount = isHessianEigenViz
    ? rankCount
    : isTensorFamilyL2Viz
      ? l2DisplayIndices.length
      : rankCount;

  const eigenSeriesVisible = useMemo(() => {
    const n = rankCount;
    const out = Array.from({ length: n }, () => true);
    const rawVis = d.eigenSeriesVisible;
    if (Array.isArray(rawVis)) {
      for (let i = 0; i < n; i++) out[i] = rawVis[i] !== false;
    } else if (d.showSeries === false) {
      for (let i = 0; i < n; i++) out[i] = false;
    }
    return out;
  }, [rankCount, d.eigenSeriesVisible, d.showSeries]);

  const steps = useMemo(() => {
    const st = d.stepTicks;
    if (!Array.isArray(st) || st.length < 1) return [];
    if (seriesList.length === 0) return [];
    const n = st.length;
    if (!seriesList.every((s) => s.length === n)) return [];
    return st.map(Number);
  }, [d.stepTicks, seriesList]);

  const bounds = useMemo(() => {
    if (isTensorFamilyL2Viz) {
      const lists = l2DisplayIndices.map((fi) => seriesList[fi]!);
      const vis = l2DisplayIndices.map((fi) => eigenSeriesVisible[fi] !== false);
      return computeBoundsMulti(steps, lists, vis, !!d.logScaleX, !!d.logScaleY);
    }
    return computeBoundsMulti(steps, seriesList, eigenSeriesVisible, !!d.logScaleX, !!d.logScaleY);
  }, [steps, seriesList, eigenSeriesVisible, d.logScaleX, d.logScaleY, isTensorFamilyL2Viz, l2DisplayIndices]);
  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - PAD_R;
  const anyEigenPlotted =
    seriesList.length > 0 &&
    (isTensorFamilyL2Viz
      ? l2DisplayIndices.some((fi) => eigenSeriesVisible[fi] !== false)
      : seriesList.some((_, si) => eigenSeriesVisible[si] !== false));
  const hasSeries =
    steps.length >= 1 &&
    seriesList.length > 0 &&
    seriesList.every((s) => s.length === steps.length) &&
    (!isTensorFamilyL2Viz || l2DisplayIndices.length > 0) &&
    anyEigenPlotted;
  const anyFiniteY = useMemo(() => {
    if (isTensorFamilyL2Viz) {
      return l2DisplayIndices.some((fi) => {
        if (eigenSeriesVisible[fi] === false) return false;
        return seriesList[fi]!.some((v) => Number.isFinite(v));
      });
    }
    return seriesList.some((row, si) => {
      if (eigenSeriesVisible[si] === false) return false;
      return row.some((v) => Number.isFinite(v));
    });
  }, [seriesList, eigenSeriesVisible, isTensorFamilyL2Viz, l2DisplayIndices]);
  const hasDrawableSeries = hasSeries && anyFiniteY;

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
    enabled: hasDrawableSeries,
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

  const paths = useMemo(() => {
    if (isTensorFamilyL2Viz) {
      return l2DisplayIndices.map((fi) =>
        eigenSeriesVisible[fi] === false
          ? ""
          : buildPathFinite(steps, seriesList[fi]!, !!d.logScaleX, !!d.logScaleY, viewBounds),
      );
    }
    return seriesList.map((vals, si) =>
      eigenSeriesVisible[si] === false
        ? ""
        : buildPathFinite(steps, vals, !!d.logScaleX, !!d.logScaleY, viewBounds),
    );
  }, [d.logScaleX, d.logScaleY, steps, seriesList, viewBounds, eigenSeriesVisible, isTensorFamilyL2Viz, l2DisplayIndices]);

  const legendIndices = useMemo(
    () => (isTensorFamilyL2Viz ? l2DisplayIndices : Array.from({ length: seriesList.length }, (_, si) => si)),
    [isTensorFamilyL2Viz, l2DisplayIndices, seriesList.length],
  );

  const plotMidX = PAD_L + (CHART_W - PAD_L - PAD_R) / 2;
  const plotMidY = PAD_T + (CHART_H - PAD_T - PAD_B) / 2;
  const clipId = `${id}-ohess-clip`;

  const orderWord = d.order === "ascending" ? "smallest-first" : "largest-first";
  const allRanksHidden =
    seriesList.length > 0 &&
    steps.length >= 1 &&
    seriesList.every((s) => s.length === steps.length) &&
    (isTensorFamilyL2Viz
      ? !l2DisplayIndices.some((fi) => eigenSeriesVisible[fi] !== false)
      : !seriesList.some((_, si) => eigenSeriesVisible[si] !== false));
  const metricKind =
    vizVariant === "gradient_norm"
      ? "Gradient L2 norms"
      : vizVariant === "weight_l2"
        ? "Weight L2 norms"
        : vizVariant === "activation_stats"
          ? "Activation statistics"
          : vizVariant === "weight_product_sv"
            ? "Weight product singular values"
            : "Hessian eigenvalues";
  const hintSeriesCount = isTensorFamilyL2Viz ? l2DisplayIndices.length : seriesList.length;
  const hint = allRanksHidden
    ? vizVariant === "hessian_eigenvalues" || !vizVariant
      ? "All eigenvalue curves are hidden — enable at least one rank above the chart."
      : "All curves are hidden — enable at least one series above the chart."
    : hasSeries
      ? anyFiniteY
        ? vizVariant === "hessian_eigenvalues" || !vizVariant
          ? `Hessian eigenvalues (${seriesList.length} ranks, ${orderWord})`
          : vizVariant === "weight_product_sv"
            ? `Weight product singular values (${hintSeriesCount} modes)`
            : isTensorFamilyL2Viz
              ? `${metricKind} (${hintSeriesCount} series · ${selectedTensorFamily})`
              : `${metricKind} (${hintSeriesCount} series)`
        : vizVariant === "hessian_eigenvalues" || !vizVariant
          ? "Hessian values are all non-finite (e.g. exact Hessian is limited to small models — try fewer / smaller layers, or check console)."
          : "Values are all non-finite — check console / model wiring."
      : "No data yet — connect Trainer “observable”, then Train.";

  const toggleEigenVisible = (idx: number, vis: boolean) => {
    const targetIdx = isTensorFamilyL2Viz ? l2DisplayIndices[idx]! : idx;
    const next = eigenSeriesVisible.slice();
    while (next.length < rankCount) next.push(true);
    next[targetIdx] = vis;
    update({ eigenSeriesVisible: next, showSeries: undefined });
  };

  const rankLabel = (uiIdx: number) => {
    const idx = isTensorFamilyL2Viz ? l2DisplayIndices[uiIdx]! : uiIdx;
    const lab = seriesLabels && seriesLabels[idx];
    if (lab) {
      if (isTensorFamilyL2Viz && idx > 0) return tensorCurveShortLabel(lab);
      return lab;
    }
    if (vizVariant === "gradient_norm" || vizVariant === "weight_l2" || vizVariant === "activation_stats") {
      return `series ${uiIdx + 1}`;
    }
    if (vizVariant === "weight_product_sv") {
      return `σ${uiIdx + 1}`;
    }
    const ord = d.order === "ascending" ? "asc" : "desc";
    return `λ${uiIdx + 1} (${ord})`;
  };

  const yAxisTitle =
    vizVariant === "weight_product_sv"
      ? "σ"
      : vizVariant === "gradient_norm" || vizVariant === "weight_l2" || vizVariant === "activation_stats"
        ? "value"
        : "λ";

  return (
    <div
      className={`cr-node cr-node--observable-viz-hessian-eigenvalues${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar annotatorSource />
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
          {isHessianEigenViz ? (
            <label className="cr-tviz-check cr-tviz-check--threshold" title="Draw a horizontal reference line at this λ value (e.g. 2/η for Edge-of-Stability threshold). Leave empty to hide.">
              <span>2/η&thinsp;=</span>
              <input
                type="number"
                className="cr-tviz-threshold-input nodrag nopan"
                step="any"
                min="0"
                placeholder="—"
                value={typeof d.sharpnessThreshold === "number" ? d.sharpnessThreshold : ""}
                onPointerDown={(e) => e.stopPropagation()}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  update({ sharpnessThreshold: Number.isFinite(v) && v > 0 ? v : undefined });
                }}
              />
            </label>
          ) : null}
          {isZoomed ? (
            <button type="button" className="cr-tviz-reset-zoom" onClick={resetZoom}>
              reset zoom-in
            </button>
          ) : null}
        </div>
        {isTensorFamilyL2Viz ? (
          <div className="cr-tviz-tensor-family-row nodrag nopan">
            <DiscreteMultiSelect
              label="Tensor kind"
              options={tensorFamilyOptions.map((x) => ({ id: x, label: x }))}
              value={selectedTensorFamily}
              onCommit={(next) => {
                const v = Array.isArray(next) ? next[0] : next;
                if (typeof v === "string") update({ l2TensorFamily: v });
              }}
              ariaLabel="Per-parameter tensor kind"
              singleSelect
            />
          </div>
        ) : null}
        {checkboxCount > 0 ? (
          <div className="cr-tviz-chart-controls cr-tviz-eigen-ranks-row nodrag nopan" aria-label="Metric curves">
            {Array.from({ length: checkboxCount }, (_, uiIdx) => {
              const fi = isTensorFamilyL2Viz ? l2DisplayIndices[uiIdx]! : uiIdx;
              return (
                <label key={`evis-${fi}-${uiIdx}`} className="cr-tviz-check cr-tviz-check--eigen-rank">
                  <input
                    type="checkbox"
                    checked={eigenSeriesVisible[fi] !== false}
                    onChange={(e) => toggleEigenVisible(uiIdx, e.target.checked)}
                  />
                  {rankLabel(uiIdx)}
                </label>
              );
            })}
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
            aria-label={
              vizVariant === "gradient_norm"
                ? "Gradient norms over training"
                : vizVariant === "weight_l2"
                  ? "Weight L2 norms over training"
                  : vizVariant === "activation_stats"
                    ? "Activation statistics over training"
                    : vizVariant === "weight_product_sv"
                      ? "Weight product singular values over training"
                      : "Hessian eigenvalues over training"
            }
            onMouseDown={(e) => {
              if (!hasDrawableSeries) return;
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
            style={{ cursor: hasDrawableSeries ? (isDraggingZoom ? "crosshair" : "zoom-in") : "default" }}
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

            {paths.map((dPath, uiPos) =>
              dPath ? (
                <path
                  key={`p-${uiPos}`}
                  d={dPath}
                  fill="none"
                  stroke={SERIES_COLORS[uiPos % SERIES_COLORS.length]!}
                  strokeWidth={1.5}
                  clipPath={`url(#${clipId})`}
                />
              ) : null,
            )}
            {isHessianEigenViz && typeof d.sharpnessThreshold === "number" && d.sharpnessThreshold > 0
              ? (() => {
                  const tRaw = transformY(d.sharpnessThreshold, !!d.logScaleY);
                  const py = yToPx(tRaw, viewBounds);
                  if (!Number.isFinite(py) || py < PAD_T - 1 || py > innerBottom + 1) return null;
                  const labelX = innerRight - 2;
                  return (
                    <g key="threshold-line" clipPath={`url(#${clipId})`}>
                      <line
                        x1={PAD_L}
                        y1={py}
                        x2={innerRight}
                        y2={py}
                        stroke="#f97316"
                        strokeWidth={1.5}
                        strokeDasharray="5 3"
                        opacity={0.9}
                      />
                      <text
                        x={labelX}
                        y={py - 3}
                        textAnchor="end"
                        className="cr-tviz-chart__tick-label"
                        style={{ fill: "#f97316", fontSize: "9px", fontWeight: 600 }}
                      >
                        2/η
                      </text>
                    </g>
                  );
                })()
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
              {yAxisTitle}
            </text>

            {hasDrawableSeries ? (
              <g className="cr-tviz-legend" transform={`translate(${PAD_L + 4}, ${PAD_T + 4})`}>
                {legendIndices.map((fi, uiPos) => {
                  if (eigenSeriesVisible[fi] === false) return null;
                  const row = legendIndices
                    .slice(0, uiPos)
                    .filter((fj) => eigenSeriesVisible[fj] !== false).length;
                  return (
                    <g key={`leg-${fi}`} transform={`translate(0, ${row * 12})`}>
                      <line
                        x1={0}
                        y1={5}
                        x2={12}
                        y2={5}
                        stroke={SERIES_COLORS[uiPos % SERIES_COLORS.length]!}
                        strokeWidth={1.5}
                      />
                      <text x={14} y={7} className="cr-tviz-chart__legend-text" fontSize="9">
                        {rankLabel(uiPos)}
                      </text>
                    </g>
                  );
                })}
              </g>
            ) : null}
          </svg>
          <p className="cr-tviz-hint">{hint}</p>
        </div>
      </div>
    </div>
  );
}
