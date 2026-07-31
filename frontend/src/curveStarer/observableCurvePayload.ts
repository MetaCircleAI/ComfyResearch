import type { Node } from "@xyflow/react";

/** One sample on an observable curve (`t` = step tick, `loss` = value). */
export type CurvePoint = { t: number; loss: number };

export type CurveSeries = {
  id: string;
  label: string;
  points: CurvePoint[];
};

export type ObservableCurvePayload = {
  sourceNodeId: string;
  sourceLabel: string;
  yAxisLabel?: string;
  curves: CurveSeries[];
};

/** Window event that asks the canvas to open the CurveStarer modal. */
export const OPEN_CURVE_STARER_EVENT = "cr:open-curve-starer";

export function nodeDisplayLabel(node: Node): string {
  const data = (node.data ?? {}) as Record<string, unknown>;
  return (
    (typeof data.instanceTitle === "string" && data.instanceTitle.trim()) ||
    (typeof data.observableName === "string" && data.observableName.trim()) ||
    (typeof data.displayName === "string" && data.displayName.trim()) ||
    String(node.type ?? node.id)
  );
}

function finiteNumberList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
}

function curvePointsKey(points: CurvePoint[]): string {
  return points.map((p) => `${p.t},${p.loss}`).join("|");
}

function buildSeriesPoints(xs: number[], ys: number[]): CurvePoint[] {
  const len = Math.min(xs.length, ys.length);
  if (len < 5) return [];
  const points = Array.from({ length: len }, (_, i) => ({ t: xs[i]!, loss: ys[i]! }))
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.loss))
    .sort((a, b) => a.t - b.t);
  if (points.length < 5) return [];
  const deduped: CurvePoint[] = [];
  for (const point of points) {
    if (!deduped.length || Math.abs(deduped[deduped.length - 1]!.t - point.t) > 1e-9) {
      deduped.push(point);
    }
  }
  return deduped.length >= 5 ? deduped : [];
}

function seriesAlreadyPresent(curves: CurveSeries[], points: CurvePoint[]): boolean {
  if (points.length < 5) return true;
  const key = curvePointsKey(points);
  return curves.some((c) => c.points.length >= 5 && curvePointsKey(c.points) === key);
}

function deriveYAxisLabel(node: Node): string {
  const nodeType = String(node.type ?? "");
  if (nodeType === "training_visualization") return "loss";
  const base = nodeDisplayLabel(node).trim();
  if (!base) return "loss";
  const stripped = base.replace(/\s*viz\s*$/i, "").trim();
  return stripped || "loss";
}

function readFiniteSeries(data: Record<string, unknown>, ...keys: string[]): number[] {
  for (const key of keys) {
    const arr = finiteNumberList(data[key]);
    if (arr.length) return arr;
  }
  return [];
}

export function buildCurvePayloadFromObservableVizNode(node: Node): ObservableCurvePayload | null {
  const nodeType = String(node.type ?? "");
  const supportedTypes = new Set(["observable_viz", "observable_accuracy", "training_visualization"]);
  if (!supportedTypes.has(nodeType)) return null;
  const data = (node.data ?? {}) as Record<string, unknown>;
  const stepTicks = readFiniteSeries(data, "stepTicks", "step_ticks");
  let valueHistory = readFiniteSeries(data, "valueHistory", "lossHistory", "value_history", "loss_history");
  let testValueHistory = readFiniteSeries(
    data,
    "testValueHistory",
    "testLossHistory",
    "test_value_history",
    "test_loss_history",
  );
  if (
    nodeType === "training_visualization" &&
    String(data.yPlotMetric ?? "").toLowerCase() === "perplexity"
  ) {
    const toPerplexity = (v: number) => {
      if (!Number.isFinite(v)) return NaN;
      const p = Math.exp(v);
      return Number.isFinite(p) ? p : NaN;
    };
    valueHistory = valueHistory.map(toPerplexity).filter((v) => Number.isFinite(v));
    testValueHistory = testValueHistory.map(toPerplexity).filter((v) => Number.isFinite(v));
  }
  const histRows = Array.isArray(data.valueHistories)
    ? (data.valueHistories as unknown[]).map((row) => finiteNumberList(row))
    : [];
  const seriesLabels = Array.isArray(data.seriesLabels)
    ? (data.seriesLabels as unknown[]).map((x) => String(x ?? "").trim())
    : [];

  const inferredLen = Math.max(
    stepTicks.length,
    valueHistory.length,
    testValueHistory.length,
    ...histRows.map((row) => row.length),
  );
  if (inferredLen < 5) return null;
  const xs = stepTicks.length >= 5 ? stepTicks : Array.from({ length: inferredLen }, (_, i) => i + 1);

  const curves: CurveSeries[] = [];
  histRows.forEach((row, index) => {
    const points = buildSeriesPoints(xs, row);
    if (points.length < 5) return;
    const raw = seriesLabels[index] ?? "";
    const label = raw || `series ${index + 1}`;
    curves.push({ id: `series_${index + 1}`, label, points });
  });
  const trainPoints = buildSeriesPoints(xs, valueHistory);
  if (trainPoints.length >= 5 && !seriesAlreadyPresent(curves, trainPoints)) {
    curves.push({ id: "train", label: "train", points: trainPoints });
  }
  const testPoints = buildSeriesPoints(xs, testValueHistory);
  if (testPoints.length >= 5 && !seriesAlreadyPresent(curves, testPoints)) {
    curves.push({ id: "test", label: "test", points: testPoints });
  }
  if (curves.length === 0) return null;

  return {
    sourceNodeId: node.id,
    sourceLabel: nodeDisplayLabel(node),
    yAxisLabel: deriveYAxisLabel(node),
    curves,
  };
}
