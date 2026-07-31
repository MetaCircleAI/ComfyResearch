import type { Edge, Node } from "@xyflow/react";
import type { CurvePoint } from "./observableCurvePayload";
import { buildCurvePayloadFromObservableVizNode } from "./observableCurvePayload";
import {
  buildCurveStarerEntryLabelForNode,
  expandWildcardMemberLabel,
  labelContainsWildcard,
  resolveCurveStarerObservableLabel,
} from "./curveStarerLabels";

export type ObservableCurveEntry = {
  entryId: string;
  nodeId: string;
  seriesId: string;
  label: string;
  yAxisLabel: string;
  points: CurvePoint[];
};

export type CollectObservableCurvesOptions = {
  /** Minimum logged points per series (2 matches live viz; LPD still prefers ≥ 5). */
  minPoints?: number;
};

/** Viz nodes whose plotted series CurveStarer analyzes (not metric source nodes). */
const CURVE_STARER_NODE_TYPES = new Set(["observable_viz", "training_visualization"]);

export const LPD_MIN_CURVE_POINTS = 5;

function curvePointsKey(points: CurvePoint[]): string {
  return points.map((p) => `${p.t},${p.loss}`).join("|");
}

function finiteNumberList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
}

function readFiniteSeries(data: Record<string, unknown>, ...keys: string[]): number[] {
  for (const key of keys) {
    const arr = finiteNumberList(data[key]);
    if (arr.length) return arr;
  }
  return [];
}

function defaultStepTicks(length: number): number[] {
  return Array.from({ length }, (_, i) => i + 1);
}

function buildSeriesPoints(xs: number[], ys: number[], minPoints: number): CurvePoint[] {
  const len = Math.min(xs.length, ys.length);
  if (len < minPoints) return [];
  const points = Array.from({ length: len }, (_, i) => ({ t: xs[i]!, loss: ys[i]! }))
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.loss))
    .sort((a, b) => a.t - b.t);
  if (points.length < minPoints) return [];
  const deduped: CurvePoint[] = [];
  for (const point of points) {
    if (!deduped.length || Math.abs(deduped[deduped.length - 1]!.t - point.t) > 1e-9) {
      deduped.push(point);
    }
  }
  return deduped.length >= minPoints ? deduped : [];
}

function seriesFromHistory(
  stepTicks: number[],
  values: number[],
  minPoints: number,
): CurvePoint[] {
  if (values.length < minPoints) return [];
  const xs = stepTicks.length >= minPoints && stepTicks.length === values.length
    ? stepTicks
    : defaultStepTicks(values.length);
  return buildSeriesPoints(xs, values, minPoints);
}

/** Stable fingerprint for the current observable / training-viz curve set. */
export function curveStarerScenarioFingerprint(curves: ObservableCurveEntry[]): string {
  return curves
    .slice()
    .sort((a, b) => a.entryId.localeCompare(b.entryId))
    .map((curve) => `${curve.entryId}\t${curvePointsKey(curve.points)}`)
    .join("\n");
}

function pushCurve(
  out: ObservableCurveEntry[],
  seenEntryIds: Set<string>,
  seenPointKeys: Set<string>,
  entry: ObservableCurveEntry,
  minPoints: number,
): void {
  if (entry.points.length < minPoints) return;
  if (seenEntryIds.has(entry.entryId)) return;
  const pk = curvePointsKey(entry.points);
  if (seenPointKeys.has(pk)) return;
  seenEntryIds.add(entry.entryId);
  seenPointKeys.add(pk);
  out.push(entry);
}

function collectFromVizNodes(
  nodes: Node[],
  nodesById: Map<string, Node>,
  out: ObservableCurveEntry[],
  seenEntryIds: Set<string>,
  seenPointKeys: Set<string>,
  minPoints: number,
): void {
  for (const node of nodes) {
    if (!CURVE_STARER_NODE_TYPES.has(String(node.type ?? ""))) continue;
    const payload = buildCurvePayloadFromObservableVizNode(node);
    if (!payload) continue;
    const sourceLabel = resolveCurveStarerObservableLabel(node, nodesById);
    const multiSeries = payload.curves.length > 1;
    for (const curve of payload.curves) {
      if (curve.points.length < minPoints) continue;
      pushCurve(
        out,
        seenEntryIds,
        seenPointKeys,
        {
          entryId: `${node.id}:${curve.id}`,
          nodeId: node.id,
          seriesId: curve.id,
          label: buildCurveStarerEntryLabelForNode(node, curve, sourceLabel, multiSeries),
          yAxisLabel: payload.yAxisLabel ?? "value",
          points: curve.points,
        },
        minPoints,
      );
    }
  }
}

function collectFromTrainer(
  trainer: Node,
  nodes: Node[],
  edges: Edge[],
  nodesById: Map<string, Node>,
  out: ObservableCurveEntry[],
  seenEntryIds: Set<string>,
  seenPointKeys: Set<string>,
  minPoints: number,
): void {
  const td = (trainer.data ?? {}) as Record<string, unknown>;
  const stepTicks = readFiniteSeries(td, "stepTicks");
  const omh = (td.observableMetricHistories ?? {}) as Record<string, number[]>;

  for (const edge of edges) {
    if (edge.source !== trainer.id) continue;
    const sh = edge.sourceHandle ?? "";
    if (sh === "loss_results") {
      const viz = nodesById.get(edge.target);
      if (!viz || viz.type !== "training_visualization") continue;
      const trainLoss = readFiniteSeries(td, "lossHistory");
      const testLoss = readFiniteSeries(td, "testLossHistory");
      const trainPoints = seriesFromHistory(stepTicks, trainLoss, minPoints);
      if (trainPoints.length >= minPoints) {
        pushCurve(
          out,
          seenEntryIds,
          seenPointKeys,
          {
            entryId: `${viz.id}:train`,
            nodeId: viz.id,
            seriesId: "train",
            label: "train loss",
            yAxisLabel: "loss",
            points: trainPoints,
          },
          minPoints,
        );
      }
      const testPoints = seriesFromHistory(stepTicks, testLoss, minPoints);
      if (testPoints.length >= minPoints) {
        pushCurve(
          out,
          seenEntryIds,
          seenPointKeys,
          {
            entryId: `${viz.id}:test`,
            nodeId: viz.id,
            seriesId: "test",
            label: "test loss",
            yAxisLabel: "loss",
            points: testPoints,
          },
          minPoints,
        );
      }
      continue;
    }
    if (sh !== "observable_results") continue;
    const viz = nodesById.get(edge.target);
    if (!viz || viz.type !== "observable_viz") continue;
    const vd = (viz.data ?? {}) as Record<string, unknown>;
    const pairedId = String(vd.pairedObservableId ?? "").trim();
    if (!pairedId) continue;
    const sourceLabel = resolveCurveStarerObservableLabel(viz, nodesById);
    const isAccuracy =
      vd.vizVariant === "accuracy" || sourceLabel.trim().toLowerCase() === "accuracy";

    const memberPrefix = `${pairedId}::member::`;
    const memberKeys: string[] = [];
    for (const key of Object.keys(omh)) {
      if (!key.startsWith(memberPrefix)) continue;
      const hist = omh[key];
      if (Array.isArray(hist) && hist.length >= minPoints) {
        memberKeys.push(key.slice(memberPrefix.length));
      }
    }
    memberKeys.sort();
    const expandWildcard = labelContainsWildcard(sourceLabel) && memberKeys.length > 0;

    const trainHist = omh[pairedId];
    if (!expandWildcard && Array.isArray(trainHist) && trainHist.length >= minPoints) {
      const points = seriesFromHistory(stepTicks, trainHist, minPoints);
      pushCurve(
        out,
        seenEntryIds,
        seenPointKeys,
        {
          entryId: `${viz.id}:train`,
          nodeId: viz.id,
          seriesId: "train",
          label: isAccuracy ? "train accuracy" : `${sourceLabel} · train`,
          yAxisLabel: isAccuracy ? "accuracy" : sourceLabel,
          points,
        },
        minPoints,
      );
    }

    const testHist = omh[`${pairedId}::test`];
    if (!expandWildcard && Array.isArray(testHist) && testHist.length >= minPoints) {
      const points = seriesFromHistory(stepTicks, testHist, minPoints);
      pushCurve(
        out,
        seenEntryIds,
        seenPointKeys,
        {
          entryId: `${viz.id}:test`,
          nodeId: viz.id,
          seriesId: "test",
          label: isAccuracy ? "test accuracy" : `${sourceLabel} · test`,
          yAxisLabel: isAccuracy ? "accuracy" : sourceLabel,
          points,
        },
        minPoints,
      );
    }

    for (const memberKey of memberKeys) {
      const hist = omh[`${memberPrefix}${memberKey}`];
      if (!Array.isArray(hist) || hist.length < minPoints) continue;
      const points = seriesFromHistory(stepTicks, hist, minPoints);
      const memberLabel = expandWildcardMemberLabel(
        `${sourceLabel} · train`,
        memberKey,
      );
      pushCurve(
        out,
        seenEntryIds,
        seenPointKeys,
        {
          entryId: `${viz.id}:member:${memberKey}`,
          nodeId: viz.id,
          seriesId: `member_${memberKey}`,
          label: memberLabel,
          yAxisLabel: sourceLabel,
          points,
        },
        minPoints,
      );
    }
  }
}

/** Collect plottable series from viz nodes and trainer metric histories (train, test, multi-series). */
export function collectObservableTrainingCurves(
  nodes: Node[],
  edges: Edge[] = [],
  options: CollectObservableCurvesOptions = {},
): ObservableCurveEntry[] {
  const minPoints = Math.max(2, options.minPoints ?? 2);
  const nodesById = new Map(nodes.map((n) => [n.id, n]));
  const out: ObservableCurveEntry[] = [];
  const seenEntryIds = new Set<string>();
  const seenPointKeys = new Set<string>();

  collectFromVizNodes(nodes, nodesById, out, seenEntryIds, seenPointKeys, minPoints);

  for (const node of nodes) {
    if (node.type !== "trainer") continue;
    collectFromTrainer(node, nodes, edges, nodesById, out, seenEntryIds, seenPointKeys, minPoints);
  }

  return out;
}
