/**
 * Train-complete accumulation wiring for curve_series_table nodes.
 * The parametric-path push path (appendParametricPathToCurveTables) lives in
 * curveSeriesParametricAppend.ts; the shared row upsert is imported from there.
 */
import type { Edge, Node } from "@xyflow/react";
import {
  CURVE_SERIES_METRIC_LABELS,
  DEFAULT_CURVE_SERIES_CAPTURE_METRICS,
  type CurveSeriesTableNodeData,
} from "../components/nodes/curveSeriesDefaults";
import type { ObservableVizUserNodeData } from "../components/nodes/observableVizUserDefaults";
import type { ParametricPathSamplerNodeData } from "../components/nodes/parametricPathSamplerDefaults";
import type { SweepDataTableNodeData } from "../components/nodes/sweepDataTableDefaults";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { appendRowsToTable, type ExtractedCurve } from "./curveSeriesParametricAppend";
import { readInstanceTitle } from "./nodeInstanceTitle";
import {
  buildTrainerRunSweepParams,
  coerceSweepParamsNumeric,
  formatSweepParamsSummary,
  mergeSweepParamRecords,
  parseSweepParamsFromSummary,
} from "./sweepParamExtract";

function nodeIdsInTrainerGraphComponent(
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
  types: Set<string>,
): Set<string> {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  if (!nodeById.has(trainerNodeId)) return new Set();

  const adj = new Map<string, string[]>();
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, []);
    if (!adj.has(e.target)) adj.set(e.target, []);
    adj.get(e.source)!.push(e.target);
    adj.get(e.target)!.push(e.source);
  }

  const out = new Set<string>();
  const seen = new Set<string>();
  const q: string[] = [trainerNodeId];
  seen.add(trainerNodeId);
  while (q.length) {
    const id = q.shift()!;
    const n = nodeById.get(id);
    if (n && types.has(String(n.type))) out.add(id);
    for (const nb of adj.get(id) ?? []) {
      if (seen.has(nb)) continue;
      seen.add(nb);
      q.push(nb);
    }
  }
  return out;
}

function finiteNumberArray(raw: unknown): number[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((v) => Number(v)).filter((n) => Number.isFinite(n));
}

function alignedSteps(steps: number[], values: number[]): { x: number[]; y: number[] } {
  const n = Math.min(steps.length, values.length);
  if (n === 0) return { x: [], y: [] };
  return { x: steps.slice(0, n), y: values.slice(0, n) };
}

function alignedEpochs(epochs: number[], length: number): number[] | undefined {
  if (epochs.length < length) return undefined;
  return epochs.slice(0, length);
}

function metricLabel(metricId: string): string {
  return CURVE_SERIES_METRIC_LABELS[metricId] ?? metricId;
}

function extractFromTrainingVisualization(
  data: TrainingVisualizationNodeData,
  captureMetrics: string[],
): ExtractedCurve[] {
  const steps = finiteNumberArray(data.stepTicks);
  const epochs = finiteNumberArray(data.epochTicks);
  const out: ExtractedCurve[] = [];
  for (const metricId of captureMetrics) {
    let values: number[] = [];
    if (metricId === "train_loss") values = finiteNumberArray(data.lossHistory);
    else if (metricId === "test_loss") values = finiteNumberArray(data.testLossHistory);
    else if (metricId === "reg_loss") values = finiteNumberArray(data.regLossHistory);
    else continue;
    const { x, y } = alignedSteps(steps, values);
    if (y.length === 0) continue;
    out.push({ metricId, label: metricLabel(metricId), x, epochX: alignedEpochs(epochs, y.length), y });
  }
  return out;
}

function observableMetaForViz(
  data: ObservableVizUserNodeData,
  nodeById: Map<string, Node>,
): { metricId: string; label: string } | null {
  const paired = typeof data.pairedObservableId === "string" ? data.pairedObservableId.trim() : "";
  if (!paired) return null;
  const obs = nodeById.get(paired);
  const obsType = obs?.type ? String(obs.type) : "";
  if (!obsType) return null;
  // label 回退链(同型多观测量/用户改名场景):viz 的 observableName
  // → 观测量节点 instanceTitle → 生成 spec 标签 → 类型名。
  const vizName = typeof (data as { observableName?: unknown }).observableName === "string"
    ? String((data as { observableName?: unknown }).observableName).trim()
    : "";
  const instanceTitle = readInstanceTitle(obs?.data, "").trim();
  const label = vizName || instanceTitle || GENERATED_NODE_SPECS[obsType]?.label || obsType;
  return { metricId: `observable:${obsType}`, label };
}

function extractFromObservableViz(
  data: ObservableVizUserNodeData,
  captureMetrics: string[],
  nodeById: Map<string, Node>,
): ExtractedCurve[] {
  const steps = finiteNumberArray(data.stepTicks);
  const epochs = finiteNumberArray(data.epochTicks);
  const out: ExtractedCurve[] = [];
  for (const metricId of captureMetrics) {
    // 通用观测量捕获:抽 valueHistory,metricId 按 obsType 落表
    // (observable:<type>),label 取生成 spec 的节点标签;acc 两 id 语义不动。
    if (metricId === "observable") {
      const meta = observableMetaForViz(data, nodeById);
      if (!meta) continue;
      const { x, y } = alignedSteps(steps, finiteNumberArray(data.valueHistory));
      if (y.length === 0) continue;
      out.push({ metricId: meta.metricId, label: meta.label, x, y });
      continue;
    }
    let values: number[] = [];
    if (metricId === "train_acc") values = finiteNumberArray(data.valueHistory);
    else if (metricId === "test_acc") values = finiteNumberArray(data.testValueHistory);
    else continue;
    const { x, y } = alignedSteps(steps, values);
    if (y.length === 0) continue;
    out.push({ metricId, label: metricLabel(metricId), x, epochX: alignedEpochs(epochs, y.length), y });
  }
  return out;
}

function extractFromParametricPathSampler(data: ParametricPathSamplerNodeData): ExtractedCurve[] {
  const x = finiteNumberArray(data.alphaSeries);
  const y = finiteNumberArray(data.valueSeries);
  if (y.length === 0) return [];
  const label = typeof data.seriesLabel === "string" && data.seriesLabel.trim() ? data.seriesLabel.trim() : "parametric path";
  const metricId = data.metric === "accuracy" ? (data.split === "test" ? "test_acc" : "train_acc") : (data.split === "test" ? "test_loss" : "train_loss");
  return [{ metricId, label, x: x.length ? x : y.map((_, i) => i), y }];
}

function extractFromStreamSource(src: Node, captureMetrics: string[], nodeById: Map<string, Node>): ExtractedCurve[] {
  if (src.type === "training_visualization") {
    return extractFromTrainingVisualization((src.data ?? {}) as TrainingVisualizationNodeData, captureMetrics);
  }
  if (src.type === "observable_viz") {
    return extractFromObservableViz((src.data ?? {}) as ObservableVizUserNodeData, captureMetrics, nodeById);
  }
  if (src.type === "parametric_path_sampler") {
    return extractFromParametricPathSampler((src.data ?? {}) as ParametricPathSamplerNodeData);
  }
  return [];
}

/** All stream sources on the table that belong to this trainer's run
 * (多源:Fig-1 需要 loss viz + observable viz 两路进同一张表;
 * 显式配对优先,未配对的按"可达且不跨其他 trainer"归属;皆空回退首边)。 */
function streamSourcesForTrainer(
  nodes: Node[],
  edges: Edge[],
  tableId: string,
  trainerNodeId: string,
): Node[] {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const incoming = edges.filter(
    (e) => e.target === tableId && (e.targetHandle ?? "") === "stream",
  );
  if (incoming.length === 0) return [];
  const first = nodeById.get(incoming[0]!.source);
  if (incoming.length === 1) return first ? [first] : [];

  const reachable = new Set<string>();
  const q: string[] = [trainerNodeId];
  reachable.add(trainerNodeId);
  while (q.length) {
    const id = q.shift()!;
    for (const e of edges) {
      let nb: string | null = null;
      if (e.source === id) nb = e.target;
      else if (e.target === id) nb = e.source;
      if (!nb || reachable.has(nb)) continue;
      const nbNode = nodeById.get(nb);
      if (nbNode?.type === "trainer" && nb !== trainerNodeId) continue;
      // 表是汇:归属可达性不得经表桥接到别的 trainer 的 viz。
      if (nbNode?.type === "curve_series_table") continue;
      reachable.add(nb);
      q.push(nb);
    }
  }

  const out: Node[] = [];
  const seen = new Set<string>();
  for (const e of incoming) {
    const src = nodeById.get(e.source);
    if (!src || seen.has(src.id)) continue;
    const paired = (src.data as { pairedTrainerId?: string } | undefined)?.pairedTrainerId;
    const belongs = typeof paired === "string" && paired ? paired === trainerNodeId : reachable.has(src.id);
    if (!belongs) continue;
    seen.add(src.id);
    out.push(src);
  }
  if (out.length > 0) return out;
  return first ? [first] : [];
}

function streamSourcesForTable(nodes: Node[], edges: Edge[], tableId: string): Node[] {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  return edges
    .filter((e) => e.target === tableId && (e.targetHandle ?? "") === "stream")
    .map((e) => nodeById.get(e.source))
    .filter((n): n is Node => n != null);
}

function trainerRunLabel(nodes: Node[], trainerNodeId: string): string {
  const trainer = nodes.find((n) => n.id === trainerNodeId);
  if (!trainer) return trainerNodeId;
  return readInstanceTitle(trainer.data, trainerNodeId);
}

function curveSeriesParamsForRun(
  nodes: Node[],
  trainerNodeId: string,
  sweepSummary?: string,
  sweepParams?: Record<string, string>,
  streamSrc?: Node | null,
): { rawSweep: string; params: Record<string, string>; paramsNumeric: Record<string, number> } {
  const trainerNode = nodes.find((n) => n.id === trainerNodeId);
  const trainerParams = trainerNode
    ? buildTrainerRunSweepParams((trainerNode.data ?? {}) as Record<string, unknown>)
    : {};
  const runLabel = trainerRunLabel(nodes, trainerNodeId);
  if (runLabel.trim()) trainerParams["trainer.run"] = runLabel.trim();
  const rawIn = (sweepSummary ?? "").trim();
  const structured = mergeSweepParamRecords(parseSweepParamsFromSummary(rawIn), sweepParams ?? {});
  let params = mergeSweepParamRecords(trainerParams, structured);
  if (streamSrc) {
    const streamLabel = readInstanceTitle(streamSrc.data, streamSrc.id);
    if (streamLabel.trim()) {
      params = mergeSweepParamRecords({ "stream.source": streamLabel.trim() }, params);
    }
  }
  const rawSweep = rawIn || formatSweepParamsSummary(params);
  return { rawSweep, params, paramsNumeric: coerceSweepParamsNumeric(params) };
}

function captureMetricsForTable(data: Partial<CurveSeriesTableNodeData>): string[] {
  const raw = data.captureMetrics;
  if (Array.isArray(raw) && raw.length > 0) {
    return raw.map((m) => String(m).trim()).filter(Boolean);
  }
  return [...DEFAULT_CURVE_SERIES_CAPTURE_METRICS];
}

/** Clear sweep / curve accumulation tables linked to a trainer (first train-series combo only). */
export function clearTrainerLinkedAccumulationTables(
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
): void {
  const sweepIds = nodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId, new Set(["sweep_data_table"]));
  const curveIds = nodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId, new Set(["curve_series_table"]));
  if (sweepIds.size === 0 && curveIds.size === 0) return;
  const runKey = trainerRunLabel(nodes, trainerNodeId);
  setNodes((prev) =>
    prev.map((n) => {
      if (sweepIds.has(n.id)) {
        const prevData = (n.data ?? {}) as SweepDataTableNodeData;
        return {
          ...n,
          data: {
            ...prevData,
            rows: [],
            selectedRowIds: null,
            paramKeyOrder: null,
          },
        };
      }
      if (curveIds.has(n.id)) {
        const prevData = (n.data ?? {}) as CurveSeriesTableNodeData;
        const multiStream = streamSourcesForTable(prev, edges, n.id).length > 1;
        const rows = multiStream
          ? (prevData.rows ?? []).filter((r) => (r.params?.["trainer.run"] ?? "") !== runKey)
          : [];
        return {
          ...n,
          data: {
            ...prevData,
            rows,
            selectedSeriesIds: multiStream ? prevData.selectedSeriesIds : null,
            paramKeyOrder: multiStream ? prevData.paramKeyOrder : null,
          },
        };
      }
      return n;
    }),
  );
}

/** Append captured curves to every curve_series_table in the trainer graph component. */
export function appendCurveSeriesOnTrainComplete(
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
  sweepSummary?: string,
  sweepParams?: Record<string, string>,
): void {
  const tableIds = nodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId, new Set(["curve_series_table"]));
  if (tableIds.size === 0) return;

  setNodes((prev) => {
    const nodesNow = prev;
    return prev.map((n) => {
      if (!tableIds.has(n.id)) return n;
      const prevData = (n.data ?? {}) as CurveSeriesTableNodeData;
      const captureMetrics = captureMetricsForTable(prevData);
      const sources = streamSourcesForTrainer(nodesNow, edges, n.id, trainerNodeId);
      if (sources.length === 0) return n;
      const nodeById = new Map(nodesNow.map((x) => [x.id, x]));

      let data = prevData;
      let appended = false;
      for (const src of sources) {
        const curves = extractFromStreamSource(src, captureMetrics, nodeById);
        if (curves.length === 0) continue;
        const { rawSweep, params, paramsNumeric } = curveSeriesParamsForRun(
          nodesNow,
          trainerNodeId,
          sweepSummary,
          sweepParams,
          src,
        );
        data = appendRowsToTable(data, curves, rawSweep, params, paramsNumeric);
        appended = true;
      }
      if (!appended) return n;

      return { ...n, data };
    });
  });
}
