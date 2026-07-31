import type { Edge, Node } from "@xyflow/react";
import type { CurvePoint } from "./observableCurvePayload";
import { serializeNodeForTrain } from "../graph/trainSeriesPlan";
import { readNdjsonTrainStream } from "../graph/readNdjsonTrainStream";
import { applySpeedUpTrick } from "./applySpeedUpTrick";
import { analyzeTargetPhase, measuredCurvePoints } from "./targetPhaseTransition";
import { evaluateSpeedUpTrick } from "./evaluateSpeedUpTrick";
import type { CurveStarerTargetConfig, SpeedUpTrickProposal, TrickTestResult } from "./speedUpTrickTypes";
import type { CurveStarerAnalyzedEntry } from "./lpdTypes";

function entryPointsFromTrainComplete(
  entry: CurveStarerAnalyzedEntry,
  stepTicks: number[],
  complete: {
    observable_viz_updates?: {
      node_id: string;
      value_history?: number[];
      test_value_history?: number[];
      value_histories?: number[][];
    }[];
    observable_metric_histories?: Record<string, number[]>;
  },
): CurvePoint[] {
  const [nodeId, seriesId] = entry.entryId.split(":");
  const useTest = (seriesId ?? "").toLowerCase() === "test";

  for (const upd of complete.observable_viz_updates ?? []) {
    if (upd.node_id !== nodeId) continue;
    const ys = useTest
      ? (upd.test_value_history ?? [])
      : (upd.value_history ?? upd.value_histories?.[0] ?? []);
    const len = Math.min(stepTicks.length, ys.length);
    const points: CurvePoint[] = [];
    for (let i = 0; i < len; i++) {
      const t = stepTicks[i]!;
      const loss = ys[i]!;
      if (Number.isFinite(t) && Number.isFinite(loss)) points.push({ t, loss });
    }
    if (points.length >= 5) return points;
  }

  const hist = complete.observable_metric_histories ?? {};
  for (const [key, ys] of Object.entries(hist)) {
    if (!key.startsWith(nodeId ?? "")) continue;
    if (useTest && !key.includes("::test")) continue;
    if (!useTest && key.includes("::test")) continue;
    const len = Math.min(stepTicks.length, ys.length);
    const points: CurvePoint[] = [];
    for (let i = 0; i < len; i++) {
      points.push({ t: stepTicks[i]!, loss: ys[i]! });
    }
    if (points.length >= 5) return points;
  }

  return entry.points;
}

function baselineCurveFromEntry(entry: CurveStarerAnalyzedEntry): CurvePoint[] {
  return measuredCurvePoints(entry);
}

export async function runSpeedUpTrickTest(args: {
  nodes: Node[];
  edges: Edge[];
  proposal: SpeedUpTrickProposal;
  targetConfig: CurveStarerTargetConfig;
  baselineEntry: CurveStarerAnalyzedEntry;
  baselineAnalysis: ReturnType<typeof analyzeTargetPhase>;
  signal?: AbortSignal;
  onProgress?: (pct: number, step: number, total: number) => void;
  /** Skip wired observables during training (loss + accuracy only). Default true for trick tests. */
  skipExtraObservables?: boolean;
}): Promise<TrickTestResult> {
  const {
    nodes,
    edges,
    proposal,
    targetConfig,
    baselineEntry,
    baselineAnalysis,
    signal,
    onProgress,
    skipExtraObservables = true,
  } = args;
  const baselineCurve = baselineCurveFromEntry(baselineEntry);
  const trainers = nodes.filter((n) => n.type === "trainer" || n.type === "crl_trainer");
  if (trainers.length === 0) {
    return {
      verdict: "error",
      message: "No trainer on canvas.",
      baseline: baselineAnalysis,
      trick: baselineAnalysis,
      baselineCurve,
      trickCurve: [],
    };
  }
  const trainerId = trainers[0]!.id;
  const hasHessian = nodes.some((n) => n.type === "observable_hessian_eigenvalues");
  let applied = applySpeedUpTrick(nodes, edges, proposal, trainerId);
  if (skipExtraObservables) {
    applied = {
      ...applied,
      nodes: applied.nodes.map((n) =>
        n.id === trainerId && n.type === "trainer"
          ? { ...n, data: { ...(n.data as object), disableExtraObservables: true } }
          : n,
      ),
    };
  }

  const body: Record<string, unknown> = {
    trainer_node_id: trainerId,
    nodes: applied.nodes.map(serializeNodeForTrain),
    edges: applied.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    })),
  };
  if (hasHessian) body.hessian_oversized_policy = "skip";

  const res = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    return {
      verdict: "error",
      message: msg,
      baseline: baselineAnalysis,
      trick: baselineAnalysis,
      baselineCurve,
      trickCurve: [],
    };
  }

  const reader = res.body.getReader();
  const { complete, aborted } = await readNdjsonTrainStream(reader, (raw) => {
    if (raw.type !== "progress") return;
    const total = Math.max(1, raw.total);
    const step = Math.min(Math.max(0, raw.step), total);
    onProgress?.(Math.min(100, Math.round((step / total) * 100)), step, total);
  });
  if (aborted || !complete) {
    return {
      verdict: "error",
      message: aborted ? "Training aborted." : "Training did not complete.",
      baseline: baselineAnalysis,
      trick: baselineAnalysis,
      baselineCurve,
      trickCurve: [],
    };
  }

  const trickCurve = entryPointsFromTrainComplete(baselineEntry, complete.step_ticks, complete);
  const trickEntry: CurveStarerAnalyzedEntry = { ...baselineEntry, points: trickCurve, lpd: null };
  const trickAnalysis = analyzeTargetPhase(
    trickEntry,
    targetConfig.objective,
    targetConfig.threshold,
  );
  return {
    ...evaluateSpeedUpTrick(baselineAnalysis, trickAnalysis),
    baselineCurve,
    trickCurve,
  };
}
