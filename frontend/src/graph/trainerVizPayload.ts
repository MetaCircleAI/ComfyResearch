import type { Edge, Node } from "@xyflow/react";
import {
  patchTensorViz0dHistoriesFromTrainerStepTicks,
  tensorViz0dNodeIdsInTrainerGraphComponent,
} from "./tensorViz0dSweepSeries";
import type { CrlTrainerNodeData } from "../components/nodes/crlTrainerDefaults";
import { defaultCrlTrainerData } from "../components/nodes/crlTrainerDefaults";
import type { TrainerNodeData } from "../components/nodes/trainerDefaults";
import { defaultTrainerData } from "../components/nodes/trainerDefaults";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";

/** NDJSON ``complete`` / ``paused`` payload fields used to refresh trainer + viz nodes. */
export type TrainerStreamVizPayload = {
  loss_history: number[];
  test_loss_history?: number[];
  reg_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  plot_png_base64: string;
  visualization_node_ids: string[];
  observable_viz_updates?: {
    node_id: string;
    /** Same as the viz node’s ``pairedObservableId``; used if ``node_id`` does not match the canvas node. */
    paired_observable_id?: string;
    value_history?: number[];
    /** Accuracy observable: test split, same length as ``value_history``. */
    test_value_history?: number[];
    value_histories?: number[][];
    /** Gradient norm / activation stats multi-series legends (matches ``value_histories`` rows). */
    series_labels?: string[];
    embedding_history?: number[][][];
    attention_map_frames?: import("../components/nodes/attentionMapVizDefaults").AttentionMapFrame[];
  }[];
  observable_metric_histories?: Record<string, number[]>;
  /** Seconds in the inner train step only (excludes batch materialization, logging, I/O). */
  train_loop_seconds?: number;
};

const SWEEP_SERIES_INTER_RUN_PAUSE_MS = 500;

/** Pause after each multi-run series experiment so the user can see viz results before the next run. */
export async function sleepBetweenSweepSeriesRuns(signal?: AbortSignal): Promise<void> {
  const ms =
    typeof document !== "undefined" && document.hidden ? 0 : SWEEP_SERIES_INTER_RUN_PAUSE_MS;
  if (ms <= 0) return;
  await new Promise<void>((resolve) => {
    const id = window.setTimeout(resolve, ms);
    if (!signal) return;
    if (signal.aborted) {
      window.clearTimeout(id);
      resolve();
      return;
    }
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(id);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * Clear loss / observable chart data on viz nodes wired from this trainer before the next
 * series / sweep train request (avoids showing the previous base setup or prior experiment).
 */
export function clearTrainerLinkedVizForSeriesRun(
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
  nodes: Node[],
  edges: Edge[],
  trainerNodeId: string,
): void {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const tvIds = new Set<string>();
  const obsVizIds = new Set<string>();
  for (const e of edges) {
    if (e.source !== trainerNodeId) continue;
    const tgt = nodeById.get(e.target);
    if (!tgt) continue;
    if (tgt.type === "training_visualization") {
      tvIds.add(e.target);
      continue;
    }
    if (tgt.type === "observable_viz" || tgt.type === "observable_viz_neuron_trajectory_2d") {
      const sh = e.sourceHandle ?? null;
      if (sh === null || sh === "" || sh === "observable_results") {
        obsVizIds.add(e.target);
      }
    }
  }
  const tv0dIds = tensorViz0dNodeIdsInTrainerGraphComponent(nodes, edges, trainerNodeId);

  if (tvIds.size === 0 && obsVizIds.size === 0 && tv0dIds.size === 0) return;
  setNodes((prev) =>
    prev.map((n) => {
      if (tvIds.has(n.id)) {
        const prevData = (n.data ?? {}) as TrainingVisualizationNodeData;
        return {
          ...n,
          data: {
            ...prevData,
            lossHistory: [],
            testLossHistory: [],
            regLossHistory: [],
            stepTicks: [],
            plotPngBase64: "",
            lastSweepSummary: undefined,
          },
        };
      }
      if (obsVizIds.has(n.id)) {
        const prevData = (n.data ?? {}) as Record<string, unknown>;
        return {
          ...n,
          data: {
            ...prevData,
            valueHistory: [],
            testValueHistory: [],
            valueHistories: [],
            seriesLabels: [],
            embeddingHistory: [],
            attentionMapFrames: [],
            stepTicks: [],
            lastSweepSummary: undefined,
          },
        };
      }
      if (tv0dIds.has(n.id)) {
        const prevData = (n.data ?? {}) as Record<string, unknown>;
        return {
          ...n,
          data: {
            ...prevData,
            stepTicks: [],
            valueHistory: [],
          },
        };
      }
      return n;
    }),
  );
}

/**
 * Apply streamed training results to the trainer node, ``training_visualization`` targets,
 * and observable viz nodes (same logic as ``TrainerNode``).
 *
 * Also patches downstream ``tensor_viz_0d`` values from ``resolveUpstreamTensor``; call
 * ``applyCheckpointToModelCheckpointNodes`` / ``applyCrlCheckpointToCheckpointNodes`` **first**
 * when ``complete.checkpoint_b64`` is present so weight-derived scalars reflect the finished run.
 */
export function applyTrainerVizPayload(
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
  payload: Pick<
    TrainerStreamVizPayload,
    | "loss_history"
    | "test_loss_history"
    | "reg_loss_history"
    | "step_ticks"
    | "epoch_ticks"
    | "plot_png_base64"
    | "visualization_node_ids"
    | "observable_viz_updates"
    | "observable_metric_histories"
    | "train_loop_seconds"
  >,
  trainerNodeId: string,
  sweepSummary?: string,
  edges?: Edge[],
  sweepParams?: Record<string, string>,
  preserveTrainUi?: boolean,
) {
  const sweepPatch =
    sweepSummary !== undefined || sweepParams !== undefined
      ? {
          ...(sweepSummary !== undefined ? { lastSweepSummary: sweepSummary } : {}),
          ...(sweepParams !== undefined ? { lastSweepParams: sweepParams } : {}),
        }
      : null;
  const vizSet = new Set(payload.visualization_node_ids);
  for (const edge of edges ?? []) {
    if (
      edge.source === trainerNodeId &&
      edge.sourceHandle === "loss_results" &&
      (edge.targetHandle === "tensor_list" || edge.targetHandle === "tensor")
    ) {
      vizSet.add(edge.target);
    }
  }
  const rawObsUpdates = payload.observable_viz_updates ?? [];
  const obsById = new Map<string, (typeof rawObsUpdates)[number]>();
  for (const u of rawObsUpdates) {
    obsById.set(u.node_id, u);
    const pid = u.paired_observable_id;
    if (typeof pid === "string" && pid.trim()) {
      obsById.set(pid.trim(), u);
    }
  }
  const omh = payload.observable_metric_histories ?? {};
  setNodes((prev) => {
    const fallbackObsByVizId = new Map<string, (typeof rawObsUpdates)[number]>();
    const nodeById = new Map(prev.map((node) => [node.id, node]));
    for (const edge of edges ?? []) {
      if (edge.source !== trainerNodeId) continue;
      const viz = nodeById.get(edge.target);
      if (viz?.type !== "observable_viz") continue;
      const pairedId = (viz.data as { pairedObservableId?: unknown } | undefined)?.pairedObservableId;
      if (typeof pairedId !== "string" || !pairedId.trim()) continue;
      const history = omh[pairedId.trim()];
      if (!Array.isArray(history) || history.length === 0) continue;
      fallbackObsByVizId.set(edge.target, {
        node_id: edge.target,
        paired_observable_id: pairedId.trim(),
        value_history: history,
      });
    }
    const next = prev.map((n) => {
      if (n.id === trainerNodeId && n.type === "trainer") {
        const prevData = (n.data ?? {}) as Partial<TrainerNodeData>;
        const def = defaultTrainerData();
        return {
          ...n,
          data: {
            ...def,
            ...prevData,
            lossHistory: payload.loss_history,
            testLossHistory: payload.test_loss_history ?? [],
            regLossHistory: payload.reg_loss_history ?? [],
            stepTicks: payload.step_ticks,
            epochTicks: payload.epoch_ticks ?? [],
            observableMetricHistories: omh,
            hostTrainUi: preserveTrainUi ? prevData.hostTrainUi : undefined,
            trainUi: preserveTrainUi ? prevData.trainUi : undefined,
            ...(typeof payload.train_loop_seconds === "number" && Number.isFinite(payload.train_loop_seconds)
              ? { lastTrainLoopSeconds: payload.train_loop_seconds }
              : {}),
          },
        };
      }
      if (n.id === trainerNodeId && n.type === "crl_trainer") {
        const prevData = (n.data ?? {}) as Partial<CrlTrainerNodeData>;
        const def = defaultCrlTrainerData();
        return {
          ...n,
          data: {
            ...prevData,
            trainingSteps: prevData.trainingSteps ?? def.trainingSteps,
            logFrequency: prevData.logFrequency ?? def.logFrequency,
            lossHistory: payload.loss_history,
            testLossHistory: payload.test_loss_history ?? [],
            regLossHistory: payload.reg_loss_history ?? [],
            stepTicks: payload.step_ticks,
            epochTicks: payload.epoch_ticks ?? [],
            observableMetricHistories: omh,
          },
        };
      }
      if (vizSet.has(n.id) && n.type === "training_visualization") {
        const prevData = (n.data ?? {}) as TrainingVisualizationNodeData;
        return {
          ...n,
          data: {
            ...prevData,
            lossHistory: payload.loss_history,
            testLossHistory: payload.test_loss_history,
            regLossHistory: payload.reg_loss_history ?? [],
            stepTicks: payload.step_ticks,
            epochTicks: payload.epoch_ticks ?? [],
            plotPngBase64: payload.plot_png_base64,
            ...(sweepPatch ?? {}),
          },
        };
      }
      const isObsVizType = n.type === "observable_viz" || n.type === "observable_viz_neuron_trajectory_2d";
      let obsPayload = obsById.get(n.id);
      if (obsPayload === undefined && isObsVizType) {
        const pd = (n.data ?? {}) as { pairedObservableId?: string };
        const poid = typeof pd.pairedObservableId === "string" ? pd.pairedObservableId.trim() : "";
        if (poid) obsPayload = obsById.get(poid);
      }
      if (obsPayload === undefined && isObsVizType) {
        obsPayload = fallbackObsByVizId.get(n.id);
      }
      if (obsPayload !== undefined && isObsVizType) {
        const prevData = (n.data ?? {}) as Record<string, unknown>;
        const patch: Record<string, unknown> = {};
        if (Array.isArray(obsPayload.value_history)) {
          patch.valueHistory = obsPayload.value_history;
        }
        if (Array.isArray(obsPayload.test_value_history)) {
          patch.testValueHistory = obsPayload.test_value_history;
        }
        if (Array.isArray(obsPayload.value_histories)) {
          patch.valueHistories = obsPayload.value_histories;
          if (Array.isArray(obsPayload.series_labels)) {
            patch.seriesLabels = obsPayload.series_labels;
          }
        } else if (Array.isArray(obsPayload.value_history)) {
          // Backend often sends a single series as ``value_history`` (e.g. weight σ_max all-layers fallback).
          // Hessian-style chart reads ``valueHistories`` only — wrap one row so the viz is not cleared.
          patch.valueHistories = [
            obsPayload.value_history.map((x) => {
              const n = Number(x);
              return Number.isFinite(n) ? n : Number.NaN;
            }),
          ];
          patch.seriesLabels =
            Array.isArray(obsPayload.series_labels) && obsPayload.series_labels.length > 0
              ? obsPayload.series_labels
              : ["global"];
        }
        if (Array.isArray(obsPayload.embedding_history)) {
          patch.embeddingHistory = obsPayload.embedding_history;
        }
        if (Array.isArray(obsPayload.attention_map_frames)) {
          patch.attentionMapFrames = obsPayload.attention_map_frames;
        }
        return {
          ...n,
          data: {
            ...prevData,
            ...patch,
            stepTicks: payload.step_ticks,
            epochTicks: payload.epoch_ticks ?? [],
            ...(sweepPatch ?? {}),
          },
        };
      }
      return n;
    });
    if (!edges?.length) return next;
    return patchTensorViz0dHistoriesFromTrainerStepTicks(next, edges, trainerNodeId, payload.step_ticks);
  });
}
