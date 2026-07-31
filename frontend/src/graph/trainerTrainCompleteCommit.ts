import type { Edge, Node } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import { flushSync } from "react-dom";
import {
  applyCheckpointToModelCheckpointNodes,
  applyCrlCheckpointToCheckpointNodes,
  type TrainEdgeWire,
} from "./trainerCheckpointApply";
import { appendCurveSeriesOnTrainComplete } from "./curveSeriesAppend";
import { applyTrainerVizPayload, type TrainerStreamVizPayload } from "./trainerVizPayload";
import { refreshTensorViz0dStoredScalarsFromLiveResolve } from "./tensorViz0dSweepSeries";

type TrainCompletePayload = Pick<
  TrainerStreamVizPayload,
  | "loss_history"
  | "test_loss_history"
  | "step_ticks"
  | "epoch_ticks"
  | "plot_png_base64"
  | "visualization_node_ids"
  | "observable_viz_updates"
  | "observable_metric_histories"
  | "train_loop_seconds"
> & { checkpoint_b64?: string };

/**
 * Commit checkpoint + trainer/viz node data synchronously, then persist ``tensor_viz_0d`` scalars
 * using the same resolution path as the UI (including ``/api/activation_tensor`` / ``/api/dataset_tensor``
 * for lazy tensors). Templates saved as **medium/large** can embed stale ``valueHistory``; lazy metrics
 * never got a sync ``ok`` resolve during ``applyTrainerVizPayload`` — this pass fixes both for blog/sweep.
 */
export async function flushCheckpointApplyTrainerVizAndHydrateTv0d(args: {
  isCrl: boolean;
  setNodes: Dispatch<SetStateAction<Node[]>>;
  getNodes: () => Node[];
  getEdges: () => Edge[];
  trainerNodeId: string;
  wires: Edge[] | TrainEdgeWire[];
  payload: TrainCompletePayload;
  sweepSummary?: string;
  sweepParams?: Record<string, string>;
  /**
   * Optional delayed refresh windows for scalar capture after train completion.
   * This helps when upstream analysis nodes auto-recompute shortly after trainer payload commit.
   */
  extraRefreshDelaysMs?: number[];
  /** Keep ``trainUi`` on trainer node during multi-run train series between combos. */
  preserveTrainUi?: boolean;
  /** Skip tensor_viz_0d hydration (faster between sweep series runs). */
  skipTensorVizHydration?: boolean;
}): Promise<void> {
  const {
    isCrl,
    setNodes,
    getNodes,
    getEdges,
    trainerNodeId,
    wires,
    payload,
    sweepSummary,
    sweepParams,
    extraRefreshDelaysMs,
    preserveTrainUi,
    skipTensorVizHydration,
  } = args;
  flushSync(() => {
    if (payload.checkpoint_b64) {
      if (isCrl) {
        applyCrlCheckpointToCheckpointNodes(setNodes, trainerNodeId, payload.checkpoint_b64, wires);
      } else {
        applyCheckpointToModelCheckpointNodes(setNodes, trainerNodeId, payload.checkpoint_b64, wires);
      }
    }
    applyTrainerVizPayload(setNodes, payload, trainerNodeId, sweepSummary, getEdges(), sweepParams, preserveTrainUi);
    appendCurveSeriesOnTrainComplete(
      setNodes,
      getNodes(),
      getEdges(),
      trainerNodeId,
      sweepSummary,
      sweepParams,
    );
  });
  if (skipTensorVizHydration) return;
  await refreshTensorViz0dStoredScalarsFromLiveResolve(
    getNodes,
    getEdges,
    setNodes,
    trainerNodeId,
    payload.step_ticks,
  );
  for (const ms of extraRefreshDelaysMs ?? []) {
    const delay = Number(ms);
    if (!Number.isFinite(delay) || delay <= 0) continue;
    await new Promise<void>((resolve) => {
      window.setTimeout(resolve, delay);
    });
    await refreshTensorViz0dStoredScalarsFromLiveResolve(
      getNodes,
      getEdges,
      setNodes,
      trainerNodeId,
      payload.step_ticks,
    );
  }
}
