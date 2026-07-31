import type { GraphDocument } from "../types/graph";

/** `large`: full JSON; `medium`: no checkpoint bytes; `small`: structure + settings only, no plot series / viz histories. */
export type GraphFileExportTier = "small" | "medium" | "large";

function cloneGraphDocument(doc: GraphDocument): GraphDocument {
  return JSON.parse(JSON.stringify(doc)) as GraphDocument;
}

function stripCheckpointFields(data: Record<string, unknown>, nodeType: string): void {
  if (nodeType === "model_checkpoint") {
    data.checkpoint_b64 = "";
    data.memoryCheckpoint_b64 = "";
  }
}

function stripVizRuntimeFields(data: Record<string, unknown>, nodeType: string): void {
  switch (nodeType) {
    case "trainer":
      delete data.lossHistory;
      delete data.testLossHistory;
      delete data.regLossHistory;
      delete data.stepTicks;
      delete data.observableMetricHistories;
      return;
    case "training_visualization":
      delete data.lossHistory;
      delete data.testLossHistory;
      delete data.regLossHistory;
      delete data.stepTicks;
      delete data.plotPngBase64;
      delete data.zoomXMin;
      delete data.zoomXMax;
      return;
    case "observable_viz":
      delete data.valueHistory;
      delete data.stepTicks;
      delete data.zoomXMin;
      delete data.zoomXMax;
      delete data.embeddingHistory;
      delete data.attentionMapFrames;
      return;
    case "tensor_viz_0d":
      delete data.stepTicks;
      delete data.valueHistory;
      return;
    case "deterministic_diffusion_sampler":
      delete data.runId;
      delete data.previewGrid;
      delete data.metadata;
      delete data.device;
      delete data.lastError;
      return;
    case "observable_paired_generation_similarity":
      delete data.meanMae;
      delete data.meanMse;
      delete data.meanLpips;
      delete data.mae;
      delete data.mse;
      delete data.histogramPng;
      delete data.imageGrid;
      delete data.device;
      delete data.lastError;
      return;
    case "observable_rp_score_sscd":
      delete data.rp;
      delete data.meanSimilarity;
      delete data.similarities;
      delete data.histogramPng;
      delete data.imageGrid;
      delete data.device;
      delete data.lastError;
      return;
    case "observable_nearest_train_gl":
      delete data.glScore;
      delete data.nearestSimilarity;
      delete data.nearestIndex;
      delete data.histogramPng;
      delete data.imageGrid;
      delete data.backend;
      delete data.device;
      delete data.lastError;
      return;
    case "observable_linear_interpolation_barrier":
      delete data.alphaSeries;
      delete data.trainLossAlongPath;
      delete data.testLossAlongPath;
      delete data.trainAccAlongPath;
      delete data.testAccAlongPath;
      delete data.lossBarrier;
      delete data.accuracyDrop;
      delete data.interpolationCurvePng;
      delete data.runSummary;
      delete data.lastError;
      return;
    case "observable_bezier_mode_connectivity":
      delete data.alphaSeries;
      delete data.linearTrainLoss;
      delete data.linearTestLoss;
      delete data.linearTrainAcc;
      delete data.linearTestAcc;
      delete data.bezierTrainLoss;
      delete data.bezierTestLoss;
      delete data.bezierTrainAcc;
      delete data.bezierTestAcc;
      delete data.linearLossBarrier;
      delete data.bezierLossBarrier;
      delete data.runSummary;
      delete data.lastError;
      return;
    default:
      return;
  }
}

/**
 * @param tier — `large`: unchanged; `medium`: drop torch checkpoint blobs; `small`: also drop plot / viz runtime series.
 */
export function applyGraphFileExportTier(doc: GraphDocument, tier: GraphFileExportTier): GraphDocument {
  if (tier === "large") {
    return doc;
  }
  const next = cloneGraphDocument(doc);
  for (const n of next.nodes) {
    const data = { ...(n.data as Record<string, unknown>) };
    stripCheckpointFields(data, n.type);
    if (tier === "small") {
      stripVizRuntimeFields(data, n.type);
    }
    n.data = data;
  }
  return next;
}
