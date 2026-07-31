import type { ListOr1 } from "./multiValueUtils";
import type { TrainSeriesAssignment } from "../../graph/trainSeriesPlan";

/** Ephemeral UI: canvas host (e.g. graph assist) streams NDJSON progress into the trainer node. */
export type TrainerHostTrainUi = {
  active: boolean;
  progressPct: number;
  seriesBarPct: number;
  seriesDual: boolean;
  captionLines: string[] | null;
};

/** Ephemeral UI: trainer node's own Train button progress — survives canvas/tab switches. */
export type TrainerTrainUi = {
  active: boolean;
  loading: boolean;
  paused: boolean;
  progressPct: number;
  seriesBarPct: number;
  seriesDual: boolean;
  captionLines: string[] | null;
  phaseText: string | null;
  error: string | null;
};

export function emptyTrainerTrainUi(): TrainerTrainUi {
  return {
    active: true,
    loading: false,
    paused: false,
    progressPct: 0,
    seriesBarPct: 0,
    seriesDual: false,
    captionLines: null,
    phaseText: null,
    error: null,
  };
}

/** Paused train stream payload — persisted on trainer node so Continue works after remount. */
export type TrainerPauseCheckpoint = {
  type: "paused";
  next_step: number;
  checkpoint_b64: string;
  loss_history: number[];
  test_loss_history?: number[];
  reg_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  plot_png_base64: string;
  visualization_node_ids: string[];
  observable_viz_updates?: {
    node_id: string;
    paired_observable_id?: string;
    value_history?: number[];
    test_value_history?: number[];
    value_histories?: number[][];
    series_labels?: string[];
    embedding_history?: number[][][];
    attention_map_frames?: import("./attentionMapVizDefaults").AttentionMapFrame[];
  }[];
  observable_metric_histories: Record<string, number[]>;
  observable_embedding_histories?: Record<string, number[][][]>;
  observable_attention_slice_histories?: Record<string, import("./attentionMapVizDefaults").AttentionMapFrame[]>;
  train_loop_seconds?: number;
};

export type TrainingLengthMode = "steps" | "epochs";
export type TrainerLogSchedule = "fixed_interval" | "idnns_logspace";
export type TrainerLogAggregation = "last_batch" | "interval_sample_mean";
export type TrainerLogTiming = "post_update" | "pre_update";
export type TrainerTestEvaluation = "log_ticks" | "final_only" | "disabled";
export type TrainerMinibatchSampling =
  | "independent_step"
  | "epoch_shuffle"
  | "affine_epoch";

export type TrainerNodeData = {
  /** When `epochs`, `trainingSteps` is derived at train time from dataset size and batch. */
  trainingLengthMode?: TrainingLengthMode;
  /** Used when `trainingLengthMode` is `epochs`. */
  trainingEpochs?: number;
  trainingSteps: number;
  logFrequency: number;
  /** Fixed step interval or the released IDNNs base-2 logspace epoch schedule. */
  logSchedule?: TrainerLogSchedule;
  /** Requested samples before duplicate logspace epoch indices are removed. */
  logSamples?: number;
  /** Current log-tick batch or sample-weighted mean since the previous tick. */
  logAggregation?: TrainerLogAggregation;
  /** Whether fixed log ticks are sampled before or after the matching update. */
  logTiming?: TrainerLogTiming;
  /** Test-set evaluation cadence. */
  testEvaluation?: TrainerTestEvaluation;
  /** -1 derives training randomness from the dataset seed. */
  trainSeed?: number;
  /** PyTorch device: cpu (default), auto, cuda, cuda:N, mps. */
  computeDevice: string;
  /** When true with cuda, train via AutoDL remote SSH; false forces local CUDA. */
  remoteGpu?: boolean;
  /** -1 = full-batch gradient on current train tensors; otherwise SGD mini-batch size. */
  batchSize: ListOr1<number>;
  /** Independent draws or a no-replacement order per epoch. */
  minibatchSampling?: TrainerMinibatchSampling;
  /** -1 derives the minibatch stream from the training seed. */
  minibatchSeed?: number;
  /** 0 = disabled (legacy behavior). */
  gradClipMaxNorm: number;
  /** Skip wired observables during training except accuracy (loss is always logged). */
  disableExtraObservables: boolean;
  /** When true, stream incremental metrics events during training for live viz updates. */
  liveVizUpdates?: boolean;
  /** Last training run — used when wiring loss/observable handles to tensor selector or tensor viz. */
  lossHistory?: number[];
  testLossHistory?: number[];
  regLossHistory?: number[];
  stepTicks?: number[];
  /** Keys are observable node ids (same as training backend). */
  observableMetricHistories?: Record<string, number[]>;
  /**
   * Last full train checkpoint (model+optimizer), mirrored from the stream for wiring that skips
   * `model_checkpoint` (e.g. activation `model` ← trainer `checkpoint`).
   */
  memoryCheckpoint_b64?: string;
  hostTrainUi?: TrainerHostTrainUi;
  /** Own Train-button progress; persisted on node data so canvas switches do not reset the bar. */
  trainUi?: TrainerTrainUi;
  /** Paused checkpoint for Continue after project/canvas remount. */
  trainPauseCheckpoint?: TrainerPauseCheckpoint;
  /** Last complete-run checkpoint for Continue training (extend training_steps and resume). */
  trainCompleteCheckpoint?: TrainerPauseCheckpoint;
  /** Series sweep context when paused mid-series (Continue resumes correct combo). */
  trainSeriesPauseCtx?: { totalRuns: number; runIndex: number };
  /** User-defined target curve for hyperparameter tuning. */
  targetCurveStepTicks?: number[];
  targetCurveLossHistory?: number[];
  /** Wall time for zero_grad → forward → backward → step only (last completed or paused run). */
  lastTrainLoopSeconds?: number;
  /** Last coordinate-descent summary string shown in UI. */
  lastAutoTuneSummary?: string;
  /** Last auto-tune comparison (curves + ranked params); shown in results modal. */
  autoTuneComparisonResult?: AutoTuneComparisonResult;
};

/** One ranked candidate curve from coordinate descent (best + up to 3 runners-up). */
export type AutoTuneRankedCurve = {
  rank: number;
  score: number;
  finalAbsError?: number;
  smoothnessPenalty?: number;
  params: Record<string, unknown>;
  stepTicks: number[];
  lossHistory: number[];
};

/** Persisted snapshot for the auto-tune comparison chart. */
export type AutoTuneComparisonResult = {
  baselineStepTicks: number[];
  baselineLossHistory: number[];
  targetStepTicks: number[];
  targetLossHistory: number[];
  ranked: AutoTuneRankedCurve[];
  bestScore: number;
};

export function defaultTrainerData(): TrainerNodeData {
  return {
    trainingLengthMode: "steps",
    trainingEpochs: 100,
    trainingSteps: 1000,
    logFrequency: 10,
    logSchedule: "fixed_interval",
    logSamples: 1800,
    logAggregation: "last_batch",
    logTiming: "post_update",
    testEvaluation: "log_ticks",
    trainSeed: -1,
    computeDevice: "cpu",
    batchSize: -1,
    minibatchSampling: "independent_step",
    minibatchSeed: -1,
    gradClipMaxNorm: 0,
    disableExtraObservables: false,
    liveVizUpdates: true,
  };
}
