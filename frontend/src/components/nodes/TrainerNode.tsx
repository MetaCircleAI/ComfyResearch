import { Handle, Position, useReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { buildSweepParamsFromCombo, buildTrainerRunSweepParams, formatSweepParamsSummary } from "../../graph/sweepParamExtract";
import {
  applyAssignmentsToNodes,
  formatTrainSeriesComboCaption,
  formatTrainSeriesSweptLines,
  getSweptAxisIdSet,
  planTrainSeriesAssignments,
  serializeExecutionGraphForTarget,
  type SerializedTrainEdge,
  type TrainSeriesAssignment,
} from "../../graph/trainSeriesPlan";
import {
  applyAssignmentsAndResolveTrainingLength,
  normalizeTrainingLengthMode,
  previewTrainingStepsForTrainer,
  resolveDatasetTrainSizeForTrainer,
  resolveTrainingLengthOnNodes,
} from "../../graph/trainingLengthResolve";
import { useFlowSurface } from "../../context/FlowSurfaceContext";
import { useResearchGraph } from "../../context/ResearchGraphContext";
import { readNdjsonTrainStream, type TrainStreamComplete } from "../../graph/readNdjsonTrainStream";
import {
  getTrainerRunSession,
  hasTrainerRunSession,
  registerTrainerRunSession,
  unregisterTrainerRunSession,
} from "../../graph/trainerRunSession";
import { patchTrainerTrainUi, resolveTrainerTrainDisplay } from "../../graph/trainerTrainUiResolve";
import { flushCheckpointApplyTrainerVizAndHydrateTv0d } from "../../graph/trainerTrainCompleteCommit";
import { createTrainProgressUiThrottler } from "../../graph/throttleTrainProgressPersist";
import { applyTrainerVizPayload, clearTrainerLinkedVizForSeriesRun, sleepBetweenSweepSeriesRuns } from "../../graph/trainerVizPayload";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { ComfyIntListField } from "./comfyMultiFields";
import { intChoices, packIntList } from "./multiValueUtils";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultTrainerData,
  type TrainerLogAggregation,
  type TrainerLogSchedule,
  type TrainerLogTiming,
  type TrainerMinibatchSampling,
  type TrainerNodeData,
  type TrainerPauseCheckpoint,
  type TrainerTestEvaluation,
  type TrainingLengthMode,
} from "./trainerDefaults";
import {
  COMPUTE_MODE_OPTIONS,
  computeDeviceFromModeUi,
  computeModeUiFromDevice,
  defaultLocalCudaDevice,
  localCudaDeviceFromIndex,
  localCudaGpuSelectOptions,
  localCudaIndexFromDevice,
  normalizeComputeModeUi,
  remoteGpuFromModeUi,
  type LocalCudaDeviceInfo,
} from "./trainerComputeDevice";

function isHessianOversizedDetail(
  d: unknown,
): d is { code: string; n_params: number; limit: number; message?: string } {
  return typeof d === "object" && d !== null && (d as { code?: string }).code === "hessian_oversized";
}

function formatTrainLoopSeconds(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${s.toFixed(2)} s`;
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}m ${r.toFixed(1)}s`;
}

function progressPctFromStep(nextStep: number, totalSteps: number): number {
  const total = Math.max(1, totalSteps);
  const step = Math.min(Math.max(0, nextStep), total);
  return Math.min(100, Math.round((step / total) * 100));
}

function patchTrainerData(
  id: string,
  patch: Partial<TrainerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTrainerData();
      const cur = (n.data ?? {}) as Partial<TrainerNodeData>;
      const prev = { ...def, ...cur } as TrainerNodeData;
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

type TrainerSocketId =
  | "dataset"
  | "model"
  | "optimizer"
  | "loss"
  | "observables"
  | "batch_schedule";
type TrainerOutputId = "checkpoint" | "loss_results" | "observable_results";

type TrainStreamProgress = { type: "progress"; step: number; total: number };
type TrainStreamPaused = {
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
    /** Hessian observable: one series per eigenvalue rank. */
    value_histories?: number[][];
    series_labels?: string[];
    embedding_history?: number[][][];
  }[];
  observable_metric_histories: Record<string, number[]>;
  observable_embedding_histories?: Record<string, number[][][]>;
  train_loop_seconds?: number;
};
type TrainResumeBody = {
  next_step: number;
  checkpoint_b64: string;
  loss_history: number[];
  test_loss_history: number[];
  reg_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  observable_metric_histories: Record<string, number[]>;
  observable_embedding_histories?: Record<string, number[][][]>;
};

type RemoteTrainConfig = {
  host: string;
  user: string;
  remote_path: string;
  python: string;
  identity: string;
  password: string;
  extra_opts: string;
  enabled: boolean;
  upload_dataset: boolean;
};

type RemoteTrainStatus = {
  mode: "local" | "remote";
  remoteHost?: string;
  source?: string;
  lastValidationOk?: boolean | null;
  lastValidationError?: string;
};

const DEFAULT_REMOTE_PYTHON = "/root/miniconda3/bin/python3";

const DEFAULT_REMOTE_TRAIN_CONFIG: RemoteTrainConfig = {
  host: "",
  user: "ubuntu",
  remote_path: "",
  python: DEFAULT_REMOTE_PYTHON,
  identity: "",
  password: "",
  extra_opts: "",
  enabled: false,
  upload_dataset: false,
};

function _unquoteToken(s: string): string {
  if (s.length >= 2) {
    const first = s[0];
    const last = s[s.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return s.slice(1, -1);
    }
  }
  return s;
}

function parseSshCommandToRemoteConfig(raw: string): Partial<RemoteTrainConfig> | null {
  const text = raw.trim();
  if (!text) return null;
  if (!text.startsWith("ssh ")) return null;
  const tokens = text.match(/"[^"]*"|'[^']*'|\S+/g)?.map(_unquoteToken) ?? [];
  if (tokens.length < 2 || tokens[0] !== "ssh") return null;

  const extra: string[] = [];
  let identity = "";
  let user = "";
  let host = "";

  for (let i = 1; i < tokens.length; i++) {
    const tok = tokens[i]!;
    if (tok === "-i") {
      const val = tokens[i + 1];
      if (val) {
        identity = val;
        i++;
      }
      continue;
    }
    if (tok.startsWith("-i") && tok.length > 2) {
      identity = tok.slice(2);
      continue;
    }
    if (tok === "-p" || tok === "-o") {
      const val = tokens[i + 1];
      if (val) {
        extra.push(tok, val);
        i++;
      } else {
        extra.push(tok);
      }
      continue;
    }
    if (tok.startsWith("-")) {
      extra.push(tok);
      continue;
    }
    if (!host) {
      const at = tok.indexOf("@");
      if (at > 0) {
        user = tok.slice(0, at);
        host = tok.slice(at + 1);
      } else {
        host = tok;
      }
      continue;
    }
    break;
  }

  if (!host) return null;
  return {
    host,
    user: user || "ubuntu",
    identity,
    extra_opts: extra.join(" "),
  };
}

function normalizeRemotePythonValue(raw: string | undefined | null): string {
  const s = String(raw ?? "").trim();
  if (!s) return DEFAULT_REMOTE_PYTHON;
  if (s === "python3") return DEFAULT_REMOTE_PYTHON;
  return s;
}

function buildSshCommandFromRemoteConfig(cfg: RemoteTrainConfig): string {
  const parts: string[] = ["ssh"];
  if (cfg.identity.trim()) {
    parts.push("-i", cfg.identity.trim());
  }
  if (cfg.extra_opts.trim()) {
    parts.push(cfg.extra_opts.trim());
  }
  const host = cfg.host.trim();
  const user = cfg.user.trim() || "ubuntu";
  if (host) {
    parts.push(`${user}@${host}`);
  }
  return parts.join(" ").trim();
}

function buildResumeBody(p: TrainStreamPaused): TrainResumeBody {
  return {
    next_step: p.next_step,
    checkpoint_b64: p.checkpoint_b64,
    loss_history: p.loss_history,
    test_loss_history: p.test_loss_history ?? [],
    reg_loss_history: p.reg_loss_history ?? [],
    step_ticks: p.step_ticks,
    epoch_ticks: p.epoch_ticks ?? [],
    observable_metric_histories: p.observable_metric_histories,
    observable_embedding_histories: p.observable_embedding_histories ?? {},
  };
}

function buildCompleteCheckpoint(c: TrainStreamComplete): TrainerPauseCheckpoint | null {
  if (!c.checkpoint_b64) return null;
  const nextStep = c.step_ticks.length ? c.step_ticks[c.step_ticks.length - 1]! : 0;
  return {
    type: "paused",
    next_step: nextStep,
    checkpoint_b64: c.checkpoint_b64,
    loss_history: c.loss_history,
    test_loss_history: c.test_loss_history ?? [],
    reg_loss_history: c.reg_loss_history ?? [],
    step_ticks: c.step_ticks,
    epoch_ticks: c.epoch_ticks ?? [],
    plot_png_base64: c.plot_png_base64,
    visualization_node_ids: c.visualization_node_ids,
    observable_viz_updates: c.observable_viz_updates,
    observable_metric_histories: c.observable_metric_histories ?? {},
    observable_embedding_histories: c.observable_embedding_histories,
    observable_attention_slice_histories: c.observable_attention_slice_histories,
    train_loop_seconds: c.train_loop_seconds,
  };
}

function TrainerInWrap({
  handleId,
  label,
  fullWidth,
}: {
  handleId: TrainerSocketId;
  label: string;
  fullWidth?: boolean;
}) {
  return (
    <div
      className={`cr-trainer-io-row__leftwrap${fullWidth ? " cr-trainer-io-row__leftwrap--full" : ""}`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id={handleId}
        className={`cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--${handleId}`}
      />
      <span className="cr-trainer-socket-label">{label}</span>
    </div>
  );
}

type TrainerOutAccent = "model" | "loss" | "observable";

function TrainerOutWrap({
  handleId,
  label,
  accent,
}: {
  handleId: TrainerOutputId;
  label: string;
  accent: TrainerOutAccent;
}) {
  return (
    <div className="cr-trainer-io-row__rightwrap">
      <span className="cr-trainer-output-label">{label}</span>
      <Handle
        type="source"
        position={Position.Right}
        id={handleId}
        className={`cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--${accent}`}
      />
    </div>
  );
}

const TRAINING_LENGTH_MODE_OPTIONS: { value: TrainingLengthMode; label: string }[] = [
  { value: "steps", label: "steps" },
  { value: "epochs", label: "epochs" },
];
const TRAINER_LOG_SCHEDULE_OPTIONS: { id: TrainerLogSchedule; label: string }[] = [
  { id: "fixed_interval", label: "Fixed step interval" },
  { id: "idnns_logspace", label: "IDNNs log-space epochs" },
];
const TRAINER_LOG_AGGREGATION_OPTIONS: { id: TrainerLogAggregation; label: string }[] = [
  { id: "last_batch", label: "Last batch at log tick" },
  { id: "interval_sample_mean", label: "Sample-weighted interval mean" },
];
const TRAINER_LOG_TIMING_OPTIONS: { id: TrainerLogTiming; label: string }[] = [
  { id: "post_update", label: "After matching update" },
  { id: "pre_update", label: "Before matching update" },
];
const TRAINER_TEST_EVALUATION_OPTIONS: { id: TrainerTestEvaluation; label: string }[] = [
  { id: "log_ticks", label: "Every log tick" },
  { id: "final_only", label: "Final log tick only" },
  { id: "disabled", label: "Disabled" },
];
const TRAINER_MINIBATCH_SAMPLING_OPTIONS: { id: TrainerMinibatchSampling; label: string }[] = [
  { id: "independent_step", label: "Independent random per step" },
  { id: "epoch_shuffle", label: "Epoch shuffle (no replacement)" },
  { id: "affine_epoch", label: "Affine epoch (no replacement)" },
];

function TrainerLengthModeSegment({
  mode,
  onModeChange,
}: {
  mode: TrainingLengthMode;
  onModeChange: (mode: TrainingLengthMode) => void;
}) {
  return (
    <span className="cr-trainer-length-segment nodrag nopan" role="group" aria-label="Training length">
      {TRAINING_LENGTH_MODE_OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          className={`cr-trainer-length-segment-btn${
            mode === opt.value ? " cr-trainer-length-segment-btn--active" : ""
          }`}
          aria-pressed={mode === opt.value}
          onClick={() => onModeChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </span>
  );
}

export function TrainerNode({ id, data, selected }: NodeProps) {
  const research = useResearchGraph();
  const def = defaultTrainerData();
  const raw = (data ?? {}) as Partial<TrainerNodeData>;
  const d: TrainerNodeData = {
    ...def,
    ...raw,
    trainingLengthMode: normalizeTrainingLengthMode(raw.trainingLengthMode ?? def.trainingLengthMode),
    trainingEpochs: raw.trainingEpochs ?? def.trainingEpochs,
  };
  const trainingLengthMode = normalizeTrainingLengthMode(d.trainingLengthMode);
  const logSchedule: TrainerLogSchedule =
    d.logSchedule === "idnns_logspace" ? "idnns_logspace" : "fixed_interval";
  const logAggregation: TrainerLogAggregation =
    d.logAggregation === "interval_sample_mean" ? "interval_sample_mean" : "last_batch";
  const logTiming: TrainerLogTiming =
    d.logTiming === "pre_update" ? "pre_update" : "post_update";
  const testEvaluation: TrainerTestEvaluation =
    d.testEvaluation === "final_only"
      ? "final_only"
      : d.testEvaluation === "disabled"
        ? "disabled"
        : "log_ticks";
  const minibatchSampling: TrainerMinibatchSampling =
    d.minibatchSampling === "affine_epoch"
      ? "affine_epoch"
      : d.minibatchSampling === "epoch_shuffle"
        ? "epoch_shuffle"
        : "independent_step";
  const rf = useReactFlow();
  const flowSurface = useFlowSurface();
  const setNodes = flowSurface?.applyNodes ?? rf.setNodes;
  const getNodes = flowSurface?.getNodes ?? rf.getNodes;
  const getEdges = flowSurface?.getEdges ?? rf.getEdges;
  const previewEpochSteps = useMemo(() => {
    if (trainingLengthMode !== "epochs") return null;
    const edges = getEdges().map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    }));
    const trainSize = resolveDatasetTrainSizeForTrainer(
      getNodes().map((n) => ({ id: n.id, data: (n.data ?? {}) as Record<string, unknown> })),
      edges,
      id,
    );
    return previewTrainingStepsForTrainer(d as Record<string, unknown>, trainSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingLengthMode, d.trainingEpochs, d.batchSize, getNodes, getEdges, id]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainPaused, setTrainPaused] = useState(false);
  const [trainPausing, setTrainPausing] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [remoteCfg, setRemoteCfg] = useState<RemoteTrainConfig>(DEFAULT_REMOTE_TRAIN_CONFIG);
  const [remoteCfgLoading, setRemoteCfgLoading] = useState(false);
  const [remoteStatus, setRemoteStatus] = useState<RemoteTrainStatus | null>(null);
  const [remoteStatusText, setRemoteStatusText] = useState<string | null>(null);
  const [remoteStatusError, setRemoteStatusError] = useState<string | null>(null);
  const [remoteSshCommand, setRemoteSshCommand] = useState("");
  const [localCudaDevices, setLocalCudaDevices] = useState<LocalCudaDeviceInfo[]>([]);
  const [localCudaLoading, setLocalCudaLoading] = useState(false);
  const [localCudaFetchError, setLocalCudaFetchError] = useState<string | null>(null);
  const [trainPhaseText, setTrainPhaseText] = useState<string | null>(null);
  const [trainProgressPct, setTrainProgressPct] = useState(0);
  const [trainSeriesBarPct, setTrainSeriesBarPct] = useState(0);
  const [trainSeriesDual, setTrainSeriesDual] = useState(false);
  const [trainSeriesCaptionLines, setTrainSeriesCaptionLines] = useState<string[] | null>(null);
  const progressHideTo = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  /** True once remote bootstrap finished and NDJSON stream is active (cooperative pause/abort). */
  const remoteCooperativeAbortRef = useRef(false);
  const pausedCheckpointRef = useRef<TrainStreamPaused | null>(null);
  /** Last train request’s parameter combo (so resume uses scalar `data`, not list fields). */
  const lastSeriesComboRef = useRef<TrainSeriesAssignment[] | null>(null);
  const currentSweepSummaryForVizRef = useRef("");
  const currentSweepParamsForVizRef = useRef<Record<string, string>>({});
  const seriesProgressCtxRef = useRef<{ totalRuns: number; runIndex: number } | null>(null);
  const pausedSeriesCtxRef = useRef<{ totalRuns: number; runIndex: number } | null>(null);
  const remoteAutosaveKeyRef = useRef("");
  const hessianChoiceResolverRef = useRef<((choice: "skip" | "force") => void) | null>(null);
  const hessianOversizedPolicyRef = useRef<"skip" | "force" | null>(null);
  // Default to true when the field is missing (pre-existing nodes don't have it).
  const liveVizUpdatesRef = useRef<boolean>(d.liveVizUpdates !== false);
  useEffect(() => {
    liveVizUpdatesRef.current = d.liveVizUpdates !== false;
  }, [d.liveVizUpdates]);
  const [hessianOversizeOpen, setHessianOversizeOpen] = useState(false);
  const [hessianOversizeDetail, setHessianOversizeDetail] = useState<{
    nParams: number;
    limit: number;
    message: string;
  } | null>(null);
  const hessianOversizeTitleId = useId();

  const loadRemoteState = useCallback(async () => {
    setRemoteCfgLoading(true);
    try {
      const [cfgRes, statusRes] = await Promise.all([
        fetch("/api/train/remote/config", { method: "GET" }),
        fetch("/api/train/remote/status", { method: "GET" }),
      ]);
      if (cfgRes.ok) {
        const cfg = (await cfgRes.json()) as RemoteTrainConfig;
        setRemoteCfg({
          host: cfg.host ?? "",
          user: cfg.user ?? "ubuntu",
          remote_path: cfg.remote_path ?? "",
          python: normalizeRemotePythonValue(cfg.python),
          identity: cfg.identity ?? "",
          password: cfg.password ?? "",
          extra_opts: cfg.extra_opts ?? "",
          enabled: Boolean(cfg.enabled),
          upload_dataset: Boolean(cfg.upload_dataset),
        });
        setRemoteSshCommand(
          buildSshCommandFromRemoteConfig({
            host: cfg.host ?? "",
            user: cfg.user ?? "ubuntu",
            remote_path: cfg.remote_path ?? "",
            python: normalizeRemotePythonValue(cfg.python),
            identity: cfg.identity ?? "",
            password: cfg.password ?? "",
            extra_opts: cfg.extra_opts ?? "",
            enabled: Boolean(cfg.enabled),
            upload_dataset: Boolean(cfg.upload_dataset),
          }),
        );
      }
      if (statusRes.ok) {
        const st = (await statusRes.json()) as RemoteTrainStatus;
        setRemoteStatus(st);
      }
    } catch {
      /* ignore boot-time remote state errors */
    } finally {
      setRemoteCfgLoading(false);
    }
  }, []);

  useEffect(
    () => () => {
      if (progressHideTo.current !== null) {
        window.clearTimeout(progressHideTo.current);
        progressHideTo.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    void loadRemoteState();
  }, [loadRemoteState]);

  useEffect(() => {
    if (!hessianOversizeOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        hessianChoiceResolverRef.current?.("skip");
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [hessianOversizeOpen]);

  const update = (patch: Partial<TrainerNodeData>) => patchTrainerData(id, patch, setNodes);
  const persistTrainUi = useCallback(
    (patch: Parameters<typeof patchTrainerTrainUi>[1]) => patchTrainerTrainUi(id, patch, setNodes),
    [id, setNodes],
  );
  const computeModeUi = computeModeUiFromDevice(d.computeDevice, d.remoteGpu);
  const usingAutoDlGpu = computeModeUi === "autodl_gpu";
  const usingLocalCuda = computeModeUi === "local_cuda";
  const localCudaGpuOptions = useMemo(() => localCudaGpuSelectOptions(localCudaDevices), [localCudaDevices]);
  const localCudaGpuValue = String(localCudaIndexFromDevice(d.computeDevice));

  useEffect(() => {
    if (!usingLocalCuda) return;
    let cancelled = false;
    setLocalCudaLoading(true);
    setLocalCudaFetchError(null);
    void fetch("/api/train/cuda-devices", { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) {
          let detail = res.statusText;
          try {
            const j = (await res.json()) as { detail?: unknown };
            if (j.detail != null) detail = String(j.detail);
          } catch {
            /* ignore */
          }
          throw new Error(detail);
        }
        return res.json() as Promise<{ devices?: LocalCudaDeviceInfo[] }>;
      })
      .then((payload) => {
        if (cancelled) return;
        setLocalCudaDevices(Array.isArray(payload.devices) ? payload.devices : []);
      })
      .catch((e) => {
        if (cancelled) return;
        setLocalCudaDevices([]);
        setLocalCudaFetchError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLocalCudaLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [usingLocalCuda]);

  useEffect(() => {
    if (!usingLocalCuda || localCudaLoading || localCudaDevices.length === 0) return;
    const idx = localCudaIndexFromDevice(d.computeDevice);
    if (localCudaDevices.some((dev) => dev.index === idx)) return;
    update({ computeDevice: localCudaDeviceFromIndex(localCudaDevices[0]!.index) });
  }, [usingLocalCuda, localCudaLoading, localCudaDevices, d.computeDevice, update]);

  const saveRemoteConfig = useCallback(async (enabled: boolean, opts?: { silent?: boolean }) => {
    const silent = Boolean(opts?.silent);
    if (!silent) {
      setRemoteStatusError(null);
      setRemoteStatusText(null);
    }
    try {
      const payload: RemoteTrainConfig = { ...remoteCfg, enabled };
      const res = await fetch("/api/train/remote/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      const saved = (await res.json()) as RemoteTrainConfig;
      setRemoteCfg({
        host: saved.host ?? "",
        user: saved.user ?? "ubuntu",
        remote_path: saved.remote_path ?? "",
        python: normalizeRemotePythonValue(saved.python),
        identity: saved.identity ?? "",
        password: saved.password ?? "",
        extra_opts: saved.extra_opts ?? "",
        enabled: Boolean(saved.enabled),
        upload_dataset: Boolean(saved.upload_dataset),
      });
      setRemoteSshCommand(
        buildSshCommandFromRemoteConfig({
          host: saved.host ?? "",
          user: saved.user ?? "ubuntu",
          remote_path: saved.remote_path ?? "",
          python: normalizeRemotePythonValue(saved.python),
          identity: saved.identity ?? "",
          password: saved.password ?? "",
          extra_opts: saved.extra_opts ?? "",
          enabled: Boolean(saved.enabled),
          upload_dataset: Boolean(saved.upload_dataset),
        }),
      );
      if (!silent) {
        setRemoteStatusText(enabled ? "AutoDL remote enabled." : "AutoDL remote disabled.");
        await loadRemoteState();
      }
    } catch (e) {
      setRemoteStatusError(e instanceof Error ? e.message : String(e));
    }
  }, [loadRemoteState, remoteCfg]);

  const prevUsingAutoDlGpuRef = useRef(false);

  useEffect(() => {
    if (!usingAutoDlGpu) {
      prevUsingAutoDlGpuRef.current = false;
      return;
    }
    if (remoteCfgLoading) return;
    const justEnabled = !prevUsingAutoDlGpuRef.current;
    prevUsingAutoDlGpuRef.current = true;
    if (justEnabled) {
      remoteAutosaveKeyRef.current = "";
      setTrainError(null);
      persistTrainUi({ error: null });
    }
    const payload: RemoteTrainConfig = { ...remoteCfg, enabled: true };
    const key = JSON.stringify(payload);
    if (key === remoteAutosaveKeyRef.current) return;
    const timer = window.setTimeout(() => {
      remoteAutosaveKeyRef.current = key;
      void (async () => {
        await saveRemoteConfig(true, { silent: true });
        await loadRemoteState();
      })();
    }, 420);
    return () => window.clearTimeout(timer);
  }, [loadRemoteState, persistTrainUi, remoteCfg, remoteCfgLoading, saveRemoteConfig, usingAutoDlGpu]);

  const syncPauseCheckpoint = useCallback(
    (paused: TrainStreamPaused | null) => {
      pausedCheckpointRef.current = paused;
      if (paused === null) {
        setNodes((nodes) =>
          nodes.map((n) => {
            if (n.id !== id || n.type !== "trainer") return n;
            const cur = (n.data ?? {}) as Partial<TrainerNodeData>;
            if (!cur.trainPauseCheckpoint) return n;
            const { trainPauseCheckpoint: _c, ...rest } = cur;
            return { ...n, data: rest };
          }),
        );
      } else {
        update({ trainPauseCheckpoint: paused as TrainerPauseCheckpoint });
      }
    },
    [id, setNodes, update],
  );

  const clearPausedTrainState = useCallback(() => {
    setTrainPaused(false);
    setTrainPausing(false);
    setTrainProgressPct(0);
    setTrainSeriesBarPct(0);
    setTrainSeriesDual(false);
    setTrainSeriesCaptionLines(null);
    pausedSeriesCtxRef.current = null;
    seriesProgressCtxRef.current = null;
    persistTrainUi(null);
    syncPauseCheckpoint(null);
    update({ trainSeriesPauseCtx: undefined });
  }, [persistTrainUi, syncPauseCheckpoint, update]);

  const syncSeriesPauseCtx = useCallback(
    (ctx: { totalRuns: number; runIndex: number } | null) => {
      if (ctx === null) {
        update({ trainSeriesPauseCtx: undefined });
        return;
      }
      pausedSeriesCtxRef.current = { totalRuns: ctx.totalRuns, runIndex: ctx.runIndex };
      update({ trainSeriesPauseCtx: ctx });
    },
    [update],
  );

  const remoteCooperativeControlReady = useCallback(
    () => usingAutoDlGpu && remoteCfg.enabled && remoteCooperativeAbortRef.current,
    [remoteCfg.enabled, usingAutoDlGpu],
  );

  const fallbackFetchAbortIfNeeded = useCallback(() => {
    if (!remoteCooperativeControlReady()) {
      abortControllerRef.current?.abort();
    }
  }, [remoteCooperativeControlReady]);

  const applyPausedTrainUi = useCallback(
    (paused: TrainStreamPaused) => {
      const pctWithin = progressPctFromStep(paused.next_step, d.trainingSteps);
      setTrainProgressPct(pctWithin);
      const ctx = pausedSeriesCtxRef.current;
      let seriesPct = 0;
      if (ctx && ctx.totalRuns > 1) {
        seriesPct = Math.min(
          100,
          Math.round(((ctx.runIndex + paused.next_step / Math.max(1, d.trainingSteps)) / ctx.totalRuns) * 100),
        );
        setTrainSeriesBarPct(seriesPct);
      }
      persistTrainUi({
        loading: false,
        paused: true,
        progressPct: pctWithin,
        seriesBarPct: seriesPct,
      });
    },
    [d.trainingSteps, persistTrainUi],
  );

  const sendTrainControl = useCallback(async (action: "pause" | "abort") => {
    try {
      const res = await fetch("/api/train/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trainer_node_id: id, action }),
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* ignore */
        }
        setTrainError(detail);
        if (res.status === 404 && usingAutoDlGpu && remoteCfg.enabled) {
          fallbackFetchAbortIfNeeded();
        }
      }
    } catch (e) {
      setTrainError(e instanceof Error ? e.message : String(e));
    }
  }, [fallbackFetchAbortIfNeeded, id, remoteCfg.enabled, usingAutoDlGpu]);

  const requestTrainPause = useCallback(() => {
    if (trainPausing) return;
    setTrainPausing(true);
    const sess = getTrainerRunSession(id);
    if (sess) abortControllerRef.current = sess;
    void sendTrainControl("pause");
  }, [id, sendTrainControl, trainPausing]);

  const runTrain = useCallback(
    async (resume: TrainResumeBody | null = null) => {
      const expectRemoteBootstrap =
        usingAutoDlGpu && !!remoteCfg.host.trim() && !!remoteCfg.remote_path.trim();
      const nodesForTrain = () =>
        getNodes().filter((n) => String(n.type) !== "graph_assist_failure_overlay");
      if (resume === null && hasTrainerRunSession(id)) {
        return;
      }

      setTrainError(null);
      setTrainLoading(true);
      setTrainPaused(false);
      setTrainPausing(false);
      remoteCooperativeAbortRef.current = false;
      const bootPhase = expectRemoteBootstrap ? "Remote: bootstrapping runtime..." : null;
      setTrainPhaseText(bootPhase);
      update({ hostTrainUi: undefined });
      if (resume === null) {
        setTrainProgressPct(0);
        setTrainSeriesBarPct(0);
        setTrainSeriesDual(false);
        setTrainSeriesCaptionLines(null);
        syncPauseCheckpoint(null);
        update({ trainCompleteCheckpoint: undefined });
        lastSeriesComboRef.current = null;
        pausedSeriesCtxRef.current = null;
        seriesProgressCtxRef.current = null;
        currentSweepSummaryForVizRef.current = "";
        currentSweepParamsForVizRef.current = {};
        hessianOversizedPolicyRef.current = null;
        persistTrainUi({
          loading: true,
          paused: false,
          progressPct: 0,
          seriesBarPct: 0,
          seriesDual: false,
          captionLines: null,
          phaseText: bootPhase,
          error: null,
        });
      } else {
        seriesProgressCtxRef.current = pausedSeriesCtxRef.current;
        const ctx = pausedSeriesCtxRef.current;
        setTrainSeriesDual(!!ctx && ctx.totalRuns > 1);
        const combo = lastSeriesComboRef.current;
        if (ctx && combo?.length) {
          const sweptIds = getSweptAxisIdSet(nodesForTrain(), getEdges(), id);
          const resumeLabelNodes = nodesForTrain().map((n) => ({
            id: n.id,
            type: n.type as string,
            position: n.position,
            data: (n.data as Record<string, unknown>) ?? {},
          }));
          const lines = formatTrainSeriesSweptLines(combo, resumeLabelNodes, sweptIds);
          setTrainSeriesCaptionLines(ctx.totalRuns > 1 ? lines : null);
        } else {
          setTrainSeriesCaptionLines(null);
        }
        persistTrainUi({
          loading: true,
          paused: false,
          phaseText: bootPhase,
          error: null,
        });
      }
      if (progressHideTo.current !== null) {
        window.clearTimeout(progressHideTo.current);
        progressHideTo.current = null;
      }

      const ac = new AbortController();
      abortControllerRef.current = ac;
      registerTrainerRunSession(id, ac);

      let preserveProgressAfter = false;

      const trainWireSnapshot = getEdges().map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle ?? null,
        targetHandle: e.targetHandle ?? null,
      }));

      const progressUiThrottler = createTrainProgressUiThrottler({
        flush: (patch) => persistTrainUi(patch),
      });

      const handleStreamProgress = (raw: TrainStreamProgress) => {
        if (raw.type !== "progress") return;
        const remotePhase = "Remote: running training...";
        if (expectRemoteBootstrap) setTrainPhaseText(remotePhase);
        const total = Math.max(1, raw.total);
        const step = Math.min(Math.max(0, raw.step), total);
        const within = step / total;
        const pctWithin = Math.min(100, Math.round(within * 100));
        setTrainProgressPct(pctWithin);
        const ctx = seriesProgressCtxRef.current;
        let seriesPct = 0;
        if (ctx && ctx.totalRuns > 1) {
          seriesPct = Math.min(100, Math.round(((ctx.runIndex + within) / ctx.totalRuns) * 100));
          setTrainSeriesBarPct(seriesPct);
        }
        progressUiThrottler.schedule(
          {
            progressPct: pctWithin,
            seriesBarPct: seriesPct,
            phaseText: expectRemoteBootstrap ? remotePhase : null,
          },
          step >= total,
        );
      };

      const onRemoteSessionReady = () => {
        remoteCooperativeAbortRef.current = true;
      };

      const readTrainStream = (reader: ReadableStreamDefaultReader<Uint8Array>) =>
        readNdjsonTrainStream(reader, handleStreamProgress, {
          onRemoteSession: expectRemoteBootstrap ? onRemoteSessionReady : undefined,
          onPhase: expectRemoteBootstrap ? (ev) => setTrainPhaseText(ev.message) : undefined,
          onMetrics: liveVizUpdatesRef.current
            ? (ev) => {
                applyTrainerVizPayload(
                  setNodes,
                  {
                    loss_history: ev.loss_history,
                    test_loss_history: ev.test_loss_history ?? [],
                    reg_loss_history: ev.reg_loss_history ?? [],
                    step_ticks: ev.step_ticks,
                    epoch_ticks: ev.epoch_ticks ?? [],
                    plot_png_base64: "",
                    visualization_node_ids: [],
                    observable_viz_updates: ev.observable_viz_updates,
                    observable_metric_histories: ev.observable_metric_histories ?? {},
                  },
                  id,
                  undefined,
                  trainWireSnapshot,
                  undefined,
                  true,
                );
              }
            : undefined,
        });

      const fetchTrain = async (
        nodesPayload: SerializedTrainNode[],
        edgesPayload: SerializedTrainEdge[],
        resumeBody: TrainResumeBody | null,
      ) => {
        while (true) {
          const nodesForPost = resumeBody
            ? nodesPayload
            : resolveTrainingLengthOnNodes(nodesPayload, edgesPayload);
          const body: Record<string, unknown> = {
            trainer_node_id: id,
            nodes: nodesForPost,
            edges: edgesPayload,
          };
          if (resumeBody) body.resume = resumeBody;
          const pol = hessianOversizedPolicyRef.current;
          if (pol) body.hessian_oversized_policy = pol;

          if (expectRemoteBootstrap) setTrainPhaseText("Remote: bootstrapping runtime...");
          const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: ac.signal,
          });

          if (expectRemoteBootstrap && res.ok) {
            setTrainPhaseText("Remote: sending experiment graph...");
          }

          if (res.status === 409) {
            let detail: unknown;
            try {
              const j = (await res.json()) as { detail?: unknown };
              detail = j.detail;
            } catch {
              throw new Error("Training rejected (409) with a non-JSON body.");
            }
            if (isHessianOversizedDetail(detail)) {
              const choice = await new Promise<"skip" | "force">((resolve, reject) => {
                const onAbort = () => {
                  hessianChoiceResolverRef.current = null;
                  setHessianOversizeOpen(false);
                  setHessianOversizeDetail(null);
                  ac.signal.removeEventListener("abort", onAbort);
                  reject(new DOMException("Aborted", "AbortError"));
                };
                const finish = (c: "skip" | "force") => {
                  ac.signal.removeEventListener("abort", onAbort);
                  hessianChoiceResolverRef.current = null;
                  setHessianOversizeOpen(false);
                  setHessianOversizeDetail(null);
                  resolve(c);
                };
                hessianChoiceResolverRef.current = finish;
                ac.signal.addEventListener("abort", onAbort, { once: true });
                setHessianOversizeDetail({
                  nParams: detail.n_params,
                  limit: detail.limit,
                  message:
                    typeof detail.message === "string" && detail.message.trim()
                      ? detail.message
                      : `This model has ${detail.n_params} trainable parameters (limit ${detail.limit} for interactive Hessian choice).`,
                });
                setHessianOversizeOpen(true);
              });
              hessianOversizedPolicyRef.current = choice;
              continue;
            }
            let msg = res.statusText;
            if (detail != null) msg = typeof detail === "string" ? detail : JSON.stringify(detail);
            throw new Error(msg);
          }

          if (!res.ok) {
            let detail = res.statusText;
            try {
              const j = (await res.json()) as { detail?: unknown };
              if (j?.detail != null) {
                detail =
                  typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
              }
            } catch {
              /* ignore */
            }
            throw new Error(detail);
          }

          const reader = res.body?.getReader();
          if (!reader) throw new Error("No response body");
          return reader;
        }
      };

      const finishSeriesSweep = () => {
        syncPauseCheckpoint(null);
        update({ trainSeriesPauseCtx: undefined });
        setTrainPaused(false);
        setTrainPausing(false);
        pausedSeriesCtxRef.current = null;
        seriesProgressCtxRef.current = null;
      };

      const runSeriesSweep = async (startRunIdx: number): Promise<void> => {
        const seriesRfNodesSnapshot = getNodes().filter((n) => String(n.type) !== "graph_assist_failure_overlay");
        let combos: TrainSeriesAssignment[][];
        try {
          combos = planTrainSeriesAssignments(seriesRfNodesSnapshot, trainWireSnapshot as Edge[], id);
        } catch (planErr) {
          setTrainError(planErr instanceof Error ? planErr.message : String(planErr));
          return;
        }
        if (startRunIdx === 0) {
          research?.onTrainerTrainPressed?.(id);
        }
        const totalRuns = combos.length;
        const sweptAxisIds = getSweptAxisIdSet(seriesRfNodesSnapshot, trainWireSnapshot as Edge[], id);
        setTrainSeriesDual(totalRuns > 1);
        if (startRunIdx === 0) {
          persistTrainUi({ seriesDual: totalRuns > 1 });
        }

        const trainGraph = serializeExecutionGraphForTarget(
          seriesRfNodesSnapshot,
          trainWireSnapshot as Edge[],
          id,
        );
        const baseNodes = trainGraph.nodes;

        for (let runIdx = startRunIdx; runIdx < totalRuns; runIdx++) {
          if (ac.signal.aborted) break;
          if (totalRuns > 1) {
            clearTrainerLinkedVizForSeriesRun(setNodes, nodesForTrain(), trainWireSnapshot as Edge[], id);
          }
          const combo = combos[runIdx]!;
          lastSeriesComboRef.current = combo;
          seriesProgressCtxRef.current = totalRuns > 1 ? { totalRuns, runIndex: runIdx } : null;
          const paramLines = formatTrainSeriesSweptLines(combo, baseNodes, sweptAxisIds);
          const nodesRunPreview = applyAssignmentsAndResolveTrainingLength(baseNodes, combo, trainWireSnapshot);
          const trainerRunParams = buildTrainerRunSweepParams(
            (nodesRunPreview.find((n) => n.id === id)?.data ?? {}) as Record<string, unknown>,
          );
          currentSweepSummaryForVizRef.current =
            totalRuns > 1 ? paramLines.join(", ") : formatSweepParamsSummary(trainerRunParams) || paramLines.join(", ");
          currentSweepParamsForVizRef.current =
            totalRuns > 1 ? buildSweepParamsFromCombo(combo, baseNodes, sweptAxisIds) : trainerRunParams;
          setTrainSeriesCaptionLines(totalRuns > 1 ? paramLines : null);
          setTrainProgressPct(0);
          if (totalRuns > 1) {
            const bar = Math.min(100, Math.round((runIdx / totalRuns) * 100));
            setTrainSeriesBarPct(bar);
            persistTrainUi({
              seriesDual: true,
              captionLines: paramLines,
              progressPct: 0,
              seriesBarPct: bar,
            });
          } else {
            persistTrainUi({ progressPct: 0, seriesBarPct: 0, captionLines: null, seriesDual: false });
          }
          const nodesRun = nodesRunPreview;
          const reader = await fetchTrain(nodesRun, trainGraph.edges, null);
          const streamResult = await readTrainStream(reader);
          if (streamResult.error) throw new Error(streamResult.error);
          const { complete, paused, aborted } = streamResult;

          if (aborted || ac.signal.aborted) {
            setTrainPaused(false);
            setTrainPausing(false);
            syncPauseCheckpoint(null);
            setTrainProgressPct(0);
            setTrainSeriesBarPct(0);
            setTrainSeriesDual(false);
            setTrainSeriesCaptionLines(null);
            pausedSeriesCtxRef.current = null;
            seriesProgressCtxRef.current = null;
            persistTrainUi(null);
            return;
          }

          if (paused) {
            preserveProgressAfter = true;
            pausedSeriesCtxRef.current = seriesProgressCtxRef.current
              ? { ...seriesProgressCtxRef.current }
              : null;
            await flushCheckpointApplyTrainerVizAndHydrateTv0d({
              isCrl: false,
              setNodes,
              getNodes,
              getEdges,
              trainerNodeId: id,
              wires: trainWireSnapshot,
              payload: paused,
              sweepSummary: currentSweepSummaryForVizRef.current,
              sweepParams:
                Object.keys(currentSweepParamsForVizRef.current).length > 0
                  ? currentSweepParamsForVizRef.current
                  : undefined,
              preserveTrainUi: true,
            });
            syncPauseCheckpoint(paused);
            setTrainPausing(false);
            setTrainPaused(true);
            applyPausedTrainUi(paused);
            if (pausedSeriesCtxRef.current) {
              syncSeriesPauseCtx({
                totalRuns: pausedSeriesCtxRef.current.totalRuns,
                runIndex: pausedSeriesCtxRef.current.runIndex,
              });
            }
            return;
          }

          if (!complete) {
            throw new Error("Train stream ended without a complete or paused event");
          }

          const hasMoreSeriesRuns = totalRuns > 1 && runIdx < totalRuns - 1;

          await flushCheckpointApplyTrainerVizAndHydrateTv0d({
            isCrl: false,
            setNodes,
            getNodes,
            getEdges,
            trainerNodeId: id,
            wires: trainWireSnapshot,
            payload: complete,
            sweepSummary: currentSweepSummaryForVizRef.current,
            sweepParams:
              Object.keys(currentSweepParamsForVizRef.current).length > 0
                ? currentSweepParamsForVizRef.current
                : undefined,
            extraRefreshDelaysMs: hasMoreSeriesRuns ? [] : [250, 900],
            preserveTrainUi: hasMoreSeriesRuns,
            skipTensorVizHydration: hasMoreSeriesRuns,
          });
          if (!hasMoreSeriesRuns) {
            const ck = buildCompleteCheckpoint(complete);
            if (ck) update({ trainCompleteCheckpoint: ck });
          }
          if (hasMoreSeriesRuns) {
            persistTrainUi({ loading: true, paused: false });
          }
          if (hasMoreSeriesRuns) {
            await sleepBetweenSweepSeriesRuns(ac.signal);
            if (ac.signal.aborted) break;
          }
        }

        finishSeriesSweep();
      };

      try {
        if (resume) {
          const trainGraph = serializeExecutionGraphForTarget(
            nodesForTrain(),
            trainWireSnapshot as Edge[],
            id,
          );
          const freshBase = trainGraph.nodes;
          const resumeNodes =
            lastSeriesComboRef.current && lastSeriesComboRef.current.length
              ? applyAssignmentsToNodes(freshBase, lastSeriesComboRef.current)
              : freshBase;
          const resumeVizSummary = (): string | undefined => {
            const combo = lastSeriesComboRef.current;
            if (!combo?.length) return undefined;
            const sweptIds = getSweptAxisIdSet(nodesForTrain(), getEdges(), id);
            return formatTrainSeriesComboCaption(combo, resumeNodes, sweptIds);
          };
          const resumeVizParams = (): Record<string, string> | undefined => {
            const combo = lastSeriesComboRef.current;
            if (!combo?.length) return undefined;
            const sweptIds = getSweptAxisIdSet(nodesForTrain(), getEdges(), id);
            const params = buildSweepParamsFromCombo(combo, resumeNodes, sweptIds);
            return Object.keys(params).length ? params : undefined;
          };
          const reader = await fetchTrain(resumeNodes, trainGraph.edges, resume);
          const streamResult = await readTrainStream(reader);
          if (streamResult.error) throw new Error(streamResult.error);
          const { complete, paused, aborted } = streamResult;

          if (aborted || ac.signal.aborted) {
            setTrainPaused(false);
            setTrainPausing(false);
            syncPauseCheckpoint(null);
            setTrainProgressPct(0);
            setTrainSeriesBarPct(0);
            setTrainSeriesDual(false);
            setTrainSeriesCaptionLines(null);
            pausedSeriesCtxRef.current = null;
            seriesProgressCtxRef.current = null;
            persistTrainUi(null);
            return;
          }

          if (paused) {
            preserveProgressAfter = true;
            await flushCheckpointApplyTrainerVizAndHydrateTv0d({
              isCrl: false,
              setNodes,
              getNodes,
              getEdges,
              trainerNodeId: id,
              wires: trainWireSnapshot,
              payload: paused,
              sweepSummary: resumeVizSummary(),
              sweepParams: resumeVizParams(),
              preserveTrainUi: true,
            });
            syncPauseCheckpoint(paused);
            setTrainPausing(false);
            setTrainPaused(true);
            applyPausedTrainUi(paused);
            if (pausedSeriesCtxRef.current) {
              syncSeriesPauseCtx({
                totalRuns: pausedSeriesCtxRef.current.totalRuns,
                runIndex: pausedSeriesCtxRef.current.runIndex,
              });
            }
            if (pausedSeriesCtxRef.current && pausedSeriesCtxRef.current.totalRuns > 1) {
              const combo = lastSeriesComboRef.current;
              if (combo?.length) {
                const sweptIds = getSweptAxisIdSet(nodesForTrain(), getEdges(), id);
                setTrainSeriesCaptionLines(formatTrainSeriesSweptLines(combo, resumeNodes, sweptIds));
              }
            }
            return;
          }

          if (!complete) {
            throw new Error("Train stream ended without a complete or paused event");
          }

          const seriesCtx = pausedSeriesCtxRef.current;
          const hasMoreAfterResume =
            seriesCtx !== null && seriesCtx.totalRuns > 1 && seriesCtx.runIndex < seriesCtx.totalRuns - 1;

          await flushCheckpointApplyTrainerVizAndHydrateTv0d({
            isCrl: false,
            setNodes,
            getNodes,
            getEdges,
            trainerNodeId: id,
            wires: trainWireSnapshot,
            payload: complete,
            sweepSummary: resumeVizSummary(),
            sweepParams: resumeVizParams(),
            extraRefreshDelaysMs: hasMoreAfterResume ? [] : [250, 900],
            preserveTrainUi: hasMoreAfterResume,
            skipTensorVizHydration: hasMoreAfterResume,
          });
          const seriesCompleteCp = buildCompleteCheckpoint(complete);
          if (seriesCompleteCp) update({ trainCompleteCheckpoint: seriesCompleteCp });
          syncPauseCheckpoint(null);
          update({ trainSeriesPauseCtx: undefined });
          setTrainPaused(false);

          if (!hasMoreAfterResume) {
            const ck = buildCompleteCheckpoint(complete);
            if (ck) update({ trainCompleteCheckpoint: ck });
          }

          if (hasMoreAfterResume && seriesCtx) {
            pausedSeriesCtxRef.current = null;
            persistTrainUi({ loading: true, paused: false });
            seriesProgressCtxRef.current = { totalRuns: seriesCtx.totalRuns, runIndex: seriesCtx.runIndex };
            await sleepBetweenSweepSeriesRuns(ac.signal);
            if (!ac.signal.aborted) {
              await runSeriesSweep(seriesCtx.runIndex + 1);
            }
          } else {
            pausedSeriesCtxRef.current = null;
            seriesProgressCtxRef.current = null;
            setTrainSeriesDual(false);
            setTrainSeriesCaptionLines(null);
            setTrainSeriesBarPct(0);
          }
        } else {
          await runSeriesSweep(0);
        }
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") {
          setTrainPaused(false);
          setTrainPausing(false);
          syncPauseCheckpoint(null);
          setTrainProgressPct(0);
          setTrainSeriesBarPct(0);
          setTrainSeriesDual(false);
          setTrainSeriesCaptionLines(null);
          pausedSeriesCtxRef.current = null;
          seriesProgressCtxRef.current = null;
          persistTrainUi(null);
          return;
        }
        const errMsg = e instanceof Error ? e.message : String(e);
        setTrainError(errMsg);
        setTrainProgressPct(0);
        setTrainSeriesBarPct(0);
        setTrainPaused(false);
        setTrainPausing(false);
        syncPauseCheckpoint(null);
        setTrainSeriesDual(false);
        setTrainSeriesCaptionLines(null);
        pausedSeriesCtxRef.current = null;
        seriesProgressCtxRef.current = null;
        persistTrainUi({ error: errMsg, loading: false, paused: false, progressPct: 0, seriesBarPct: 0 });
      } finally {
        progressUiThrottler.flushNow();
        progressUiThrottler.cancel();
        unregisterTrainerRunSession(id);
        if (!preserveProgressAfter) {
          setTrainLoading(false);
          setTrainPausing(false);
        }
        setTrainPhaseText(null);
        if (preserveProgressAfter) {
          persistTrainUi({ phaseText: null });
        } else {
          persistTrainUi({ loading: false, phaseText: null });
        }
        abortControllerRef.current = null;
        if (!preserveProgressAfter) {
          progressHideTo.current = window.setTimeout(() => {
            progressHideTo.current = null;
            setTrainProgressPct(0);
            setTrainSeriesBarPct(0);
            setTrainSeriesDual(false);
            setTrainSeriesCaptionLines(null);
            persistTrainUi(null);
          }, 420);
        }
      }
    },
    [applyPausedTrainUi, d.trainSeriesPauseCtx, getEdges, getNodes, id, persistTrainUi, research, remoteCfg.host, remoteCfg.remote_path, setNodes, syncPauseCheckpoint, syncSeriesPauseCtx, update, usingAutoDlGpu],
  );

  const trainDisp = resolveTrainerTrainDisplay(
    {
      trainLoading,
      trainPaused,
      trainProgressPct,
      trainSeriesBarPct,
      trainSeriesDual,
      trainSeriesCaptionLines,
      trainPhaseText,
      trainError,
    },
    d.trainUi,
    d.hostTrainUi,
  );
  const runSessionActive = hasTrainerRunSession(id);
  const uiLoading = trainDisp.loading || (runSessionActive && !trainDisp.paused);
  const uiPaused = trainDisp.paused && !uiLoading;
  const hostDrivingProgress = trainDisp.hostDrivingProgress;
  const dispSeriesDual = trainDisp.seriesDual;
  const dispSeriesBarPct = trainDisp.seriesBarPct;
  const dispProgressPct = trainDisp.progressPct;
  const dispCaptionLines = trainDisp.captionLines;

  const showTrainProgress =
    trainDisp.showProgress || (runSessionActive && !uiPaused);
  const showRemotePhaseBanner =
    Boolean(trainDisp.phaseText) &&
    !uiPaused &&
    !hostDrivingProgress &&
    uiLoading &&
    trainDisp.progressPct <= 0 &&
    trainDisp.seriesBarPct <= 0;

  const seriesLinesForUi = dispCaptionLines ?? [];

  useLayoutEffect(() => {
    const sess = getTrainerRunSession(id);
    if (sess) abortControllerRef.current = sess;

    const ui = d.trainUi;
    if (ui?.active) {
      setTrainLoading(Boolean(ui.loading));
      setTrainPaused(Boolean(ui.paused));
      setTrainProgressPct(ui.progressPct);
      setTrainSeriesBarPct(ui.seriesBarPct);
      setTrainSeriesDual(Boolean(ui.seriesDual));
      setTrainSeriesCaptionLines(ui.captionLines ?? null);
      setTrainPhaseText(ui.phaseText ?? null);
      setTrainError(ui.error ?? null);
    } else if (sess) {
      setTrainLoading(true);
      setTrainPaused(false);
    }

    const ck = d.trainPauseCheckpoint;
    if (ck) {
      pausedCheckpointRef.current = ck as TrainStreamPaused;
      if (d.trainUi?.paused) setTrainPaused(true);
    } else if (!d.trainUi?.paused) {
      pausedCheckpointRef.current = null;
    }

    const seriesCtx = d.trainSeriesPauseCtx;
    if (seriesCtx) {
      pausedSeriesCtxRef.current = { totalRuns: seriesCtx.totalRuns, runIndex: seriesCtx.runIndex };
      seriesProgressCtxRef.current = { totalRuns: seriesCtx.totalRuns, runIndex: seriesCtx.runIndex };
      const legacyCombo = (seriesCtx as { combo?: TrainSeriesAssignment[] }).combo;
      if (legacyCombo?.length) {
        lastSeriesComboRef.current = legacyCombo;
      } else {
        try {
          const nodes = getNodes().filter((n) => String(n.type) !== "graph_assist_failure_overlay");
          const combos = planTrainSeriesAssignments(nodes, getEdges(), id);
          const combo = combos[seriesCtx.runIndex];
          if (combo?.length) lastSeriesComboRef.current = combo;
        } catch {
          /* graph may not be ready */
        }
      }
    }
  }, [d.trainPauseCheckpoint, d.trainSeriesPauseCtx, d.trainUi, getEdges, getNodes, id]);

  const hessianOversizeModal =
    hessianOversizeOpen && hessianOversizeDetail
      ? createPortal(
          <div
            className="cr-modal-backdrop"
            style={{ zIndex: 10032 }}
            role="presentation"
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) hessianChoiceResolverRef.current?.("skip");
            }}
          >
            <div
              className="cr-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby={hessianOversizeTitleId}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <h2 id={hessianOversizeTitleId} className="cr-modal__title">
                Large model — Hessian eigenvalues
              </h2>
              <p className="cr-modal__hint">{hessianOversizeDetail.message}</p>
              <div className="cr-modal__actions cr-modal__actions--paste-choice">
                <button
                  type="button"
                  className="cr-modal__btn cr-modal__btn--primary"
                  onClick={() => hessianChoiceResolverRef.current?.("skip")}
                >
                  Skip Hessian
                </button>
                <button
                  type="button"
                  className="cr-modal__btn cr-modal__btn--primary"
                  onClick={() => hessianChoiceResolverRef.current?.("force")}
                >
                  Continue anyway
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
    <div
      className={`cr-node cr-node--trainer${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-trainer)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        {showTrainProgress ? (
          showRemotePhaseBanner ? (
            <div className="cr-trainer-header-progress-with-grip">
              <div className="cr-trainer-phase-banner nodrag nopan">{trainDisp.phaseText}</div>
            </div>
          ) : dispSeriesDual ? (
            <div className="cr-trainer-header-progress-with-grip">
              <div className="cr-trainer-progress-stack nodrag nopan">
              <div
                className="cr-trainer-progress cr-trainer-progress--series nodrag nopan"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(dispSeriesBarPct)}
                aria-valuetext={
                  seriesLinesForUi.length
                    ? `${seriesLinesForUi.join("; ")} — ${Math.round(dispSeriesBarPct)}%`
                    : "Sweep progress"
                }
                aria-label="Sweep series progress"
              >
                <div
                  className="cr-trainer-progress__fill cr-trainer-progress__fill--series"
                  style={{ width: `${Math.min(100, dispSeriesBarPct)}%` }}
                />
                <div
                  className="cr-trainer-progress__lines"
                  title={seriesLinesForUi.length ? seriesLinesForUi.join(", ") : undefined}
                >
                  {seriesLinesForUi.length ? (
                    seriesLinesForUi.map((line, i) => (
                      <div key={`${i}:${line}`} className="cr-trainer-progress__line">
                        {line}
                      </div>
                    ))
                  ) : (
                    <div className="cr-trainer-progress__line cr-trainer-progress__line--placeholder">
                      {"\u00a0"}
                    </div>
                  )}
                </div>
              </div>
              <div
                className="cr-trainer-progress cr-trainer-progress--within nodrag nopan"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(dispProgressPct)}
                aria-valuetext={`${Math.min(100, Math.round(dispProgressPct))}% of current run`}
                aria-label="Current experiment progress"
              >
                <div
                  className="cr-trainer-progress__fill"
                  style={{ width: `${Math.min(100, dispProgressPct)}%` }}
                />
                <span className="cr-trainer-progress__label">
                  {Math.min(100, Math.round(dispProgressPct))}%
                </span>
              </div>
              </div>
            </div>
          ) : (
            <div className="cr-trainer-header-progress-with-grip">
              <div
                className="cr-trainer-progress nodrag nopan"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(dispProgressPct)}
                aria-valuetext={`${Math.min(100, Math.round(dispProgressPct))}%`}
                aria-label="Training progress"
              >
                <div
                  className="cr-trainer-progress__fill"
                  style={{ width: `${Math.min(100, dispProgressPct)}%` }}
                />
                <span className="cr-trainer-progress__label">
                  {Math.min(100, Math.round(dispProgressPct))}%
                </span>
              </div>
            </div>
          )
        ) : null}
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title nodrag nopan">Trainer</span>
          {uiPaused ? (
            <div className="cr-trainer-train-btn-row nodrag nopan">
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl nodrag nopan"
                onClick={() => {
                  const p = pausedCheckpointRef.current ?? d.trainPauseCheckpoint ?? null;
                  if (p) void runTrain(buildResumeBody(p));
                }}
              >
                Continue
              </button>
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl cr-trainer-train-btn--danger nodrag nopan"
                onClick={() => clearPausedTrainState()}
              >
                Abort
              </button>
            </div>
          ) : uiLoading ? (
            <div className="cr-trainer-train-btn-row nodrag nopan">
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl nodrag nopan"
                disabled={trainPausing}
                aria-busy={trainPausing}
                onClick={() => requestTrainPause()}
              >
                {trainPausing ? "Pausing…" : "Pause"}
              </button>
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl cr-trainer-train-btn--danger nodrag nopan"
                onClick={() => {
                  setTrainPausing(false);
                  void sendTrainControl("abort");
                  if (!remoteCooperativeControlReady()) {
                    abortControllerRef.current?.abort();
                  }
                }}
              >
                Abort
              </button>
            </div>
          ) : d.trainCompleteCheckpoint ? (
            <div className="cr-trainer-train-btn-row nodrag nopan">
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl nodrag nopan"
                onClick={() => {
                  const cp = d.trainCompleteCheckpoint;
                  if (cp) void runTrain(buildResumeBody(cp));
                }}
              >
                Continue
              </button>
              <button
                type="button"
                className="cr-trainer-train-btn cr-trainer-train-btn--ctl cr-trainer-train-btn--danger nodrag nopan"
                onClick={() => update({ trainCompleteCheckpoint: undefined })}
              >
                Dismiss
              </button>
            </div>
          ) : (
            <div className="cr-trainer-train-btn-row">
              <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void runTrain(null)}>
                Train
              </button>
            </div>
          )}
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Trainer inputs and outputs">
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="dataset" label="dataset" fullWidth />
          </div>
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="model" label="model" />
            <TrainerOutWrap handleId="checkpoint" label="model ckpt" accent="model" />
          </div>
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="optimizer" label="optimizer" fullWidth />
          </div>
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="loss" label="loss" />
            <TrainerOutWrap handleId="loss_results" label="loss" accent="loss" />
          </div>
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="observables" label="observables" />
            <TrainerOutWrap handleId="observable_results" label="observable" accent="observable" />
          </div>
          <div className="cr-trainer-io-row">
            <TrainerInWrap handleId="batch_schedule" label="batch sched" fullWidth />
          </div>
        </div>

        {trainingLengthMode === "steps" ? (
          <ComfyIntField
            labelNode={
              <TrainerLengthModeSegment
                mode={trainingLengthMode}
                onModeChange={(mode) => update({ trainingLengthMode: normalizeTrainingLengthMode(mode) })}
              />
            }
            value={d.trainingSteps}
            min={1}
            onCommit={(trainingSteps) => update({ trainingSteps })}
            ariaLabel="Training steps"
          />
        ) : (
          <>
            <ComfyIntField
              labelNode={
                <TrainerLengthModeSegment
                  mode={trainingLengthMode}
                  onModeChange={(mode) => update({ trainingLengthMode: normalizeTrainingLengthMode(mode) })}
                />
              }
              value={d.trainingEpochs ?? def.trainingEpochs ?? 100}
              min={1}
              title="Steps computed at train time: epochs × ceil(trainSize / batchSize)"
              onCommit={(trainingEpochs) => update({ trainingEpochs })}
              ariaLabel="Training epochs"
            />
            {previewEpochSteps != null ? (
              <p className="cr-tviz-sweep-line nodrag nopan">
                ≈ {previewEpochSteps} steps at current batch (from wired dataset train size)
              </p>
            ) : (
              <p className="cr-sweep-viz__plot-muted nodrag nopan">Connect a dataset to preview steps from epochs</p>
            )}
          </>
        )}

        {logSchedule === "fixed_interval" ? (
          <ComfyIntField
            label="log frequency"
            value={d.logFrequency}
            min={1}
            title="Log metrics every N steps"
            onCommit={(logFrequency) => update({ logFrequency })}
            ariaLabel="Log frequency (steps)"
          />
        ) : null}

        <ComfyIntListField
          label="batch size"
          values={intChoices(d.batchSize, -1)}
          min={-1}
          title="-1 = full batch; otherwise use the selected minibatch policy. Comma-separated for sweep."
          onCommit={(vals) => update({ batchSize: packIntList(vals) })}
          ariaLabel="Training batch size (-1 for full batch)"
        />

        <DiscreteMultiSelect<ComputeModeUi>
          label="compute device"
          options={COMPUTE_MODE_OPTIONS}
          value={computeModeUi}
          singleSelect
          presentation="segmented"
          segmentLabels={{ local_cuda: "CUDA", autodl_gpu: "AutoDL" }}
          ariaLabel="Compute device mode"
          onCommit={(next) => {
            const mode = normalizeComputeModeUi(next);
            if (mode === "local_cuda") {
              update({
                computeDevice: defaultLocalCudaDevice(d.computeDevice),
                remoteGpu: false,
              });
              return;
            }
            update({
              computeDevice: computeDeviceFromModeUi(mode),
              remoteGpu: remoteGpuFromModeUi(mode),
              ...(mode === "autodl_gpu" ? { hostTrainUi: undefined } : {}),
            });
            if (mode === "autodl_gpu") {
              setTrainError(null);
            }
          }}
        />

        {usingLocalCuda ? (
          localCudaGpuOptions.length > 0 ? (
            <DiscreteMultiSelect
              label="local GPU"
              options={localCudaGpuOptions}
              value={localCudaGpuValue}
              singleSelect
              ariaLabel="Local CUDA GPU index"
              onCommit={(next) => {
                const picked = String(Array.isArray(next) ? next[0] : next);
                const index = Number.parseInt(picked, 10);
                if (!Number.isFinite(index) || index < 0) return;
                update({ computeDevice: localCudaDeviceFromIndex(index), remoteGpu: false });
              }}
            />
          ) : (
            <p className="cr-trainer-remote__hint">
              {localCudaLoading
                ? "Detecting local CUDA GPUs…"
                : localCudaFetchError
                  ? `Could not list CUDA GPUs: ${localCudaFetchError}`
                  : "No local CUDA GPU detected on this machine."}
            </p>
          )
        ) : null}

        {usingAutoDlGpu ? (
        <div className="cr-trainer-remote">
          <div className="cr-trainer-remote__title-row">
            <span className="cr-trainer-remote__title">AutoDL remote GPU</span>
            {remoteCfgLoading ? <span className="cr-trainer-remote__hint">loading…</span> : null}
          </div>
          <p className="cr-trainer-remote__hint">
            {remoteStatus?.mode === "remote"
              ? `Connected: ${remoteStatus.remoteHost || remoteCfg.host || "remote host"} (${remoteStatus.source || "stored"}).`
              : "Local mode. Configure SSH + path below; settings are auto-saved."}
          </p>
          <div className="cr-comfy-widget">
            <span className="cr-comfy-widget__label">SSH command</span>
            <div className="cr-trainer-remote__ssh-command-row">
              <textarea
                className="cr-input cr-comfy-widget__control nodrag nopan"
                aria-label="AutoDL ssh command"
                value={remoteSshCommand}
                rows={2}
                onChange={(e) => {
                  const next = e.target.value;
                  setRemoteSshCommand(next);
                  const parsed = parseSshCommandToRemoteConfig(next);
                  if (!parsed) return;
                  setRemoteStatusError(null);
                  setRemoteCfg((prev) => ({
                    ...prev,
                    host: parsed.host ?? prev.host,
                    user: parsed.user ?? prev.user,
                    identity: parsed.identity ?? prev.identity,
                    extra_opts: parsed.extra_opts ?? prev.extra_opts,
                  }));
                }}
                onBlur={() => {
                  const parsed = parseSshCommandToRemoteConfig(remoteSshCommand);
                  if (!parsed) {
                    setRemoteStatusError(
                      "Invalid SSH command. Example: ssh -p 12033 root@region-42.seetacloud.com",
                    );
                  }
                }}
                placeholder="ssh -p 12033 root@region-42.seetacloud.com"
              />
            </div>
          </div>
          <div className="cr-comfy-widget">
            <span className="cr-comfy-widget__label">Password</span>
            <input
              type="password"
              className="cr-input cr-comfy-widget__control nodrag nopan"
              aria-label="AutoDL ssh password"
              value={remoteCfg.password}
              onChange={(e) => setRemoteCfg((prev) => ({ ...prev, password: e.target.value }))}
              placeholder="optional, for password auth"
            />
          </div>
          <div className="cr-comfy-widget">
            <span className="cr-comfy-widget__label">Remote path</span>
            <input
              className="cr-input cr-comfy-widget__control nodrag nopan"
              aria-label="AutoDL remote repo path"
              value={remoteCfg.remote_path}
              onChange={(e) => setRemoteCfg((prev) => ({ ...prev, remote_path: e.target.value }))}
              placeholder="/root/ComfyResearch"
            />
          </div>
          <div className="cr-comfy-widget">
            <span className="cr-comfy-widget__label">Python</span>
            <input
              className="cr-input cr-comfy-widget__control nodrag nopan"
              aria-label="AutoDL remote python command"
              value={remoteCfg.python}
              onChange={(e) => setRemoteCfg((prev) => ({ ...prev, python: e.target.value }))}
              placeholder={DEFAULT_REMOTE_PYTHON}
            />
          </div>
          <div className="cr-comfy-widget">
            <span className="cr-comfy-widget__label">SSH key path</span>
            <input
              className="cr-input cr-comfy-widget__control nodrag nopan"
              aria-label="AutoDL ssh identity path"
              value={remoteCfg.identity}
              onChange={(e) => setRemoteCfg((prev) => ({ ...prev, identity: e.target.value }))}
              placeholder="~/.ssh/id_rsa"
            />
          </div>
          <label className="cr-node__field cr-node__field--checkbox nodrag nopan">
            <input
              type="checkbox"
              aria-label="Upload local dataset"
              checked={Boolean(remoteCfg.upload_dataset)}
              onChange={(e) =>
                setRemoteCfg((prev) => ({ ...prev, upload_dataset: e.target.checked }))
              }
            />
            <span>Upload local dataset (server downloads by default)</span>
          </label>
          {remoteStatusText ? <p className="cr-trainer-remote__ok">{remoteStatusText}</p> : null}
          {remoteStatus?.lastValidationError && (uiLoading || trainDisp.error) ? (
            <p className="cr-trainer-remote__warn">{remoteStatus.lastValidationError}</p>
          ) : null}
          {remoteStatusError ? <p className="cr-trainer-train-err">{remoteStatusError}</p> : null}
        </div>
        ) : null}

        <ComfyFloatField
          label="grad clip max norm"
          value={d.gradClipMaxNorm}
          min={0}
          title="0 disables global L2 norm clip after backward."
          onCommit={(gradClipMaxNorm) => update({ gradClipMaxNorm })}
          ariaLabel="Gradient clipping max norm"
        />

        <label className="cr-node__field cr-node__field--checkbox nodrag nopan">
          <input
            type="checkbox"
            checked={Boolean(d.disableExtraObservables)}
            onChange={(e) => update({ disableExtraObservables: e.target.checked })}
          />
          <span>Disable extra observables (keep loss + accuracy only)</span>
        </label>

        <label className="cr-node__field cr-node__field--checkbox nodrag nopan">
          <input
            type="checkbox"
            checked={d.liveVizUpdates !== false}
            onChange={(e) => update({ liveVizUpdates: e.target.checked })}
          />
          <span>Live viz updates (stream curves during training)</span>
        </label>

        <details
          className="cr-trainer-advanced nodrag nopan"
          onDoubleClick={(event) => event.stopPropagation()}
        >
          <summary className="cr-trainer-advanced__summary nodrag nopan">
            <span>Advanced</span>
          </summary>
          <div className="cr-trainer-advanced__body">
            <ComfyIntField
              label="train seed"
              value={d.trainSeed ?? -1}
              min={-1}
              title="-1 derives training randomness from the dataset seed."
              onCommit={(trainSeed) => update({ trainSeed })}
              ariaLabel="Trainer random seed"
            />

            <DiscreteMultiSelect<TrainerLogSchedule>
              label="logging"
              options={TRAINER_LOG_SCHEDULE_OPTIONS}
              value={logSchedule}
              singleSelect
              ariaLabel="Trainer logging schedule"
              onCommit={(next) => {
                const value = Array.isArray(next) ? next[0] : next;
                update({ logSchedule: value ?? "fixed_interval" });
              }}
            />

            {logSchedule === "idnns_logspace" ? (
              <ComfyIntField
                label="log-space samples"
                value={d.logSamples ?? 1800}
                min={1}
                title="Requested base-2 log-space epoch samples; duplicate integer epochs are removed."
                onCommit={(logSamples) => update({ logSamples })}
                ariaLabel="IDNNs log-space sample count"
              />
            ) : null}

            <DiscreteMultiSelect<TrainerLogAggregation>
              label="log aggregation"
              options={TRAINER_LOG_AGGREGATION_OPTIONS}
              value={logAggregation}
              singleSelect
              ariaLabel="Trainer log aggregation"
              onCommit={(next) => {
                const value = Array.isArray(next) ? next[0] : next;
                update({ logAggregation: value ?? "last_batch" });
              }}
            />

            <DiscreteMultiSelect<TrainerLogTiming>
              label="log timing"
              options={TRAINER_LOG_TIMING_OPTIONS}
              value={logTiming}
              singleSelect
              ariaLabel="Trainer log timing"
              onCommit={(next) => {
                const value = Array.isArray(next) ? next[0] : next;
                update({ logTiming: value ?? "post_update" });
              }}
            />

            <DiscreteMultiSelect<TrainerTestEvaluation>
              label="test evaluation"
              options={TRAINER_TEST_EVALUATION_OPTIONS}
              value={testEvaluation}
              singleSelect
              ariaLabel="Trainer test evaluation cadence"
              onCommit={(next) => {
                const value = Array.isArray(next) ? next[0] : next;
                update({ testEvaluation: value ?? "log_ticks" });
              }}
            />

            <DiscreteMultiSelect<TrainerMinibatchSampling>
              label="minibatch sampling"
              options={TRAINER_MINIBATCH_SAMPLING_OPTIONS}
              value={minibatchSampling}
              singleSelect
              ariaLabel="Trainer minibatch sampling"
              onCommit={(next) => {
                const value = Array.isArray(next) ? next[0] : next;
                update({ minibatchSampling: value ?? "independent_step" });
              }}
            />

            <ComfyIntField
              label="minibatch seed"
              value={d.minibatchSeed ?? -1}
              min={-1}
              title="-1 derives the minibatch stream from the trainer seed."
              onCommit={(minibatchSeed) => update({ minibatchSeed })}
              ariaLabel="Trainer minibatch seed"
            />
          </div>
        </details>

        {trainDisp.error ? <p className="cr-trainer-train-err">{trainDisp.error}</p> : null}
        {d.lastTrainLoopSeconds != null && Number.isFinite(d.lastTrainLoopSeconds) ? (
          <p
            className="cr-trainer-loop-time"
            title="Wall time in zero_grad → forward → backward → optimizer.step for the last run (excludes batch materialization, metric logging, and checkpoint I/O)."
          >
            Training loop (last run): {formatTrainLoopSeconds(d.lastTrainLoopSeconds)}
          </p>
        ) : null}
      </div>
    </div>
    {hessianOversizeModal}
    </>
  );
}
