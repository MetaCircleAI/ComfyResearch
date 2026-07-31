import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { useFlowSurface } from "../../context/FlowSurfaceContext";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { readNdjsonTrainStream, type TrainStreamProgress } from "../../graph/readNdjsonTrainStream";
import { flushCheckpointApplyTrainerVizAndHydrateTv0d } from "../../graph/trainerTrainCompleteCommit";
import { serializeExecutionGraphForTarget } from "../../graph/trainSeriesPlan";
import { ComfyIntField } from "./comfyNumberFields";
import { defaultCrlTrainerData, type CrlTrainerNodeData } from "./crlTrainerDefaults";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";

function patchCrlTrainerData(
  id: string,
  patch: Partial<CrlTrainerNodeData>,
  setNodes: Dispatch<SetStateAction<Node[]>>,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultCrlTrainerData();
      const cur = (n.data ?? {}) as Partial<CrlTrainerNodeData>;
      const prev: CrlTrainerNodeData = {
        trainingSteps: cur.trainingSteps ?? def.trainingSteps,
        logFrequency: cur.logFrequency ?? def.logFrequency,
        computeDevice: cur.computeDevice ?? def.computeDevice,
        batchSize: cur.batchSize ?? def.batchSize,
        unrollLength: cur.unrollLength ?? def.unrollLength,
        sgdStepsPerTrainStep: cur.sgdStepsPerTrainStep ?? def.sgdStepsPerTrainStep,
        gamma: cur.gamma ?? def.gamma,
        logsumexpPenaltyCoeff: cur.logsumexpPenaltyCoeff ?? def.logsumexpPenaltyCoeff,
        entropyParam: cur.entropyParam ?? def.entropyParam,
        disableEntropy: cur.disableEntropy ?? def.disableEntropy,
        maxReplayChunks: cur.maxReplayChunks ?? def.maxReplayChunks,
        seed: cur.seed ?? def.seed,
        lossHistory: cur.lossHistory,
        testLossHistory: cur.testLossHistory,
        stepTicks: cur.stepTicks,
        observableMetricHistories: cur.observableMetricHistories,
        memoryCheckpoint_b64: cur.memoryCheckpoint_b64,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function CrlInRow({ handleId, label, fullWidth }: { handleId: string; label: string; fullWidth?: boolean }) {
  return (
    <div className={`cr-trainer-io-row__leftwrap${fullWidth ? " cr-trainer-io-row__leftwrap--full" : ""}`}>
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

function CrlOutRow({
  handleId,
  label,
  accent,
}: {
  handleId: string;
  label: string;
  accent: "model" | "loss" | "observable";
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

export function CrlTrainerNode({ id, data, selected }: NodeProps) {
  const def = defaultCrlTrainerData();
  const raw = (data ?? {}) as Partial<CrlTrainerNodeData>;
  const d: CrlTrainerNodeData = {
    trainingSteps: raw.trainingSteps ?? def.trainingSteps,
    logFrequency: raw.logFrequency ?? def.logFrequency,
    computeDevice: raw.computeDevice ?? def.computeDevice,
    batchSize: raw.batchSize ?? def.batchSize,
    unrollLength: raw.unrollLength ?? def.unrollLength,
    sgdStepsPerTrainStep: raw.sgdStepsPerTrainStep ?? def.sgdStepsPerTrainStep,
    gamma: raw.gamma ?? def.gamma,
    logsumexpPenaltyCoeff: raw.logsumexpPenaltyCoeff ?? def.logsumexpPenaltyCoeff,
    entropyParam: raw.entropyParam ?? def.entropyParam,
    disableEntropy: raw.disableEntropy ?? def.disableEntropy,
    maxReplayChunks: raw.maxReplayChunks ?? def.maxReplayChunks,
    seed: raw.seed ?? def.seed,
    lossHistory: raw.lossHistory,
    testLossHistory: raw.testLossHistory,
    stepTicks: raw.stepTicks,
    observableMetricHistories: raw.observableMetricHistories,
    memoryCheckpoint_b64: raw.memoryCheckpoint_b64,
  };

  const rf = useReactFlow();
  const flowSurface = useFlowSurface();
  const setNodes = flowSurface?.applyNodes ?? rf.setNodes;
  const { getNodes, getEdges } = rf;
  const update = (patch: Partial<CrlTrainerNodeData>) => patchCrlTrainerData(id, patch, setNodes);
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainProgressPct, setTrainProgressPct] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const progressHideToRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (progressHideToRef.current) {
        clearTimeout(progressHideToRef.current);
        progressHideToRef.current = null;
      }
    },
    [],
  );

  const runTrain = useCallback(async () => {
    setTrainError(null);
    setTrainLoading(true);
    setTrainProgressPct(0);
    if (progressHideToRef.current) {
      clearTimeout(progressHideToRef.current);
      progressHideToRef.current = null;
    }
    const ac = new AbortController();
    abortControllerRef.current = ac;
    const trainGraph = serializeExecutionGraphForTarget(
      getNodes().filter((n) => String(n.type) !== "graph_assist_failure_overlay"),
      getEdges(),
      id,
    );
    const nodesPayload = trainGraph.nodes;
    const edgesForCurrentGraph = trainGraph.edges;
    const onProgress = (raw: TrainStreamProgress) => {
      if (raw.type !== "progress") return;
      const total = Math.max(1, raw.total);
      const step = Math.min(Math.max(0, raw.step), total);
      setTrainProgressPct(Math.min(100, Math.round((step / total) * 100)));
    };
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trainer_node_id: id,
          nodes: nodesPayload,
          edges: edgesForCurrentGraph,
        }),
        signal: ac.signal,
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
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const { complete, aborted } = await readNdjsonTrainStream(reader, onProgress);
      if (aborted || ac.signal.aborted) {
        setTrainProgressPct(0);
        return;
      }
      if (!complete) throw new Error("Train stream ended without complete");
      await flushCheckpointApplyTrainerVizAndHydrateTv0d({
        isCrl: true,
        setNodes,
        getNodes,
        getEdges,
        trainerNodeId: id,
        wires: edgesForCurrentGraph,
        payload: complete,
        extraRefreshDelaysMs: [250, 900],
      });
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setTrainProgressPct(0);
      } else {
        setTrainError(e instanceof Error ? e.message : String(e));
        setTrainProgressPct(0);
      }
    } finally {
      setTrainLoading(false);
      abortControllerRef.current = null;
      progressHideToRef.current = window.setTimeout(() => {
        progressHideToRef.current = null;
        setTrainProgressPct(0);
      }, 420);
    }
  }, [getEdges, getNodes, id, setNodes]);

  const showTrainProgress = trainLoading || trainProgressPct > 0;

  return (
    <div
      className={`cr-node cr-node--trainer cr-node--crl-trainer${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-trainer)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        {showTrainProgress ? (
          <div className="cr-trainer-header-progress-with-grip">
            <div
              className="cr-trainer-progress nodrag nopan"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(trainProgressPct)}
              aria-valuetext={`${Math.min(100, Math.round(trainProgressPct))}%`}
              aria-label="CRL training progress"
            >
              <div
                className="cr-trainer-progress__fill"
                style={{ width: `${Math.min(100, trainProgressPct)}%` }}
              />
              <span className="cr-trainer-progress__label">
                {Math.min(100, Math.round(trainProgressPct))}%
              </span>
            </div>
          </div>
        ) : null}
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <div className="cr-node__header-main">{readInstanceTitle(d as Record<string, unknown>, "CRL trainer")}</div>
          <div className="cr-crl-trainer-header-tail nodrag nopan">
            <button type="button" className="cr-trainer-train-btn nodrag nopan" disabled={trainLoading} onClick={() => void runTrain()}>
              {trainLoading ? "…" : "Train"}
            </button>
            <NodeSpecHeaderActions
              nodeId={id}
              graphNodeType="crl_trainer"
              generatedCode={`# CRL training — server comfy_research/engine/crl_run.py\n`}
              codeKind="model"
              infoTitle="CRL trainer"
              infoText="Contrastive RL in PyTorch (InfoNCE critic, Gaussian actor). Wire env, crl_residual_mlp, and Adam."
            />
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="CRL trainer IO">
          <div className="cr-trainer-io-row">
            <CrlInRow handleId="env" label="env" fullWidth />
          </div>
          <div className="cr-trainer-io-row">
            <CrlInRow handleId="model" label="model" />
            <CrlOutRow handleId="checkpoint" label="model ckpt" accent="model" />
          </div>
          <div className="cr-trainer-io-row">
            <CrlInRow handleId="optimizer" label="optimizer" fullWidth />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap" />
            <CrlOutRow handleId="loss_results" label="loss" accent="loss" />
          </div>
          <div className="cr-trainer-io-row">
            <CrlInRow handleId="observables" label="observables" />
            <CrlOutRow handleId="observable_results" label="observable" accent="observable" />
          </div>
        </div>

        <ComfyIntField label="training steps" value={d.trainingSteps} min={1} ariaLabel="Training steps" onCommit={(trainingSteps) => update({ trainingSteps })} />
        <ComfyIntField label="log frequency" value={d.logFrequency} min={1} ariaLabel="Log frequency" onCommit={(logFrequency) => update({ logFrequency })} />
        <ComfyIntField label="batch size" value={d.batchSize} min={4} ariaLabel="Batch size" onCommit={(batchSize) => update({ batchSize })} />
        <ComfyIntField label="unroll length" value={d.unrollLength} min={4} ariaLabel="Unroll length" onCommit={(unrollLength) => update({ unrollLength })} />
        <ComfyIntField
          label="SGD steps / train step"
          value={d.sgdStepsPerTrainStep}
          min={1}
          ariaLabel="SGD steps per train step"
          onCommit={(sgdStepsPerTrainStep) => update({ sgdStepsPerTrainStep })}
        />
        <ComfyIntField label="max replay chunks" value={d.maxReplayChunks} min={10} ariaLabel="Max replay chunks" onCommit={(maxReplayChunks) => update({ maxReplayChunks })} />
        <ComfyIntField label="seed" value={d.seed} min={0} ariaLabel="Trainer seed" onCommit={(seed) => update({ seed })} />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush">
            <span className="cr-comfy-widget__label">Compute device</span>
            <div className="cr-comfy-widget__control-col">
              <select
                className="cr-input cr-comfy-widget__control"
                aria-label="Compute device"
                value={d.computeDevice}
                onChange={(e) => update({ computeDevice: e.target.value })}
              >
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
                <option value="mps">mps</option>
              </select>
            </div>
          </div>
        </div>
        {trainError ? <p className="cr-trainer-train-err">{trainError}</p> : null}
      </div>
    </div>
  );
}
