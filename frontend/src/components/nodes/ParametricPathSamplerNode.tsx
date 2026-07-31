import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { serializeExecutionGraphForTarget } from "../../graph/trainSeriesPlan";
import { appendParametricPathToCurveTables } from "../../graph/curveSeriesParametricAppend";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultParametricPathSamplerData,
  type ParametricPathSamplerNodeData,
} from "./parametricPathSamplerDefaults";
import {
  COMPUTE_MODE_OPTIONS,
  type ComputeModeUi,
  computeDeviceFromModeUi,
  computeModeUiFromDevice,
  defaultLocalCudaDevice,
  normalizeComputeModeUi,
  remoteGpuFromModeUi,
} from "./trainerComputeDevice";

function patchData(
  id: string,
  patch: Partial<ParametricPathSamplerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const prev = { ...defaultParametricPathSamplerData(), ...(n.data as Partial<ParametricPathSamplerNodeData>) };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function ParametricPathSamplerNode({ id, data, selected }: NodeProps) {
  const d = { ...defaultParametricPathSamplerData(), ...(data as Partial<ParametricPathSamplerNodeData>) };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [busy, setBusy] = useState(false);
  const computeModeUi = useMemo(
    () => computeModeUiFromDevice(d.computeDevice, d.remoteGpu),
    [d.computeDevice, d.remoteGpu],
  );

  const update = useCallback(
    (patch: Partial<ParametricPathSamplerNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  const runSampler = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    update({ runError: null });
    try {
      const g = serializeExecutionGraphForTarget(getNodes(), getEdges(), id);
      const res = await fetch("/api/parametric_path_sampler", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sampler_node_id: id,
          nodes: g.nodes,
          edges: g.edges,
        }),
      });
      if (!res.ok) {
        let detail = `Sampler failed (${res.status}).`;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j.detail != null) {
            detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          const txt = await res.text().catch(() => "");
          if (txt) detail = txt;
        }
        throw new Error(detail);
      }
      const j = (await res.json()) as {
        alphaSeries?: number[];
        series?: Array<{ metricId: string; label: string; values: number[] }>;
        summary?: string;
      };
      const series = Array.isArray(j.series) ? j.series : [];
      update({
        alphaSeries: Array.isArray(j.alphaSeries) ? j.alphaSeries : [],
        valueSeries: series[0]?.values ?? [],
        runSummary: j.summary ?? null,
        runError: null,
      });
      appendParametricPathToCurveTables(setNodes, getNodes(), getEdges(), id, {
        alphaSeries: j.alphaSeries ?? [],
        series,
      });
    } catch (e) {
      update({ runError: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }, [busy, getEdges, getNodes, id, setNodes, update]);

  return (
    <div
      className={`cr-node cr-node--parametric-path-sampler${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--activation">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">
            {readInstanceTitle(data, "Parametric path sampler")}
          </span>
          <button
            type="button"
            className="cr-activation-collect-btn nodrag nopan"
            disabled={busy}
            onClick={() => void runSampler()}
          >
            {busy ? "…" : "Run"}
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Parametric path sampler sockets">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="checkpoint_sb"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle"
                style={{ top: "18%" }}
              />
              <span className="cr-trainer-socket-label">checkpoint SB</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="checkpoint_lb"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle"
                style={{ top: "32%" }}
              />
              <span className="cr-trainer-socket-label">checkpoint LB</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model"
                style={{ top: "46%" }}
              />
              <span className="cr-trainer-socket-label">model</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
                style={{ top: "60%" }}
              />
              <span className="cr-trainer-socket-label">dataset</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="loss"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--loss"
                style={{ top: "74%" }}
              />
              <span className="cr-trainer-socket-label">loss</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">stream</span>
              <Handle
                type="source"
                position={Position.Right}
                id="stream"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-handle"
              />
            </div>
          </div>
        </div>
        <DiscreteMultiSelect<ComputeModeUi>
          label="compute device"
          options={COMPUTE_MODE_OPTIONS}
          value={computeModeUi}
          singleSelect
          ariaLabel="Parametric path compute device"
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
            });
          }}
        />
        <ComfyFloatField
          label="α min"
          value={d.alphaMin}
          onCommit={(n) => update({ alphaMin: n })}
          ariaLabel="Alpha minimum"
        />
        <ComfyFloatField
          label="α max"
          value={d.alphaMax}
          onCommit={(n) => update({ alphaMax: n })}
          ariaLabel="Alpha maximum"
        />
        <ComfyIntField
          label="α steps"
          value={d.alphaSteps}
          min={2}
          max={500}
          onCommit={(n) => update({ alphaSteps: Math.max(2, n) })}
          ariaLabel="Alpha steps"
        />
        <p className="cr-observable-hint">
          Each Run samples train/test loss &amp; acc along w(α)=w<sub>SB</sub>+α(w<sub>LB</sub>−w<sub>SB</sub>).
          Trainable parameters are interpolated; evaluation uses per-batch BatchNorm statistics,
          matching the paper protocol.
          AutoDL GPU runs on remote via SSH.
        </p>
        {d.runSummary ? <p className="cr-observable-hint">{d.runSummary}</p> : null}
        {d.runError ? <p className="cr-node-error">{d.runError}</p> : null}
      </div>
    </div>
  );
}
