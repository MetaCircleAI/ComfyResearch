import { useCallback, useState, type ReactNode } from "react";
import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { diffusionReproRequestGraph } from "../../graph/diffusionReproRequestGraph";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";

type Data = Record<string, unknown>;
type Progress = { step: number; total: number } | null;

function numberText(value: unknown, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function updateNodeData(id: string, patch: Data, setNodes: (updater: (nodes: Node[]) => Node[]) => void) {
  setNodes((nodes) => nodes.map((node) => node.id === id ? { ...node, data: { ...(node.data as Data), ...patch } } : node));
}

async function readSamplerStream(
  response: Response,
  onProgress: (progress: Progress) => void,
): Promise<Data> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Sampler returned no response body.");
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as Data;
      if (event.type === "progress") {
        onProgress({ step: Number(event.step) || 0, total: Math.max(1, Number(event.total) || 1) });
      } else if (event.type === "complete") {
        return event;
      } else if (event.type === "error") {
        throw new Error(typeof event.detail === "string" ? event.detail : "Sampling failed.");
      }
    }
    if (done) break;
  }
  throw new Error("Sampler stream ended without a result.");
}

function useRun(id: string, endpoint: string, kind: "sampler" | "observable") {
  const { getNodes, getEdges, setNodes } = useReactFlow();
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress>(null);
  const run = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setProgress(kind === "sampler" ? { step: 0, total: 1 } : null);
    try {
      const graph = diffusionReproRequestGraph(getNodes(), getEdges(), id, kind);
      const key = kind === "sampler" ? "sampler_node_id" : "observable_node_id";
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...graph, [key]: id }),
      });
      if (!response.ok) {
        let detail = `Request failed (${response.status}).`;
        try {
          const payload = (await response.json()) as { detail?: unknown };
          if (typeof payload.detail === "string") detail = payload.detail;
        } catch {
          /* Keep the HTTP status message when a proxy returned a non-JSON body. */
        }
        throw new Error(detail);
      }
      const result = kind === "sampler"
        ? await readSamplerStream(response, setProgress)
        : await response.json() as Data;
      updateNodeData(id, { ...result, lastError: "" }, setNodes);
    } catch (error) {
      updateNodeData(id, { lastError: error instanceof Error ? error.message : String(error) }, setNodes);
    } finally {
      setProgress(null);
      setRunning(false);
    }
  }, [endpoint, getEdges, getNodes, id, kind, running, setNodes]);
  return { running, progress, run };
}

function Shell({ title, selected, children }: { title: string; selected: boolean; children: ReactNode }) {
  return <div className={`cr-node cr-node--observable-weight-l2${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}><div className="cr-node__header"><div className="cr-node__header-title">{title}</div></div><div className="cr-node__body">{children}</div></div>;
}

function ErrorMessage({ data }: { data: Data }) {
  return typeof data.lastError === "string" && data.lastError ? <p className="cr-observable-hint cr-observable-hint--error">{data.lastError}</p> : null;
}

function RunSummary({ data, running, progress, action }: { data: Data; running: boolean; progress: Progress; action: string }) {
  if (progress) return <p className="cr-observable-hint">{action}: {Math.min(progress.step, progress.total)} / {progress.total}</p>;
  if (running) return <p className="cr-observable-hint">{action}...</p>;
  if (typeof data.runId === "string" && data.runId) return <p className="cr-observable-hint">saved run: {data.runId}</p>;
  return <p className="cr-observable-hint">Ready</p>;
}

function EvidenceImages({ data, title }: { data: Data; title: string }) {
  return <>{["imageGrid", "histogramPng"].map((key) => typeof data[key] === "string" && data[key] ? <img key={key} src={data[key] as string} alt={`${title}: ${key === "imageGrid" ? "paired examples" : "distribution"}`} style={{ width: "100%", marginTop: 8 }} /> : null)}</>;
}

export function DeterministicDiffusionSamplerNode({ id, data, selected }: NodeProps) {
  const d = (data ?? {}) as Data;
  const { setNodes } = useReactFlow();
  const { running, progress, run } = useRun(id, "/api/diffusion/sampler/stream", "sampler");
  const update = useCallback((patch: Data) => updateNodeData(id, patch, setNodes), [id, setNodes]);
  return <Shell title="Deterministic sampler" selected={selected}>
    <Handle type="target" position={Position.Left} id="checkpoint" />
    <Handle type="source" position={Position.Right} id="samples" />
    <ComfyIntListField label="noise seed" values={intChoices(d.noiseSeed, 0)} min={0} ariaLabel="Deterministic sampler noise seed" onCommit={(values) => update({ noiseSeed: packIntList(values) })} />
    <ComfyIntListField label="sample count" values={intChoices(d.sampleCount, 64)} min={1} max={512} ariaLabel="Deterministic sampler sample count" onCommit={(values) => update({ sampleCount: packIntList(values) })} />
    <ComfyIntListField label="sampling steps" values={intChoices(d.numSteps, 50)} min={2} ariaLabel="Deterministic sampler DDIM steps" onCommit={(values) => update({ numSteps: packIntList(values) })} />
    <button type="button" className="cr-button nodrag nopan" disabled={running} onClick={() => void run()}>{running ? "Sampling..." : "Sample"}</button>
    <RunSummary data={d} running={running} progress={progress} action="Sampling" />
    <EvidenceImages data={{ ...d, imageGrid: d.previewGrid }} title="Generated samples" />
    <ErrorMessage data={d} />
  </Shell>;
}

function Observable({ id, data, selected, title, endpoint, inputs, children, metrics }: NodeProps & { title: string; endpoint: string; inputs: string[]; children?: ReactNode; metrics: Array<[string, unknown]> }) {
  const d = (data ?? {}) as Data;
  const { running, run } = useRun(id, endpoint, "observable");
  return <Shell title={title} selected={selected}>
    {inputs.map((input, index) => <Handle key={input} type="target" position={Position.Left} id={input} style={{ top: 38 + index * 20 }} />)}
    {children}
    <button type="button" className="cr-button nodrag nopan" disabled={running} onClick={() => void run()}>{running ? "Running..." : "Run"}</button>
    <div className="cr-observable-hint">{metrics.map(([label, value]) => <div key={label}>{label}: {numberText(value)}</div>)}</div>
    <EvidenceImages data={d} title={title} />
    <ErrorMessage data={d} />
  </Shell>;
}

export function PairedGenerationSimilarityNode(props: NodeProps) {
  const d = (props.data ?? {}) as Data;
  return <Observable {...props} title="Paired similarity" endpoint="/api/diffusion/paired-similarity" inputs={["sampler_a", "sampler_b"]} metrics={[["mean MAE", d.meanMae], ["mean MSE", d.meanMse]]} />;
}

export function RpScoreSscdNode(props: NodeProps) {
  const d = (props.data ?? {}) as Data;
  const { setNodes } = useReactFlow();
  const update = useCallback((patch: Data) => updateNodeData(props.id, patch, setNodes), [props.id, setNodes]);
  return <Observable {...props} title="RP reproducibility" endpoint="/api/diffusion/rp-score" inputs={["sampler_a", "sampler_b"]} metrics={[["RP", d.rp], ["mean similarity", d.meanSimilarity]]}>
    <ComfyFloatListField label="RP threshold" values={floatChoices(d.threshold, 0.95)} min={0} max={1} ariaLabel="RP reproducibility threshold" onCommit={(values) => update({ threshold: packFloatList(values) })} />
  </Observable>;
}

export function NearestTrainGlNode(props: NodeProps) {
  const d = (props.data ?? {}) as Data;
  const { setNodes } = useReactFlow();
  const update = useCallback((patch: Data) => updateNodeData(props.id, patch, setNodes), [props.id, setNodes]);
  return <Observable {...props} title="Nearest-train GL" endpoint="/api/diffusion/nearest-train" inputs={["generated", "train_dataset"]} metrics={[["GL score", d.glScore]]}>
    <ComfyFloatListField label="GL threshold" values={floatChoices(d.glThreshold, 0.95)} min={0} max={1} ariaLabel="Nearest-train GL threshold" onCommit={(values) => update({ glThreshold: packFloatList(values) })} />
    {typeof d.backend === "string" && d.backend ? <p className="cr-observable-hint">backend: {d.backend}</p> : null}
  </Observable>;
}
