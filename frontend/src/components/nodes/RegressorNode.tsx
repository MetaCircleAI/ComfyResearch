import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import katex from "katex";
import { useCallback, useEffect, useMemo, useState } from "react";
import { getObservableVizVariant } from "../../graph/observableVizVariant";
import {
  orderedSelectedTensorKeysForPicker,
  resolveUpstreamTensor,
  tensorChoicesForTensorsInput,
  type FlowEdge,
  type FlowNodeBare,
} from "../../graph/resolveUpstreamTensor";
import type { ObservableVizUserNodeData } from "./observableVizUserDefaults";
import type { ObservableVizWeightL1NodeData } from "./observableVizWeightL1Defaults";
import type { ObservableVizWeightL2NodeData } from "./observableVizWeightL2Defaults";
import type { ObservableVizReluNonlinearNodeData } from "./observableVizReluNonlinearDefaults";
import type { ObservableVizHessianEigenvaluesNodeData } from "./observableVizHessianEigenvaluesDefaults";
import { tensorSelectorOutputIndexFromSourceHandle, type TensorSelectorNodeData } from "./tensorSelectorDefaults";
import type { TrainerNodeData } from "./trainerDefaults";
import type { TrainingVisualizationNodeData } from "./trainingVisualizationDefaults";
import { resolveTableVizTensorSeries } from "../../graph/tableVizRegressor";
import { fetchActivationTensorAsOk } from "../../graph/fetchActivationTensor";
import { readActivationManifest, readActivationRunId } from "../../graph/activationNodeData";
import type { ActivationNodeData } from "./activationDefaults";
import { defaultRegressorData, type RegressorNodeData } from "./regressorDefaults";
import { fitExponential, fitPowerLaw, type FitResult } from "../../graph/curveFits";

type SeriesResolved =
  | { kind: "ok"; x: number[]; y: number[]; sourceSummary: string }
  | { kind: "none"; detail: string }
  | { kind: "lazy_activation_line"; runId: string; repId: string; sourceSummary: string };

function filterByZoom(x: number[], y: number[], zoomXMin?: number, zoomXMax?: number): { x: number[]; y: number[] } {
  if (x.length !== y.length) return { x: [], y: [] };
  if (typeof zoomXMin !== "number" || typeof zoomXMax !== "number") return { x, y };
  const min = Math.min(zoomXMin, zoomXMax);
  const max = Math.max(zoomXMin, zoomXMax);
  const xf: number[] = [];
  const yf: number[] = [];
  for (let i = 0; i < x.length; i++) {
    if (x[i]! < min || x[i]! > max) continue;
    xf.push(x[i]!);
    yf.push(y[i]!);
  }
  return { x: xf, y: yf };
}

function resolveSeriesFromSource(nodes: Node[], edges: FlowEdge[], srcId: string, sh: string, tsKey = ""): SeriesResolved {
  const src = nodes.find((n) => n.id === srcId);
  if (!src) return { kind: "none", detail: "Source node missing." };

  if (src.type === "training_visualization") {
    const d = (src.data ?? {}) as Partial<TrainingVisualizationNodeData>;
    const x = (d.stepTicks ?? []).map(Number);
    const pick = sh === "out_tensor_list" && tsKey ? tsKey : "train_loss";
    const y = (pick === "test_loss" ? d.testLossHistory ?? [] : d.lossHistory ?? []).map(Number);
    const f = filterByZoom(x, y, d.zoomXMin, d.zoomXMax);
    if (f.x.length < 3) return { kind: "none", detail: "Training viz needs at least 3 visible points." };
    return { kind: "ok", x: f.x, y: f.y, sourceSummary: `Training viz · ${pick === "test_loss" ? "test" : "train"} loss` };
  }

  if (src.type === "observable_viz" || src.type === "observable_accuracy") {
    const v = getObservableVizVariant(src);
    const detailPrefix = "Observable viz needs at least 3 visible points.";
    if (v === "weight_l2") {
      const d = (src.data ?? {}) as Partial<ObservableVizWeightL2NodeData>;
      const f = filterByZoom((d.stepTicks ?? []).map(Number), (d.valueHistory ?? []).map(Number), d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz" };
    }
    if (v === "weight_l1") {
      const d = (src.data ?? {}) as Partial<ObservableVizWeightL1NodeData>;
      const f = filterByZoom((d.stepTicks ?? []).map(Number), (d.valueHistory ?? []).map(Number), d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz" };
    }
    if (v === "relu_nonlinear") {
      const d = (src.data ?? {}) as Partial<ObservableVizReluNonlinearNodeData>;
      const f = filterByZoom((d.stepTicks ?? []).map(Number), (d.valueHistory ?? []).map(Number), d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz" };
    }
    if (v === "user") {
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const f = filterByZoom((d.stepTicks ?? []).map(Number), (d.valueHistory ?? []).map(Number), d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz" };
    }
    if (v === "capacity") {
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const f = filterByZoom((d.stepTicks ?? []).map(Number), (d.valueHistory ?? []).map(Number), d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz" };
    }
    if (v === "accuracy") {
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const x = (d.stepTicks ?? []).map(Number);
      const useTest = tsKey === "test_accuracy";
      const y = (useTest ? d.testValueHistory ?? [] : d.valueHistory ?? []).map(Number);
      const f = filterByZoom(x, y, d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      const label = src.type === "observable_accuracy" ? "Accuracy viz" : "Observable viz";
      return {
        kind: "ok",
        x: f.x,
        y: f.y,
        sourceSummary: `${label} · ${useTest ? "test" : "train"} acc`,
      };
    }
    if (v === "hessian_eigenvalues") {
      const d = (src.data ?? {}) as Partial<ObservableVizHessianEigenvaluesNodeData>;
      const y0 = (d.valueHistories?.[0] ?? []).map(Number);
      const f = filterByZoom((d.stepTicks ?? []).map(Number), y0, d.zoomXMin, d.zoomXMax);
      if (f.x.length < 3) return { kind: "none", detail: detailPrefix };
      return { kind: "ok", x: f.x, y: f.y, sourceSummary: "Observable viz · Hessian λ₁" };
    }
    return { kind: "none", detail: "This observable viz variant is not supported for regression." };
  }
  if (src.type === "trainer") {
    const d = (src.data ?? {}) as Partial<TrainerNodeData>;
    const x = (d.stepTicks ?? []).map(Number);
    const y = (sh === "observable_results" ? [] : d.lossHistory ?? []).map(Number);
    if (x.length < 3 || y.length !== x.length) {
      return { kind: "none", detail: "Trainer needs a line series with at least 3 points." };
    }
    return { kind: "ok", x, y, sourceSummary: "Trainer · train loss" };
  }
  if (src.type === "activation" && sh === "tensor_list") {
    const actRaw = (src.data ?? {}) as Record<string, unknown>;
    const actData = actRaw as Partial<ActivationNodeData>;
    const runId = readActivationRunId(actRaw);
    const manifest = readActivationManifest(actRaw);
    if (runId && manifest?.[tsKey]?.shape) {
      const shape = manifest[tsKey]!.shape.map((x) => Number(x));
      if (shape.length !== 1) {
        return { kind: "none", detail: "Regressor expects a 1-D tensor from activation." };
      }
      return {
        kind: "lazy_activation_line",
        runId,
        repId: tsKey,
        sourceSummary: `Activation · ${tsKey}`,
      };
    }
    const collected = actData.collectedActivations;
    const legacy = collected?.[tsKey];
    if (legacy && Array.isArray(legacy.shape) && Array.isArray(legacy.values)) {
      const shape = legacy.shape.map((x) => Number(x));
      if (shape.length !== 1) {
        return { kind: "none", detail: "Regressor expects a 1-D tensor from activation." };
      }
      const y = legacy.values.map(Number);
      return {
        kind: "ok",
        x: y.map((_, i) => i),
        y,
        sourceSummary: `Activation · ${tsKey}`,
      };
    }
    return { kind: "none", detail: "Activation has no collected tensors for this key yet." };
  }

  if (src.type === "dimension_permutator") {
    const t = resolveUpstreamTensor(nodes as FlowNodeBare[], edges, src.id, "tensor_in");
    if (t.kind !== "ok") {
      return { kind: "none", detail: t.kind === "none" ? t.detail : "Tensor not ready for regression." };
    }
    if (t.shape.length !== 1 || t.values.length < 3) {
      return {
        kind: "none",
        detail: "Regressor expects a 1-D tensor (≥3 elements) after the permutator.",
      };
    }
    const y = t.values;
    const x = y.map((_, i) => i);
    return { kind: "ok", x, y, sourceSummary: t.sourceSummary };
  }

  if (src.type === "tensor_slicing") {
    const t = resolveUpstreamTensor(nodes as FlowNodeBare[], edges, src.id, "tensor");
    if (t.kind !== "ok") {
      return { kind: "none", detail: t.kind === "none" ? t.detail : "Tensor not ready for regression." };
    }
    if (t.shape.length !== 1 || t.values.length < 3) {
      return {
        kind: "none",
        detail: "Regressor expects a 1-D tensor (>=3 elements) after slicing.",
      };
    }
    const y = t.values;
    const x = y.map((_, i) => i);
    return { kind: "ok", x, y, sourceSummary: t.sourceSummary };
  }

  if (src.type === "tensor_selector") {
    const td = (src.data ?? {}) as Partial<TensorSelectorNodeData>;
    const choices = tensorChoicesForTensorsInput(nodes as FlowNodeBare[], edges, src.id);
    const ordered = orderedSelectedTensorKeysForPicker(td, choices);
    const idx = tensorSelectorOutputIndexFromSourceHandle(sh);
    const key = (ordered[idx] ?? ordered[0] ?? "").trim();
    if (!key) return { kind: "none", detail: "Pick a tensor in the tensor selector first." };
    const inEdge = edges.find((e) => e.target === src.id && (e.targetHandle ?? "") === "tensor_list");
    if (!inEdge?.source) return { kind: "none", detail: "Tensor selector must be fed by a tensor list source." };
    return resolveSeriesFromSource(nodes, edges, inEdge.source, inEdge.sourceHandle ?? "", key);
  }
  if (src.type === "table_viz" && sh === "tensor") {
    const r = resolveTableVizTensorSeries(nodes, edges, src.id, tsKey || null);
    if (r.kind === "ok") {
      return { kind: "ok", x: r.x, y: r.y, sourceSummary: r.sourceSummary };
    }
    return { kind: "none", detail: r.detail };
  }
  return { kind: "none", detail: "Connect from a line-plot viz output (or a selector fed by one)." };
}

function resolveRegressorSeries(nodes: Node[], edges: FlowEdge[], nodeId: string): SeriesResolved {
  const e = edges.find((x) => x.target === nodeId && x.targetHandle === "tensor");
  if (!e?.source) return { kind: "none", detail: "Connect a line-plot tensor to run regression." };
  return resolveSeriesFromSource(nodes, edges, e.source, e.sourceHandle ?? "");
}

function RegressorFormulaMath({ latex }: { latex: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        throwOnError: false,
        displayMode: false,
        output: "html",
      });
    } catch {
      return "";
    }
  }, [latex]);
  if (!html.trim()) {
    return (
      <code className="cr-regressor-katex-fallback" title={latex}>
        {latex}
      </code>
    );
  }
  return (
    <span
      className="cr-regressor-katex"
      title={latex}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/** Scientific notation with 4 significant digits (mantissa uses `toExponential(3)`). */
function formatSci4(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0.000e+0";
  return v.toExponential(3);
}

export function RegressorNode({ id, data, selected }: NodeProps) {
  const def = defaultRegressorData();
  const raw = (data ?? {}) as Partial<RegressorNodeData>;
  const d: RegressorNodeData = { fitNonce: raw.fitNonce ?? def.fitNonce };
  const { setNodes } = useReactFlow();

  const bumpFit = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const cur = { ...defaultRegressorData(), ...(n.data as Partial<RegressorNodeData>) };
        return { ...n, data: { ...cur, fitNonce: (cur.fitNonce ?? 0) + 1 } };
      }),
    );
  }, [id, setNodes]);

  const baseSeries = useStore(
    useCallback(
      (state) => resolveRegressorSeries(state.nodes as Node[], state.edges as FlowEdge[], id),
      [id],
    ),
  );

  const [lazySeries, setLazySeries] = useState<SeriesResolved | null>(null);
  const lazyRunId = baseSeries.kind === "lazy_activation_line" ? baseSeries.runId : "";
  const lazyRepId = baseSeries.kind === "lazy_activation_line" ? baseSeries.repId : "";
  const lazySummary =
    baseSeries.kind === "lazy_activation_line" ? baseSeries.sourceSummary : "";
  useEffect(() => {
    if (!lazyRunId || !lazyRepId) {
      setLazySeries(null);
      return;
    }
    let cancelled = false;
    setLazySeries(null);
    fetchActivationTensorAsOk({
      kind: "lazy_activation",
      runId: lazyRunId,
      repId: lazyRepId,
      shape: [0],
      sourceSummary: lazySummary,
    }).then((r) => {
      if (cancelled) return;
      if (r.kind !== "ok") {
        setLazySeries({ kind: "none", detail: r.detail });
        return;
      }
      if (r.rank !== 1) {
        setLazySeries({ kind: "none", detail: "Regressor expects a 1-D tensor from activation." });
        return;
      }
      const y = r.values;
      setLazySeries({
        kind: "ok",
        x: y.map((_, i) => i),
        y,
        sourceSummary: r.sourceSummary,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [lazyRunId, lazyRepId, lazySummary]);

  const resolved: SeriesResolved = useMemo(() => {
    if (baseSeries.kind !== "lazy_activation_line") return baseSeries;
    if (lazySeries) return lazySeries;
    return { kind: "none", detail: "Loading activation tensor…" };
  }, [baseSeries, lazySeries]);

  const fits = useMemo(() => {
    void d.fitNonce;
    if (resolved.kind !== "ok") return [] as FitResult[];
    const out: FitResult[] = [];
    const exp = fitExponential(resolved.x, resolved.y);
    if (exp) out.push(exp);
    const pow = fitPowerLaw(resolved.x, resolved.y);
    if (pow) out.push(pow);
    return out;
  }, [resolved, d.fitNonce]);

  return (
    <div
      className={`cr-node cr-node--regressor${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis, var(--cr-accent-tensor))" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Regressor</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={bumpFit}>
            Run
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Regressor tensor input">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor</span>
            </div>
          </div>
        </div>

        {resolved.kind === "ok" ? (
          <div className="cr-regressor-table-wrap nodrag nopan">
            <p className="cr-regressor-meta">
              source: {resolved.sourceSummary} · points: {resolved.x.length}
            </p>
            <table className="cr-regressor-table">
              <thead>
                <tr>
                  <th>model</th>
                  <th>formula</th>
                  <th>coefficients</th>
                  <th className="cr-regressor-table-th-r2">
                    R<sup>2</sup>
                  </th>
                </tr>
              </thead>
              <tbody>
                {fits.map((fit) => (
                  <tr key={fit.name}>
                    <td>{fit.name}</td>
                    <td className="cr-regressor-table-td-formula">
                      <RegressorFormulaMath latex={fit.latex} />
                    </td>
                    <td>
                      {Object.entries(fit.params)
                        .map(([k, v]) => `${k}=${formatSci4(v)}`)
                        .join(", ")}
                    </td>
                    <td className="cr-regressor-table-td-r2">{formatSci4(fit.r2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="cr-tensor-viz__hint">{resolved.detail}</p>
        )}
      </div>
    </div>
  );
}
