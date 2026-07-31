import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo } from "react";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { resolveUpstreamTensor, type FlowEdge, type FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import type { TensorSelectorNodeData } from "./tensorSelectorDefaults";
import { tensorSelectorOutputIndexFromSourceHandle } from "./tensorSelectorDefaults";
import type { TrainingVisualizationNodeData } from "./trainingVisualizationDefaults";
import type { TrainerNodeData } from "./trainerDefaults";
import {
  SL_CHART_H,
  SL_CHART_W,
  SL_PAD_B,
  SL_PAD_L,
  SL_PAD_T,
  SL_TICK_LEN,
  SL_Y_LABEL_X,
  slBuildSeriesPathScalar,
  slComputeBoundsScalar,
  slFormatXTickScalar,
  slFormatYTick,
  slGenerateXTicksScalar,
  slGenerateYTicks,
  slInnerBottom,
  slInnerRight,
  slXToPx,
  slYToPx,
} from "./scalarLineChartShared";
import {
  defaultDerivativeCurveData,
  type DerivativeCurveNodeData,
  type DerivativeOrder,
} from "./derivativeCurveDefaults";

const DERIVATIVE_ORDER_OPTIONS: { id: DerivativeOrder; label: string }[] = [
  { id: "1", label: "1st" },
  { id: "2", label: "2nd" },
  { id: "3", label: "3rd" },
  { id: "4", label: "4th" },
  { id: "5", label: "5th" },
];

function patchDerivativeCurveData(
  id: string,
  patch: Partial<DerivativeCurveNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = defaultDerivativeCurveData((n.data ?? {}) as Partial<DerivativeCurveNodeData>);
      const next = { ...cur, ...patch };
      if (
        next.order === cur.order &&
        next.logScaleX === cur.logScaleX &&
        next.logScaleY === cur.logScaleY &&
        next.lastError === cur.lastError &&
        next.outputTensor === cur.outputTensor
      ) {
        return n;
      }
      return { ...n, data: next };
    }),
  );
}

function finiteStepTicks(raw: unknown): number[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((v) => Number(v)).filter((v) => Number.isFinite(v));
}

function estimateStepSpacing(steps: number[] | null): number {
  if (!steps || steps.length < 2) return 1;
  const deltas: number[] = [];
  for (let i = 1; i < steps.length; i += 1) {
    const d = steps[i]! - steps[i - 1]!;
    if (Number.isFinite(d) && d > 0) deltas.push(d);
  }
  if (deltas.length === 0) return 1;
  deltas.sort((a, b) => a - b);
  return deltas[Math.floor(deltas.length / 2)] ?? 1;
}

function stepTicksFromTrainingLikeNode(n: Node | undefined): number[] | null {
  if (!n) return null;
  if (n.type === "training_visualization") {
    const d = (n.data ?? {}) as Partial<TrainingVisualizationNodeData>;
    const ticks = finiteStepTicks(d.stepTicks);
    return ticks.length >= 2 ? ticks : null;
  }
  if (n.type === "trainer" || n.type === "crl_trainer") {
    const d = (n.data ?? {}) as Partial<TrainerNodeData>;
    const ticks = finiteStepTicks(d.stepTicks);
    return ticks.length >= 2 ? ticks : null;
  }
  return null;
}

function resolveUpstreamStepTicks(nodes: Node[], edges: Edge[], consumerId: string): number[] | null {
  const inc = edges.find((e) => e.target === consumerId && (e.targetHandle ?? "tensor") === "tensor");
  if (!inc?.source) return null;
  const src = nodes.find((n) => n.id === inc.source);
  if (!src) return null;

  if (
    (src.type === "training_visualization" && (inc.sourceHandle ?? "") === "out_tensor_list") ||
    ((src.type === "trainer" || src.type === "crl_trainer") && (inc.sourceHandle ?? "") === "loss_results")
  ) {
    return stepTicksFromTrainingLikeNode(src);
  }

  if (src.type !== "tensor_selector") return null;
  const listEdge = edges.find((e) => e.target === src.id && (e.targetHandle ?? "") === "tensor_list");
  if (!listEdge?.source) return null;
  const listSource = nodes.find((n) => n.id === listEdge.source);
  if (!listSource) return null;
  const tsData = (src.data ?? {}) as Partial<TensorSelectorNodeData>;
  const selected = Array.isArray(tsData.selectedTensorKeys)
    ? tsData.selectedTensorKeys.map((k) => String(k).trim()).filter(Boolean)
    : [String(tsData.selectedTensorKey ?? "").trim()].filter(Boolean);
  const selIdx = tensorSelectorOutputIndexFromSourceHandle(inc.sourceHandle);
  const selectedKey = selected[selIdx] ?? selected[0] ?? "";
  if (!selectedKey) return null;

  const listSh = listEdge.sourceHandle ?? "";
  if (
    listSource.type === "training_visualization" &&
    listSh === "out_tensor_list" &&
    (selectedKey === "train_loss" || selectedKey === "test_loss")
  ) {
    return stepTicksFromTrainingLikeNode(listSource);
  }
  if (
    (listSource.type === "trainer" || listSource.type === "crl_trainer") &&
    listSh === "loss_results" &&
    (selectedKey === "train_loss" || selectedKey === "test_loss")
  ) {
    return stepTicksFromTrainingLikeNode(listSource);
  }
  return null;
}

function centralDifference(values: number[], spacing: number): number[] {
  const dx = Math.max(1e-9, spacing);
  if (values.length <= 1) return new Array(values.length).fill(0);
  const out = new Array(values.length).fill(0);
  out[0] = (values[1]! - values[0]!) / dx;
  for (let i = 1; i < values.length - 1; i += 1) {
    out[i] = (values[i + 1]! - values[i - 1]!) / (2 * dx);
  }
  out[values.length - 1] = (values[values.length - 1]! - values[values.length - 2]!) / dx;
  return out;
}

function nthDerivative(values: number[], order: number, spacing: number): number[] {
  let cur = [...values];
  for (let i = 0; i < order; i += 1) {
    cur = centralDifference(cur, spacing);
  }
  return cur;
}

function ProcessedCurveChart({
  values,
  stepTicks,
  logScaleX,
  logScaleY,
}: {
  values: number[];
  stepTicks: number[] | null;
  logScaleX: boolean;
  logScaleY: boolean;
}) {
  if (values.length === 0) return <p className="cr-node__hint">No output values yet.</p>;
  if (values.length === 1) {
    return (
      <p className="cr-node__hint">
        Output has one point: <strong>{slFormatYTick(values[0]!, logScaleY)}</strong>
      </p>
    );
  }
  const steps = useMemo(() => {
    if (stepTicks && stepTicks.length === values.length) return stepTicks;
    return values.map((_, i) => i);
  }, [values, stepTicks]);
  const bounds = useMemo(
    () => slComputeBoundsScalar(steps, values, true, logScaleX, logScaleY, false),
    [steps, values, logScaleX, logScaleY],
  );
  const path = useMemo(
    () => slBuildSeriesPathScalar(steps, values, logScaleX, logScaleY, bounds, false),
    [steps, values, logScaleX, logScaleY, bounds],
  );
  const xTicks = useMemo(
    () => slGenerateXTicksScalar(bounds.minX, bounds.maxX, logScaleX, false),
    [bounds.minX, bounds.maxX, logScaleX],
  );
  const yTicks = useMemo(
    () => slGenerateYTicks(bounds.minY, bounds.maxY, logScaleY),
    [bounds.minY, bounds.maxY, logScaleY],
  );

  const innerBottom = slInnerBottom();
  const innerRight = slInnerRight();
  const plotMidX = SL_PAD_L + (SL_CHART_W - SL_PAD_L) / 2;
  const plotMidY = SL_PAD_T + (SL_CHART_H - SL_PAD_T - SL_PAD_B) / 2;
  const nonPositiveCount = logScaleY ? values.filter((v) => v <= 0).length : 0;
  const clipId = `derivative-curve-clip-${Math.random().toString(36).slice(2, 9)}`;

  return (
    <div className="cr-tviz-chart-wrap">
      <svg
        className="cr-tviz-chart nodrag nopan"
        viewBox={`0 0 ${SL_CHART_W} ${SL_CHART_H}`}
        width={SL_CHART_W}
        height={SL_CHART_H}
        aria-label="Derivative 1D curve"
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={SL_PAD_L}
              y={SL_PAD_T}
              width={SL_CHART_W - SL_PAD_L - (SL_CHART_W - innerRight)}
              height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
            />
          </clipPath>
        </defs>
        <rect
          x={SL_PAD_L}
          y={SL_PAD_T}
          width={SL_CHART_W - SL_PAD_L - (SL_CHART_W - innerRight)}
          height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
          rx={4}
          className="cr-tviz-chart__plot-bg"
        />
        {yTicks.map((yt) => {
          const py = slYToPx(yt, bounds);
          return (
            <line
              key={`gy-${yt}`}
              x1={SL_PAD_L}
              y1={py}
              x2={innerRight}
              y2={py}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        {xTicks.map((xt) => {
          const px = slXToPx(xt, bounds);
          return (
            <line
              key={`gx-${xt}`}
              x1={px}
              y1={SL_PAD_T}
              x2={px}
              y2={innerBottom}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        <line
          x1={SL_PAD_L}
          y1={innerBottom}
          x2={innerRight}
          y2={innerBottom}
          className="cr-tviz-chart__axis-line"
        />
        <line x1={SL_PAD_L} y1={SL_PAD_T} x2={SL_PAD_L} y2={innerBottom} className="cr-tviz-chart__axis-line" />
        {xTicks.map((xt) => {
          const px = slXToPx(xt, bounds);
          return (
            <g key={`xt-${xt}`}>
              <line
                x1={px}
                y1={innerBottom}
                x2={px}
                y2={innerBottom + SL_TICK_LEN}
                className="cr-tviz-chart__tick"
              />
              <text x={px} y={innerBottom + 12} textAnchor="middle" className="cr-tviz-chart__tick-label">
                {slFormatXTickScalar(xt, logScaleX, false)}
              </text>
            </g>
          );
        })}
        {yTicks.map((yt) => {
          const py = slYToPx(yt, bounds);
          return (
            <g key={`yt-${yt}`}>
              <line
                x1={SL_PAD_L - SL_TICK_LEN}
                y1={py}
                x2={SL_PAD_L}
                y2={py}
                className="cr-tviz-chart__tick"
              />
              <text
                x={SL_PAD_L - 6}
                y={py}
                dominantBaseline="middle"
                textAnchor="end"
                className="cr-tviz-chart__tick-label"
              >
                {slFormatYTick(yt, logScaleY)}
              </text>
            </g>
          );
        })}
        {path ? (
          <path
            d={path}
            fill="none"
            className="cr-tviz-chart__line"
            strokeWidth={1.6}
            clipPath={`url(#${clipId})`}
          />
        ) : null}
        <text x={plotMidX} y={SL_CHART_H - 2} textAnchor="middle" className="cr-tviz-chart__axis-title">
          step
        </text>
        <text
          x={SL_Y_LABEL_X}
          y={plotMidY}
          textAnchor="middle"
          dominantBaseline="middle"
          transform={`rotate(-90, ${SL_Y_LABEL_X}, ${plotMidY})`}
          className="cr-tviz-chart__axis-title"
        >
          value
        </text>
      </svg>
      {nonPositiveCount > 0 ? (
        <p className="cr-tviz-hint">
          log y clamps {nonPositiveCount} non-positive point{nonPositiveCount === 1 ? "" : "s"} to a small floor.
        </p>
      ) : null}
    </div>
  );
}

export function DerivativeCurveNode({ id, data, selected }: NodeProps) {
  const d = defaultDerivativeCurveData(data as Partial<DerivativeCurveNodeData>);
  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
  );

  const update = useCallback(
    (patch: Partial<DerivativeCurveNodeData>) => patchDerivativeCurveData(id, patch, setNodes),
    [id, setNodes],
  );
  const stepTicks = useStore(
    useCallback(
      (state) => resolveUpstreamStepTicks(state.nodes as Node[], state.edges as Edge[], id),
      [id],
    ),
  );
  const stepSpacing = useMemo(() => estimateStepSpacing(stepTicks), [stepTicks]);

  const compute = useCallback(async () => {
    const hydrated = await hydrateResolved(resolved);
    if (hydrated.kind !== "ok") {
      update({ outputTensor: null, lastError: hydrated.detail });
      return;
    }
    if (hydrated.shape.length !== 1) {
      update({
        outputTensor: null,
        lastError: "Input must be a rank-1 (1D) tensor. Reduce or slice first.",
      });
      return;
    }
    if (hydrated.values.length === 0) {
      update({ outputTensor: null, lastError: "Input series is empty." });
      return;
    }
    const order = Number.parseInt(d.order, 10);
    const outValues = nthDerivative(hydrated.values, Number.isFinite(order) ? order : 1, stepSpacing);
    const bad = outValues.some((v) => !Number.isFinite(v));
    update({
      outputTensor: bad ? null : { shape: [...hydrated.shape], values: outValues },
      lastError: bad ? "Derivative output contains non-finite values." : null,
    });
  }, [d.order, resolved, stepSpacing, update]);

  useEffect(() => {
    if (resolved.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    update({ outputTensor: null, lastError: resolved.detail });
  }, [resolved, d.outputTensor, update]);

  const outputValues = useMemo(() => d.outputTensor?.values ?? [], [d.outputTensor]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Derivative curve")}</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
          <div className="cr-tviz-socket-row__left">
            <Handle
              type="target"
              position={Position.Left}
              id="tensor"
              className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
            />
            <span className="cr-tviz-socket-label">tensor</span>
          </div>
          <div className="cr-tviz-socket-row__right cr-tviz-socket-row--dual">
            <div className="cr-tviz-socket-pair">
              <span className="cr-tviz-socket-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
              />
            </div>
          </div>
        </div>
        <DiscreteMultiSelect<DerivativeOrder>
          label="order"
          options={DERIVATIVE_ORDER_OPTIONS}
          value={d.order}
          singleSelect
          onCommit={(next) => update({ order: Array.isArray(next) ? (next[0] ?? "1") : next })}
          ariaLabel="Derivative order"
        />
        <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked nodrag nopan">
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--scales">
            <label className="cr-tviz-check cr-tviz-check--log-x">
              <input
                type="checkbox"
                checked={!!d.logScaleX}
                onChange={(e) => update({ logScaleX: e.target.checked })}
              />
              log x
            </label>
            <label className="cr-tviz-check cr-tviz-check--log-y">
              <input
                type="checkbox"
                checked={!!d.logScaleY}
                onChange={(e) => update({ logScaleY: e.target.checked })}
              />
              log y
            </label>
          </div>
        </div>
        <p className="cr-node__hint">Computes finite-difference derivative for a 1D series (order 1 to 5).</p>
        <ProcessedCurveChart
          values={outputValues}
          stepTicks={stepTicks}
          logScaleX={!!d.logScaleX}
          logScaleY={!!d.logScaleY}
        />
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
