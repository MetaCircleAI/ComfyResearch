import { Handle, Position, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, type ReactNode } from "react";
import { resolveUpstreamTensor, resolvedTensorEqual, type FlowEdge } from "../../graph/resolveUpstreamTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { slFormatYTick } from "./scalarLineChartShared";

type Effective1d = {
  values: number[];
  sourceSummary: string;
  shapeText: string;
  dim: number;
};

const CHART_W = 228;
const CHART_H = 132;
const AXIS_STROKE = "#5c5c6a";
const TICK_LEN = 4;

function resolveTensor(nodes: Node[], edges: FlowEdge[], vizId: string, targetHandle: string) {
  return resolveUpstreamTensor(nodes, edges, vizId, targetHandle);
}

function niceStep(span: number): number {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const exp = Math.floor(Math.log10(span));
  const f = span / 10 ** exp;
  let nf = 1;
  if (f < 1.5) nf = 1;
  else if (f < 3) nf = 2;
  else if (f < 7) nf = 5;
  else nf = 10;
  return nf * 10 ** exp;
}

function generateTicks(min: number, max: number, maxTicks = 4): number[] {
  const span = max - min;
  if (span <= 0 || !Number.isFinite(span)) return [min];
  const step = niceStep(span / Math.max(1, maxTicks - 1));
  const start = Math.ceil(min / step - 1e-9) * step;
  const out: number[] = [];
  for (let t = start; t <= max + step * 1e-9; t += step) {
    if (t >= min - 1e-9 && t <= max + 1e-9) out.push(t);
    if (out.length > 12) break;
  }
  return out.length > 0 ? out : [min, max];
}

function formatNumberTick(t: number): string {
  return slFormatYTick(t, false);
}

function normalizeEffective1d(resolved: Resolved): { ok?: Effective1d; err?: string } {
  if (resolved.kind === "none") return { err: resolved.detail };
  if (resolved.kind !== "ok") return { err: "Loading tensor from server…" };
  const { rank, shape, values, sourceSummary } = resolved;
  if (rank === 0) return { err: "Connected tensor is scalar. Scatter expects a 1-D tensor." };
  if (shape.some((d) => !Number.isFinite(d) || d <= 0)) {
    return { err: `Invalid tensor shape [${shape.join(", ")}].` };
  }
  const expected = shape.reduce((acc, d) => acc * d, 1);
  if (expected !== values.length) {
    return { err: `Tensor value count mismatch: expected ${expected}, got ${values.length}.` };
  }
  const nonSingleton = shape.filter((d) => d > 1);
  if (rank === 1) {
    return { ok: { values, sourceSummary, shapeText: `[${shape.join(" × ")}]`, dim: shape[0] ?? values.length } };
  }
  if (nonSingleton.length !== 1) {
    return {
      err: `Expected effectively 1-D tensor (only one dimension > 1). Got shape [${shape.join(", ")}].`,
    };
  }
  return {
    ok: {
      values,
      sourceSummary,
      shapeText: `[${shape.join(" × ")}] → [${nonSingleton[0]}]`,
      dim: nonSingleton[0]!,
    },
  };
}

function mapXLinear(left: number, w: number, v: number, vmin: number, vmax: number): number {
  const s = vmax - vmin || 1;
  return left + ((v - vmin) / s) * w;
}

function mapYLinear(top: number, h: number, v: number, vmin: number, vmax: number): number {
  const s = vmax - vmin || 1;
  return top + h - ((v - vmin) / s) * h;
}

function ScatterPlot({ xValues, yValues }: { xValues: number[]; yValues: number[] }) {
  const xmin = Math.min(...xValues);
  const xmax = Math.max(...xValues);
  const ymin = Math.min(...yValues);
  const ymax = Math.max(...yValues);
  const left = 36;
  const top = 8;
  const width = CHART_W - 36 - 10;
  const height = CHART_H - 8 - 26;
  const xTicks = generateTicks(xmin, xmax, 4);
  const yTicks = generateTicks(ymin, ymax, 4);
  const n = xValues.length;
  const r = n > 400 ? 1.6 : n > 150 ? 2.2 : 2.8;

  return (
    <svg
      className="cr-tensor-viz__svg nodrag nopan"
      width={CHART_W}
      height={CHART_H}
      viewBox={`0 0 ${CHART_W} ${CHART_H}`}
      aria-label="Scatter plot"
    >
      {xValues.map((xv, i) => (
        <circle
          key={i}
          cx={mapXLinear(left, width, xv, xmin, xmax)}
          cy={mapYLinear(top, height, yValues[i]!, ymin, ymax)}
          r={r}
          fill="color-mix(in srgb, var(--cr-accent-tensor) 55%, #1a1a22)"
          stroke="color-mix(in srgb, var(--cr-accent-tensor) 88%, #fff)"
          strokeWidth={0.45}
        />
      ))}
      <g className="cr-tensor-viz__axes" aria-hidden>
        <line x1={left} y1={top + height} x2={left + width} y2={top + height} stroke={AXIS_STROKE} strokeWidth={1} />
        <line x1={left} y1={top + height} x2={left} y2={top} stroke={AXIS_STROKE} strokeWidth={1} />
        {yTicks.map((ty, yi) => {
          const py = mapYLinear(top, height, ty, ymin, ymax);
          return (
            <g key={`y-${yi}-${ty}`}>
              <line x1={left - TICK_LEN} y1={py} x2={left} y2={py} stroke={AXIS_STROKE} strokeWidth={1} />
              <text
                className="cr-tensor-viz__axis-tick"
                x={left - TICK_LEN - 2}
                y={py}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {formatNumberTick(ty)}
              </text>
            </g>
          );
        })}
        {xTicks.map((tx, xi) => {
          const px = mapXLinear(left, width, tx, xmin, xmax);
          return (
            <g key={`x-${xi}-${tx}`}>
              <line x1={px} y1={top + height} x2={px} y2={top + height + TICK_LEN} stroke={AXIS_STROKE} strokeWidth={1} />
              <text
                className="cr-tensor-viz__axis-tick"
                x={px}
                y={top + height + TICK_LEN + 9}
                textAnchor="middle"
                dominantBaseline="hanging"
              >
                {formatNumberTick(tx)}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

export function TensorScatterVizNode({ id, selected }: NodeProps) {
  const resolved1 = useStore(
    useCallback((state) => resolveTensor(state.nodes as Node[], state.edges as FlowEdge[], id, "tensor_1"), [id]),
    resolvedTensorEqual,
  );
  const resolved2 = useStore(
    useCallback((state) => resolveTensor(state.nodes as Node[], state.edges as FlowEdge[], id, "tensor_2"), [id]),
    resolvedTensorEqual,
  );
  const { display: display1 } = useHydratedResolved(resolved1);
  const { display: display2 } = useHydratedResolved(resolved2);

  const body = useMemo((): ReactNode => {
    const n1 = normalizeEffective1d(display1);
    if (!n1.ok) return <p className="cr-tensor-viz__hint">tensor 1: {n1.err}</p>;
    const n2 = normalizeEffective1d(display2);
    if (!n2.ok) return <p className="cr-tensor-viz__hint">tensor 2: {n2.err}</p>;
    if (n1.ok.values.length !== n2.ok.values.length) {
      return (
        <p className="cr-tensor-viz__hint">
          Input lengths must match for scatter: tensor 1 has {n1.ok.values.length}, tensor 2 has {n2.ok.values.length}.
        </p>
      );
    }
    if (n1.ok.values.length === 0) {
      return <p className="cr-tensor-viz__hint">No points to plot.</p>;
    }
    return (
      <div className="cr-tensor-viz__plot">
        <ScatterPlot xValues={n1.ok.values} yValues={n2.ok.values} />
        <p className="cr-tensor-viz__meta">
          plt.scatter(tensor 1, tensor 2) · n={n1.ok.values.length}
        </p>
        <p className="cr-tensor-viz__meta">tensor 1: {n1.ok.sourceSummary} · {n1.ok.shapeText}</p>
        <p className="cr-tensor-viz__meta">tensor 2: {n2.ok.sourceSummary} · {n2.ok.shapeText}</p>
      </div>
    );
  }, [display1, display2]);

  return (
    <div
      className={`cr-node cr-node--tensor-viz${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">Scatter plot viz</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Scatter plot tensor inputs">
          <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
            <div className="cr-tviz-socket-row__left">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_1"
                className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
              />
              <span className="cr-tviz-socket-label">tensor 1</span>
            </div>
          </div>
          <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
            <div className="cr-tviz-socket-row__left">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_2"
                className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
              />
              <span className="cr-tviz-socket-label">tensor 2</span>
            </div>
          </div>
        </div>
        <div className="cr-tensor-viz__body nodrag nopan">{body}</div>
      </div>
    </div>
  );
}
