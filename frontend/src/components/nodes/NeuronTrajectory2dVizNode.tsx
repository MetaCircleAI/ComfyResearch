import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useMemo } from "react";
import { VizSocketsBar } from "./VizSocketsBar";
import {
  defaultObservableVizNeuronTrajectory2dData,
  type ObservableVizNeuronTrajectory2dNodeData,
} from "./observableVizNeuronTrajectory2dDefaults";
import { useObservableVizHeaderTitle } from "./observableVizTitle";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { slPadBounds } from "./scalarLineChartShared";

const CHART_W = 260;
const CHART_H = 220;
const PAD_L = 28;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 24;

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function patchData(
  id: string,
  patch: Partial<ObservableVizNeuronTrajectory2dNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = (n.data ?? {}) as Partial<ObservableVizNeuronTrajectory2dNodeData>;
      const prev: ObservableVizNeuronTrajectory2dNodeData = {
        ...defaultObservableVizNeuronTrajectory2dData(),
        ...cur,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function colorForNeuron(i: number, n: number): string {
  const h = Math.round((360 * (i % Math.max(1, n))) / Math.max(1, n));
  return `hsl(${h} 65% 55%)`;
}

function boundsFor(history: number[][][], dimX: number, dimY: number): Bounds {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const step of history) {
    for (const neuron of step) {
      const x = Number(neuron?.[dimX] ?? 0);
      const y = Number(neuron?.[dimY] ?? 0);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }
  return slPadBounds({ minX, maxX, minY, maxY }, 0.08, 0.08);
}

function sx(x: number, b: Bounds): number {
  return PAD_L + ((x - b.minX) / (b.maxX - b.minX || 1)) * (CHART_W - PAD_L - PAD_R);
}
function sy(y: number, b: Bounds): number {
  return PAD_T + (1 - (y - b.minY) / (b.maxY - b.minY || 1)) * (CHART_H - PAD_T - PAD_B);
}

export function NeuronTrajectory2dVizNode({ id, data, selected }: NodeProps) {
  const d = {
    ...defaultObservableVizNeuronTrajectory2dData(),
    ...(data as Partial<ObservableVizNeuronTrajectory2dNodeData>),
  } as ObservableVizNeuronTrajectory2dNodeData;
  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId);
  const { setNodes } = useReactFlow();

  const history = d.embeddingHistory ?? [];
  const neuronCount = history[0]?.length ?? 0;
  const inputDim = history[0]?.[0]?.length ?? 0;

  const dimX = d.dimX ?? 0;
  const dimY = d.dimY ?? (inputDim > 1 ? 1 : 0);
  const b = useMemo(() => boundsFor(history, dimX, dimY), [history, dimX, dimY]);

  // Trail paths: one polyline per neuron across all timesteps
  const trailPaths = useMemo(() => {
    if (!history.length || neuronCount <= 0) return [] as string[];
    return Array.from({ length: neuronCount }, (_, ni) => {
      const parts: string[] = [];
      for (let s = 0; s < history.length; s++) {
        const pt = history[s]?.[ni];
        if (!pt) continue;
        const x = sx(Number(pt[dimX] ?? 0), b);
        const y = sy(Number(pt[dimY] ?? 0), b);
        parts.push(`${parts.length ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`);
      }
      return parts.join(" ");
    });
  }, [history, neuronCount, dimX, dimY, b]);

  const initPts = history[0] ?? [];
  const finalPts = history[history.length - 1] ?? [];

  // Axis tick values
  const xTicks = useMemo(() => {
    const range = b.maxX - b.minX;
    const mid = (b.minX + b.maxX) / 2;
    return [b.minX, mid, b.maxX].map((v) => ({ v, label: range < 0.1 ? v.toExponential(1) : v.toFixed(2) }));
  }, [b]);
  const yTicks = useMemo(() => {
    const range = b.maxY - b.minY;
    const mid = (b.minY + b.maxY) / 2;
    return [b.minY, mid, b.maxY].map((v) => ({ v, label: range < 0.1 ? v.toExponential(1) : v.toFixed(2) }));
  }, [b]);

  const originInBounds =
    b.minX <= 0 && 0 <= b.maxX && b.minY <= 0 && 0 <= b.maxY;

  return (
    <div
      className={`cr-node cr-node--observable-viz-user${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle || "Neuron trajectory 2D"} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar />
        <div className="cr-tviz-chart-divider" aria-hidden />

        {inputDim > 2 ? (
          <div className="cr-tviz-chart-controls nodrag nopan" style={{ gap: 8 }}>
            <label className="cr-tviz-metric-select">
              <span className="cr-tviz-metric-select__lbl">x dim</span>
              <select value={String(dimX)} onChange={(e) => patchData(id, { dimX: Number(e.target.value) }, setNodes)}>
                {Array.from({ length: inputDim }).map((_, i) => <option key={i} value={i}>{i}</option>)}
              </select>
            </label>
            <label className="cr-tviz-metric-select">
              <span className="cr-tviz-metric-select__lbl">y dim</span>
              <select value={String(dimY)} onChange={(e) => patchData(id, { dimY: Number(e.target.value) }, setNodes)}>
                {Array.from({ length: inputDim }).map((_, i) => <option key={i} value={i}>{i}</option>)}
              </select>
            </label>
          </div>
        ) : null}

        <div className="cr-tviz-chart-wrap">
          <svg
            className="cr-tviz-chart nodrag nopan"
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            width={CHART_W}
            height={CHART_H}
          >
            {/* Plot background */}
            <rect
              x={PAD_L} y={PAD_T}
              width={CHART_W - PAD_L - PAD_R}
              height={CHART_H - PAD_T - PAD_B}
              rx={3}
              className="cr-tviz-chart__plot-bg"
            />
            {/* Axes */}
            <line x1={PAD_L} y1={CHART_H - PAD_B} x2={CHART_W - PAD_R} y2={CHART_H - PAD_B} className="cr-tviz-chart__axis-line" />
            <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={CHART_H - PAD_B} className="cr-tviz-chart__axis-line" />

            {/* Origin crosshair (if visible) */}
            {originInBounds ? (
              <>
                <line
                  x1={sx(0, b)} y1={PAD_T}
                  x2={sx(0, b)} y2={CHART_H - PAD_B}
                  stroke="var(--cr-text-3, #888)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}
                />
                <line
                  x1={PAD_L} y1={sy(0, b)}
                  x2={CHART_W - PAD_R} y2={sy(0, b)}
                  stroke="var(--cr-text-3, #888)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}
                />
              </>
            ) : null}

            {/* Axis tick labels */}
            {xTicks.map(({ v, label }, i) => (
              <text
                key={`xt-${i}`}
                x={sx(v, b)} y={CHART_H - PAD_B + 10}
                textAnchor="middle" fontSize={8}
                className="cr-tviz-chart__tick-label"
              >{label}</text>
            ))}
            {yTicks.map(({ v, label }, i) => (
              <text
                key={`yt-${i}`}
                x={PAD_L - 3} y={sy(v, b) + 3}
                textAnchor="end" fontSize={8}
                className="cr-tviz-chart__tick-label"
              >{label}</text>
            ))}

            {/* Initial positions (hollow circles) */}
            {d.showTrails !== false && d.showPoints !== false
              ? initPts.map((pt, i) => {
                  if (!pt) return null;
                  return (
                    <circle
                      key={`i-${i}`}
                      cx={sx(Number(pt[dimX] ?? 0), b)}
                      cy={sy(Number(pt[dimY] ?? 0), b)}
                      r={2.2}
                      fill="none"
                      stroke={colorForNeuron(i, neuronCount)}
                      strokeWidth={1}
                      opacity={0.55}
                    />
                  );
                })
              : null}

            {/* Trajectory trails */}
            {d.showTrails !== false
              ? trailPaths.map((p, i) => (
                  <path
                    key={`t-${i}`}
                    d={p}
                    fill="none"
                    stroke={colorForNeuron(i, neuronCount)}
                    strokeWidth={1}
                    opacity={0.45}
                  />
                ))
              : null}

            {/* Final positions (filled dots) */}
            {d.showPoints !== false
              ? finalPts.map((pt, i) => {
                  if (!pt) return null;
                  return (
                    <circle
                      key={`f-${i}`}
                      cx={sx(Number(pt[dimX] ?? 0), b)}
                      cy={sy(Number(pt[dimY] ?? 0), b)}
                      r={3}
                      fill={colorForNeuron(i, neuronCount)}
                    />
                  );
                })
              : null}
          </svg>

          {history.length === 0 ? (
            <p className="cr-tviz-hint">No trajectory yet — connect to trainer observable_results and train.</p>
          ) : (
            <p className="cr-tviz-hint">
              {history.length} steps · {neuronCount} neurons · dim ({dimX},{dimY})
              {d.showTrails !== false ? " · hollow=init, filled=final" : ""}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
