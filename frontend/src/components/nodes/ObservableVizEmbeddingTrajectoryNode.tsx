import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useMemo } from "react";
import { VizSocketsBar } from "./VizSocketsBar";
import {
  defaultObservableVizEmbeddingTrajectoryData,
  type ObservableVizEmbeddingTrajectoryNodeData,
} from "./observableVizEmbeddingTrajectoryDefaults";
import { useObservableVizHeaderTitle } from "./observableVizTitle";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { slPadBounds } from "./scalarLineChartShared";

const CHART_W = 232;
const CHART_H = 152;
const PAD_L = 28;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 22;

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function patchData(
  id: string,
  patch: Partial<ObservableVizEmbeddingTrajectoryNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = (n.data ?? {}) as Partial<ObservableVizEmbeddingTrajectoryNodeData>;
      const prev: ObservableVizEmbeddingTrajectoryNodeData = {
        ...defaultObservableVizEmbeddingTrajectoryData(),
        ...cur,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function colorForToken(i: number, n: number): string {
  const h = Math.round((360 * (i % Math.max(1, n))) / Math.max(1, n));
  return `hsl(${h} 70% 60%)`;
}

function boundsFor(history: number[][][], dimX: number, dimY: number): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const step of history) {
    for (const tok of step) {
      const x = Number(tok?.[dimX] ?? 0);
      const y = Number(tok?.[dimY] ?? 0);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  return slPadBounds({ minX, maxX, minY, maxY }, 0.055, 0.055);
}

function scaleX(x: number, b: Bounds): number {
  return PAD_L + ((x - b.minX) / (b.maxX - b.minX || 1)) * (CHART_W - PAD_L - PAD_R);
}
function scaleY(y: number, b: Bounds): number {
  return PAD_T + (1 - (y - b.minY) / (b.maxY - b.minY || 1)) * (CHART_H - PAD_T - PAD_B);
}

export function ObservableVizEmbeddingTrajectoryNode({ id, data, selected }: NodeProps) {
  const d = {
    ...defaultObservableVizEmbeddingTrajectoryData(),
    ...(data as Partial<ObservableVizEmbeddingTrajectoryNodeData>),
  } as ObservableVizEmbeddingTrajectoryNodeData;
  const headerTitle = useObservableVizHeaderTitle(d.pairedObservableId);
  const { setNodes } = useReactFlow();
  const history = d.embeddingHistory ?? [];
  const tokenCount = history[0]?.length ?? 0;
  const embedDim = history[0]?.[0]?.length ?? 0;

  const needsDimPick = embedDim > 2 && (d.dimX == null || d.dimY == null);
  const dimX = d.dimX ?? 0;
  const dimY = d.dimY ?? (embedDim > 1 ? 1 : 0);
  const b = useMemo(() => boundsFor(history, dimX, dimY), [history, dimX, dimY]);

  const tokenPaths = useMemo(() => {
    if (!history.length || tokenCount <= 0 || needsDimPick) return [] as string[];
    const out: string[] = [];
    for (let t = 0; t < tokenCount; t++) {
      const parts: string[] = [];
      for (let s = 0; s < history.length; s++) {
        const pt = history[s]?.[t];
        if (!pt) continue;
        const x = scaleX(Number(pt[dimX] ?? 0), b);
        const y = scaleY(Number(pt[dimY] ?? 0), b);
        parts.push(`${parts.length ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`);
      }
      out.push(parts.join(" "));
    }
    return out;
  }, [history, tokenCount, dimX, dimY, b, needsDimPick]);

  const latest = history[history.length - 1] ?? [];

  return (
    <div
      className={`cr-node cr-node--observable-viz-user${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={headerTitle} />
      <div className="cr-node__body cr-node__body--tviz">
        <VizSocketsBar />
        <div className="cr-tviz-chart-divider" aria-hidden />

        {embedDim > 0 ? (
          <div className="cr-tviz-chart-controls nodrag nopan" style={{ gap: 8 }}>
            <label className="cr-tviz-metric-select">
              <span className="cr-tviz-metric-select__lbl">x dim</span>
              <select
                value={String(dimX)}
                onChange={(e) => patchData(id, { dimX: Number(e.target.value) }, setNodes)}
              >
                {Array.from({ length: embedDim }).map((_, i) => (
                  <option key={`x-${i}`} value={i}>{i}</option>
                ))}
              </select>
            </label>
            <label className="cr-tviz-metric-select">
              <span className="cr-tviz-metric-select__lbl">y dim</span>
              <select
                value={String(dimY)}
                onChange={(e) => patchData(id, { dimY: Number(e.target.value) }, setNodes)}
              >
                {Array.from({ length: embedDim }).map((_, i) => (
                  <option key={`y-${i}`} value={i}>{i}</option>
                ))}
              </select>
            </label>
          </div>
        ) : null}

        <div className="cr-tviz-chart-wrap">
          <svg className="cr-tviz-chart nodrag nopan" viewBox={`0 0 ${CHART_W} ${CHART_H}`} width={CHART_W} height={CHART_H}>
            <rect
              x={PAD_L}
              y={PAD_T}
              width={CHART_W - PAD_L - PAD_R}
              height={CHART_H - PAD_T - PAD_B}
              rx={4}
              className="cr-tviz-chart__plot-bg"
            />
            <line x1={PAD_L} y1={CHART_H - PAD_B} x2={CHART_W - PAD_R} y2={CHART_H - PAD_B} className="cr-tviz-chart__axis-line" />
            <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={CHART_H - PAD_B} className="cr-tviz-chart__axis-line" />

            {!needsDimPick && d.showTrails !== false ? tokenPaths.map((p, i) => (
              <path key={`tp-${i}`} d={p} fill="none" stroke={colorForToken(i, tokenCount)} strokeWidth={1.2} opacity={0.8} />
            )) : null}

            {!needsDimPick && d.showPoints !== false ? latest.map((pt, i) => {
              const x = scaleX(Number(pt?.[dimX] ?? 0), b);
              const y = scaleY(Number(pt?.[dimY] ?? 0), b);
              return <circle key={`lp-${i}`} cx={x} cy={y} r={2.6} fill={colorForToken(i, tokenCount)} />;
            }) : null}
          </svg>
          {history.length === 0 ? (
            <p className="cr-tviz-hint">No embedding history yet — connect to Trainer observable output and train.</p>
          ) : needsDimPick ? (
            <p className="cr-tviz-hint">Embedding has {embedDim} dimensions. Please choose two dimensions to display.</p>
          ) : (
            <p className="cr-tviz-hint">{history.length} steps · {tokenCount} tokens · dims ({dimX}, {dimY})</p>
          )}
        </div>
      </div>
    </div>
  );
}
