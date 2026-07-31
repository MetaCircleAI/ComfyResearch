import type { NodeProps } from "@xyflow/react";
import { useMemo } from "react";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { VizSocketsBar } from "./VizSocketsBar";
import { useObservableVizHeaderTitle } from "./observableVizTitle";

type Data = { pairedObservableId?: string; embeddingHistory?: number[][][]; stepTicks?: number[] };
const W = 300, H = 230, L = 38, R = 12, T = 12, B = 34;
const color = (t: number) => {
  const bounded = Math.max(0, Math.min(1, t));
  return `hsl(${(278 + 132 * bounded).toFixed(0)} 90% ${(32 + 20 * bounded).toFixed(0)}%)`;
};

export function informationPlaneTicks(limit: number, intervals: number) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? limit : 1;
  return Array.from({ length: intervals + 1 }, (_, index) => safeLimit * index / intervals);
}

export function informationPlaneProgress(index: number, historyLength: number, stepTicks?: number[]) {
  const lastTick = stepTicks?.[historyLength - 1];
  const tick = stepTicks?.[index];
  if (Number.isFinite(tick) && Number.isFinite(lastTick) && Number(lastTick) > 0) {
    return Math.max(0, Math.min(1, Number(tick) / Number(lastTick)));
  }
  return Math.max(0, Math.min(1, index / Math.max(1, historyLength - 1)));
}

export function informationPlaneLimits(history: number[][][]) {
  let maxX = 1;
  let maxY = 1;
  for (const frame of history) {
    for (const point of frame) {
      if (Number.isFinite(point[0])) maxX = Math.max(maxX, point[0]);
      if (Number.isFinite(point[1])) maxY = Math.max(maxY, point[1]);
    }
  }
  return {
    x: Math.max(12, Math.ceil(maxX)),
    y: Math.max(1, Math.ceil(maxY * 10) / 10),
  };
}

export function InformationPlaneVizNode({ id, data, selected }: NodeProps) {
  const d = data as Data;
  const history = d.embeddingHistory ?? [];
  const title = useObservableVizHeaderTitle(d.pairedObservableId) || "Information plane";
  const limits = useMemo(() => informationPlaneLimits(history), [history]);
  const sx = (value: number) => L + Math.max(0, value) / limits.x * (W - L - R);
  const sy = (value: number) => T + (1 - Math.max(0, value) / limits.y) * (H - T - B);
  const xTicks = useMemo(() => informationPlaneTicks(limits.x, 6), [limits.x]);
  const yTicks = useMemo(() => informationPlaneTicks(limits.y, 5), [limits.y]);
  const connectorFrames = useMemo(() => {
    if (!history.length) return [];
    const stride = Math.max(1, Math.ceil(history.length / 96));
    const frames = history.filter((_frame, index) => index % stride === 0);
    if (frames.at(-1) !== history.at(-1)) frames.push(history.at(-1)!);
    return frames;
  }, [history]);
  const layerCount = Math.max(0, ...history.map((frame) => frame.length));
  const last = history.at(-1) ?? [];
  const gradientId = `information-plane-time-${id}`;
  const formatTick = (value: number) => Math.abs(value - Math.round(value)) < 1e-6
    ? String(Math.round(value))
    : value.toFixed(1);
  return <div className={`cr-node cr-node--observable-viz-user${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={title} />
    <div className="cr-node__body cr-node__body--tviz"><VizSocketsBar /><div className="cr-tviz-chart-divider" aria-hidden />
      <div className="cr-tviz-chart-wrap"><svg className="cr-tviz-chart nodrag nopan" viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-label="Information plane: I(X; T) versus I(T; Y), in bits">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color(0)} />
            <stop offset="50%" stopColor={color(0.5)} />
            <stop offset="100%" stopColor={color(1)} />
          </linearGradient>
        </defs>
        <rect x={L} y={T} width={W - L - R} height={H - T - B} rx={3} className="cr-tviz-chart__plot-bg" />
        {xTicks.map((tick) => <line key={`grid-x-${tick}`} x1={sx(tick)} y1={T} x2={sx(tick)} y2={H-B} className="cr-tviz-chart__grid" />)}
        {yTicks.map((tick) => <line key={`grid-y-${tick}`} x1={L} y1={sy(tick)} x2={W-R} y2={sy(tick)} className="cr-tviz-chart__grid" />)}
        {connectorFrames.map((frame, frameIndex) => frame.slice(1).map((point, layer) => {
          const before = frame[layer];
          if (!before || !point) return null;
          return <line key={`connector-${frameIndex}-${layer}`} x1={sx(before[0])} y1={sy(before[1])} x2={sx(point[0])} y2={sy(point[1])} stroke="#a5a1aa" strokeOpacity={0.2} strokeWidth={0.7} />;
        }))}
        {Array.from({ length: layerCount }, (_, layer) => history.slice(1).map((frame, index) => {
          const before = history[index]?.[layer], after = frame?.[layer];
          if (!before || !after) return null;
          return <line key={`${layer}-${index}`} x1={sx(before[0])} y1={sy(before[1])} x2={sx(after[0])} y2={sy(after[1])} stroke={color(informationPlaneProgress(index + 1, history.length, d.stepTicks))} strokeWidth={1.7} />;
        }))}
        <line x1={L} y1={H-B} x2={W-R} y2={H-B} className="cr-tviz-chart__axis-line" /><line x1={L} y1={T} x2={L} y2={H-B} className="cr-tviz-chart__axis-line" />
        {xTicks.map((tick) => <g key={`tick-x-${tick}`}><line x1={sx(tick)} y1={H-B} x2={sx(tick)} y2={H-B+3} className="cr-tviz-chart__tick" /><text x={sx(tick)} y={H-B+11} textAnchor="middle" className="cr-tviz-chart__tick-label">{formatTick(tick)}</text></g>)}
        {yTicks.map((tick) => <g key={`tick-y-${tick}`}><line x1={L-3} y1={sy(tick)} x2={L} y2={sy(tick)} className="cr-tviz-chart__tick" /><text x={L-5} y={sy(tick)} dominantBaseline="middle" textAnchor="end" className="cr-tviz-chart__tick-label">{formatTick(tick)}</text></g>)}
        <line x1={W-78} y1={T+7} x2={W-24} y2={T+7} stroke={`url(#${gradientId})`} strokeWidth={3} />
        <text x={W-82} y={T+9} textAnchor="end" className="cr-tviz-chart__legend-text">epoch</text>
        <text x={W-80} y={T+17} textAnchor="start" className="cr-tviz-chart__tick-label">early</text>
        <text x={W-22} y={T+17} textAnchor="end" className="cr-tviz-chart__tick-label">late</text>
        {last.map((point, layer) => <g key={layer}><circle cx={sx(point[0])} cy={sy(point[1])} r={3.5} fill={color(1)} stroke="#332a08" strokeWidth={0.7} /><text x={sx(point[0]) + 5} y={sy(point[1]) - 4} fontSize={7} className="cr-tviz-chart__tick-label">L{layer + 1}</text></g>)}
        <text x={(L + W - R) / 2} y={H - 3} textAnchor="middle" className="cr-tviz-chart__axis-title">I(X; T) bits</text><text x={8} y={(T + H - B) / 2} textAnchor="middle" className="cr-tviz-chart__axis-title" transform={`rotate(-90 8 ${(T + H - B) / 2})`}>I(T; Y) bits</text>
      </svg><p className="cr-tviz-hint">{history.length ? `${history.length} log points · ${layerCount} layers` : "No information-plane trajectory yet — connect to Trainer and train."}</p></div>
    </div>
  </div>;
}
