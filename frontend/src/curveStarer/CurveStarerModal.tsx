import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type PointerEvent as ReactPointerEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";
import type { Edge, Node } from "@xyflow/react";
import { DiscreteMultiSelect } from "../components/nodes/DiscreteMultiSelect";
import { slFormatYTick, slGenerateYTicks, slPadBounds } from "../components/nodes/scalarLineChartShared";
import {
  commonBoundaryGoldStyle,
  computeCommonBoundaryClusters,
  matchCommonBoundaryStrength,
  type CommonBoundaryCluster,
} from "./commonBoundaryClusters";
import { CorrelationFinderPanel } from "./CorrelationFinderPanel";
import { collectObservableTrainingCurves, LPD_MIN_CURVE_POINTS } from "./collectObservableCurves";
import {
  LPD_PREDICT_CONCURRENCY,
  type CurveStarerAnalyzedEntry,
  type CurveStarerRankBy,
  type LpdSegment,
} from "./lpdTypes";
import type { TargetObjective } from "./targetPhaseTransition";
import type { CurveStarerTargetConfig } from "./speedUpTrickTypes";
import {
  CURVE_STARER_RANK_BY_OPTIONS,
  interestingnessBreakdownForEntries,
  isCurveStarerRankBy,
  rankCurveStarerEntries,
  type CurveStarerInterestingnessBreakdown,
} from "./rankCurveStarerEntries";
import type { CurvePoint } from "./observableCurvePayload";

export type CurveStarerTab = "phase-segmentation" | "correlation-finder";

const PHASE_COLORS = [
  "#6c9eff",
  "#56d4a0",
  "#ffb347",
  "#ff7eb3",
  "#b388ff",
  "#4dd0e1",
  "#ffab91",
  "#80cbc4",
];

const DEFAULT_GRID_COLS = 3;
const MIN_GRID_COLS = 1;
const MAX_GRID_COLS = 12;

const CELL_W = 208;
const CELL_H = 118;
const PAD_L = 32;
const PAD_R = 6;
const PAD_T = 18;
const PAD_B = 16;
const TICK_LEN = 3;

const TARGET_OBJECTIVE_OPTIONS = [
  { id: "higher", label: "Higher is better" },
  { id: "lower", label: "Lower is better" },
] as const;

const TAB_OPTIONS: { id: CurveStarerTab; label: string }[] = [
  { id: "phase-segmentation", label: "Phase segmentation" },
  { id: "correlation-finder", label: "Correlation finder" },
];

function boundsForPoints(points: CurvePoint[]): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
} {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (!Number.isFinite(p.t) || !Number.isFinite(p.loss)) continue;
    minX = Math.min(minX, p.t);
    maxX = Math.max(maxX, p.t);
    minY = Math.min(minY, p.loss);
    maxY = Math.max(maxY, p.loss);
  }
  if (!Number.isFinite(minX)) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 0.5;
    maxY += 0.5;
  }
  return slPadBounds({ minX, maxX, minY, maxY }, 0.055, 0.055);
}

function yToPx(y: number, b: ReturnType<typeof boundsForPoints>): number {
  const innerH = CELL_H - PAD_T - PAD_B;
  return PAD_T + innerH * (1 - (y - b.minY) / (b.maxY - b.minY || 1));
}

function tFromChartClientX(
  clientX: number,
  svg: SVGSVGElement,
  b: ReturnType<typeof boundsForPoints>,
): number | null {
  const rect = svg.getBoundingClientRect();
  if (rect.width <= 0) return null;
  const px = ((clientX - rect.left) / rect.width) * CELL_W;
  if (px < PAD_L || px > CELL_W - PAD_R) return null;
  const innerW = CELL_W - PAD_L - PAD_R;
  const spanX = b.maxX - b.minX || 1;
  return b.minX + ((px - PAD_L) / innerW) * spanX;
}

function xPxForT(t: number, b: ReturnType<typeof boundsForPoints>): number | null {
  const innerW = CELL_W - PAD_L - PAD_R;
  const spanX = b.maxX - b.minX || 1;
  if (t < b.minX || t > b.maxX) return null;
  return PAD_L + (innerW * (t - b.minX)) / spanX;
}

function buildPath(points: CurvePoint[], b: ReturnType<typeof boundsForPoints>): string {
  const innerW = CELL_W - PAD_L - PAD_R;
  const innerH = CELL_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  let d = "";
  let started = false;
  for (const p of points) {
    if (!Number.isFinite(p.t) || !Number.isFinite(p.loss)) continue;
    const px = PAD_L + (innerW * (p.t - b.minX)) / spanX;
    const py = PAD_T + innerH * (1 - (p.loss - b.minY) / spanY);
    d += `${started ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
    started = true;
  }
  return d;
}

function phaseRects(
  segments: LpdSegment[],
  b: ReturnType<typeof boundsForPoints>,
): {
  x: number;
  width: number;
  color: string;
  label: string;
  mechanism: string;
  boundaryNorm: number | null;
}[] {
  const innerW = CELL_W - PAD_L - PAD_R;
  const spanX = b.maxX - b.minX || 1;
  return segments.map((seg, i) => {
    const x0 = PAD_L + (innerW * (seg.t_start - b.minX)) / spanX;
    const x1 = PAD_L + (innerW * (seg.t_end - b.minX)) / spanX;
    const isLast = i === segments.length - 1;
    const boundaryNorm = isLast ? null : (seg.t_end - b.minX) / spanX;
    return {
      x: x0,
      width: Math.max(2, x1 - x0),
      color: PHASE_COLORS[i % PHASE_COLORS.length]!,
      label: `L${i + 1}`,
      mechanism: seg.mechanism_label ?? seg.mechanism ?? "phase",
      boundaryNorm,
    };
  });
}

function CurveStarerHelpIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 18.5a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17Z"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <path
        d="M7.6 7.4c0-1.45 1.15-2.35 2.4-2.35 1.2 0 2.05.75 2.05 1.75 0 1-0.55 1.45-1.25 2.05-0.75 0.65-1.4 1.3-1.4 2.35"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="10" cy="14.15" r="0.85" fill="currentColor" />
    </svg>
  );
}

function CurveStarerInterestingnessPanel({
  label,
  breakdown,
}: {
  label: string;
  breakdown: CurveStarerInterestingnessBreakdown;
}) {
  return (
    <div className="cr-curve-starer-cell__interestingness nodrag nopan" role="tooltip">
      <p className="cr-curve-starer-cell__interestingness-title">Why interesting?</p>
      <p className="cr-curve-starer-cell__interestingness-summary" title={label}>
        {label}
      </p>
      {breakdown.topReason ? (
        <p className="cr-curve-starer-cell__interestingness-reason">
          Stands out on <strong>{breakdown.topReason}</strong> (overall {breakdown.overall.toFixed(2)})
        </p>
      ) : (
        <p className="cr-curve-starer-cell__interestingness-reason">
          Overall score {breakdown.overall.toFixed(2)}
        </p>
      )}
      <div className="cr-curve-starer-cell__interestingness-table-wrap">
        <table className="cr-curve-starer-cell__interestingness-table">
          <thead>
            <tr>
              <th>Attribute</th>
              <th>Value</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {[...breakdown.attributes]
              .sort((a, b) => b.score - a.score || a.label.localeCompare(b.label))
              .map((row) => {
                const isTop = row.score > 0 && row.score === breakdown.overall;
                return (
                  <tr
                    key={row.id}
                    className={isTop ? "cr-curve-starer-cell__interestingness-row--top" : undefined}
                  >
                    <td>{row.label}</td>
                    <td className="cr-curve-starer-cell__interestingness-value">{row.valueLabel}</td>
                    <td className="cr-curve-starer-cell__interestingness-score">{row.score.toFixed(2)}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
      <p className="cr-curve-starer-cell__interestingness-foot">
        Overall = max attribute score (1 = rank 1, 0 = rank last)
      </p>
    </div>
  );
}

function CurveStarerCell({
  entry,
  breakdown,
  showCommonBoundaries,
  commonBoundaryClusters,
  syncT,
  onSyncTChange,
}: {
  entry: CurveStarerAnalyzedEntry;
  breakdown: CurveStarerInterestingnessBreakdown | undefined;
  showCommonBoundaries: boolean;
  commonBoundaryClusters: CommonBoundaryCluster[];
  syncT: number | null;
  onSyncTChange: (t: number | null) => void;
}) {
  const [showHelp, setShowHelp] = useState(false);
  const points = entry.lpd?.data ?? entry.points;
  const fitted = entry.lpd?.fitted ?? [];
  const segments = entry.lpd?.segments ?? [];
  const b = useMemo(() => boundsForPoints(points), [points]);
  const yTicks = useMemo(() => slGenerateYTicks(b.minY, b.maxY, false, 4), [b.minY, b.maxY]);
  const dataPath = useMemo(() => buildPath(points, b), [points, b]);
  const fitPath = useMemo(() => (fitted.length >= 2 ? buildPath(fitted, b) : ""), [fitted, b]);
  const phases = useMemo(() => phaseRects(segments, b), [segments, b]);
  const innerRight = CELL_W - PAD_R;
  const innerBottom = CELL_H - PAD_B;
  const syncLineX = syncT != null ? xPxForT(syncT, b) : null;

  const handleChartPointerMove = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const t = tFromChartClientX(e.clientX, e.currentTarget, b);
      if (t != null) onSyncTChange(t);
    },
    [b, onSyncTChange],
  );

  const handleChartPointerLeave = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const related = e.relatedTarget;
      if (related instanceof Element && related.closest(".cr-curve-starer-cell__chart")) return;
      onSyncTChange(null);
    },
    [onSyncTChange],
  );

  return (
    <article className="cr-curve-starer-cell">
      <header className="cr-curve-starer-cell__header">
        <h3 className="cr-curve-starer-cell__title" title={entry.label}>
          {entry.label}
        </h3>
        <button
          type="button"
          className="cr-curve-starer-cell__help nodrag nopan"
          aria-label="Why is this curve interesting?"
          title="Why interesting?"
          onMouseEnter={() => setShowHelp(true)}
          onMouseLeave={() => setShowHelp(false)}
        >
          <CurveStarerHelpIcon />
        </button>
      </header>
      <div className="cr-curve-starer-cell__plot">
        <svg
          className="cr-curve-starer-cell__chart"
          viewBox={`0 0 ${CELL_W} ${CELL_H}`}
          preserveAspectRatio="xMidYMid meet"
          width="100%"
          role="img"
          aria-label={`Training dynamics for ${entry.label}`}
          onPointerMove={handleChartPointerMove}
          onPointerLeave={handleChartPointerLeave}
        >
          <rect x={PAD_L} y={PAD_T} width={CELL_W - PAD_L - PAD_R} height={CELL_H - PAD_T - PAD_B} fill="#0c0e14" rx={4} />
          {yTicks.map((yt) => (
            <line
              key={`gy-${yt}`}
              x1={PAD_L}
              y1={yToPx(yt, b)}
              x2={innerRight}
              y2={yToPx(yt, b)}
              className="cr-tviz-chart__grid"
            />
          ))}
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={innerBottom} className="cr-tviz-chart__axis-line" />
          <line x1={PAD_L} y1={innerBottom} x2={innerRight} y2={innerBottom} className="cr-tviz-chart__axis-line" />
          {yTicks.map((yt) => {
            const py = yToPx(yt, b);
            return (
              <g key={`yt-${yt}`}>
                <line x1={PAD_L - TICK_LEN} y1={py} x2={PAD_L} y2={py} className="cr-tviz-chart__tick" />
                <text
                  x={PAD_L - 5}
                  y={py}
                  dominantBaseline="middle"
                  textAnchor="end"
                  className="cr-tviz-chart__tick-label"
                >
                  {slFormatYTick(yt, false)}
                </text>
              </g>
            );
          })}
          {phases.map((ph) => {
            const commonStrength =
              showCommonBoundaries && ph.boundaryNorm != null
                ? matchCommonBoundaryStrength(ph.boundaryNorm, commonBoundaryClusters)
                : 0;
            const gold = commonStrength > 0 ? commonBoundaryGoldStyle(commonStrength) : null;
            const boundaryStroke = gold?.stroke ?? ph.color;
            const boundaryOpacity = gold?.strokeOpacity ?? 0.55;
            const boundaryWidth = gold?.strokeWidth ?? 1;
            const boundaryDash = gold ? undefined : "3 3";
            return (
              <g key={`${ph.label}-${ph.x}`}>
                <rect
                  x={ph.x}
                  y={PAD_T}
                  width={ph.width}
                  height={CELL_H - PAD_T - PAD_B}
                  fill={ph.color}
                  fillOpacity={0.14}
                />
                {ph.boundaryNorm != null ? (
                  <line
                    x1={ph.x + ph.width}
                    x2={ph.x + ph.width}
                    y1={PAD_T}
                    y2={CELL_H - PAD_B}
                    stroke={boundaryStroke}
                    strokeOpacity={boundaryOpacity}
                    strokeWidth={boundaryWidth}
                    strokeDasharray={boundaryDash}
                    style={gold?.filter ? { filter: gold.filter } : undefined}
                  />
                ) : null}
                <text
                  x={ph.x + ph.width / 2}
                  y={PAD_T + 9}
                  textAnchor="middle"
                  className="cr-curve-starer-cell__phase-label"
                  fill={ph.color}
                >
                  {ph.label}
                </text>
              </g>
            );
          })}
          {dataPath ? (
            <path d={dataPath} fill="none" stroke="#e8ecf5" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
          ) : null}
          {fitPath ? (
            <path d={fitPath} fill="none" stroke="#ef4444" strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
          ) : null}
          {syncLineX != null ? (
            <line
              x1={syncLineX}
              x2={syncLineX}
              y1={PAD_T}
              y2={innerBottom}
              className="cr-curve-starer-cell__sync-line"
              pointerEvents="none"
            />
          ) : null}
        </svg>
        {showHelp && breakdown ? (
          <CurveStarerInterestingnessPanel label={entry.label} breakdown={breakdown} />
        ) : null}
      </div>
      {entry.lpdError ? <p className="cr-curve-starer-cell__error">{entry.lpdError}</p> : null}
      {segments.length > 0 ? (
        <ul className="cr-curve-starer-cell__legend">
          {segments.map((seg, i) => (
            <li key={`${seg.t_start}-${seg.t_end}`}>
              <span className="cr-curve-starer-cell__swatch" style={{ background: PHASE_COLORS[i % PHASE_COLORS.length] }} />
              <span>
                L{i + 1}: {seg.mechanism_label ?? seg.mechanism ?? "phase"}
                {typeof seg.segment_r2 === "number" ? ` · R² ${seg.segment_r2.toFixed(3)}` : ""}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

export function CurveStarerModal({
  open,
  busy,
  progress,
  statusMessage,
  entries,
  rankBy,
  onRankByChange,
  targetConfig,
  targetCurveOptions,
  onTargetConfigChange,
  nodes,
  edges,
  onStartAnalyze,
  onClose,
}: {
  open: boolean;
  busy: boolean;
  progress: { total: number; completed: number } | null;
  statusMessage: string | null;
  entries: CurveStarerAnalyzedEntry[];
  rankBy: CurveStarerRankBy;
  onRankByChange: (next: CurveStarerRankBy) => void;
  targetConfig: CurveStarerTargetConfig | null;
  targetCurveOptions: { id: string; label: string }[];
  onTargetConfigChange: (next: CurveStarerTargetConfig) => void;
  nodes: Node[];
  edges: Edge[];
  onStartAnalyze: () => void;
  onClose: () => void;
}) {
  const [activeTab, setActiveTab] = useState<CurveStarerTab>("phase-segmentation");
  const [gridColsDraft, setGridColsDraft] = useState(String(DEFAULT_GRID_COLS));
  const [showCommonBoundaries, setShowCommonBoundaries] = useState(false);
  const [syncT, setSyncT] = useState<number | null>(null);
  const [correlationBusy, setCorrelationBusy] = useState(false);
  const [stareRequestId, setStareRequestId] = useState(0);

  const targetOptions = useMemo(() => {
    const live = collectObservableTrainingCurves(nodes, edges).map((c) => ({
      id: c.entryId,
      label: c.label,
    }));
    return live.length > 0 ? live : targetCurveOptions;
  }, [edges, nodes, targetCurveOptions]);

  const effectiveTargetConfig = useMemo((): CurveStarerTargetConfig | null => {
    if (targetConfig) return targetConfig;
    if (targetOptions.length === 0) return null;
    return { entryId: targetOptions[0]!.id, objective: "higher", threshold: 0.95 };
  }, [targetConfig, targetOptions]);


  const handleTargetChange = useCallback(
    (entryId: string) => {
      const base =
        targetConfig ??
        effectiveTargetConfig ?? { entryId: "", objective: "higher" as TargetObjective, threshold: 0.95 };
      onTargetConfigChange({ ...base, entryId });
    },
    [effectiveTargetConfig, onTargetConfigChange, targetConfig],
  );

  const handleObjectiveChange = useCallback(
    (objective: TargetObjective) => {
      const base =
        targetConfig ??
        effectiveTargetConfig ?? { entryId: targetOptions[0]?.id ?? "", objective: "higher", threshold: 0.95 };
      onTargetConfigChange({ ...base, objective });
    },
    [effectiveTargetConfig, onTargetConfigChange, targetConfig, targetOptions],
  );

  const gridCols = useMemo(() => {
    const n = Math.floor(Number.parseFloat(gridColsDraft.trim()));
    if (!Number.isFinite(n)) return DEFAULT_GRID_COLS;
    return Math.min(MAX_GRID_COLS, Math.max(MIN_GRID_COLS, n));
  }, [gridColsDraft]);

  const commitGridColsDraft = useCallback(() => {
    setGridColsDraft(String(gridCols));
  }, [gridCols]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  const interestingnessByEntryId = useMemo(() => interestingnessBreakdownForEntries(entries), [entries]);
  const displayed = useMemo(() => rankCurveStarerEntries(entries, rankBy), [entries, rankBy]);
  const commonBoundaryClusters = useMemo(() => computeCommonBoundaryClusters(entries), [entries]);
  const hasCommonBoundaries = commonBoundaryClusters.length > 0;

  /** Correlation finder uses raw training curves; LPD results are merged in when available. */
  const correlationEntries = useMemo((): CurveStarerAnalyzedEntry[] => {
    const curves = collectObservableTrainingCurves(nodes, edges);
    const lpdById = new Map(entries.map((e) => [e.entryId, e]));
    return curves.map((curve) => {
      const analyzed = lpdById.get(curve.entryId);
      return analyzed ?? { ...curve, lpd: null };
    });
  }, [edges, entries, nodes]);

  const canStartCorrelationStaring =
    correlationEntries.length >= 2 &&
    effectiveTargetConfig != null &&
    effectiveTargetConfig.entryId.trim() !== "";

  const phaseSegmentationProgressLabel =
    activeTab === "phase-segmentation" && busy && progress && progress.total > 0
      ? progress.completed > 0
        ? `Analyzed ${progress.completed} of ${progress.total} curves (${LPD_PREDICT_CONCURRENCY} in parallel on CPU)…`
        : `Starting LPD on ${progress.total} curves (${LPD_PREDICT_CONCURRENCY} in parallel on CPU)…`
      : null;

  const handleStartCorrelationStaring = useCallback(() => {
    if (!canStartCorrelationStaring) return;
    setCorrelationBusy(true);
    setStareRequestId((n) => n + 1);
  }, [canStartCorrelationStaring]);

  const handleCorrelationStareComplete = useCallback(() => {
    setCorrelationBusy(false);
  }, []);

  if (!open) return null;

  return createPortal(
    <div className="cr-modal-backdrop cr-curve-starer-backdrop" onMouseDown={handleBackdropMouseDown}>
      <div className="cr-modal cr-curve-starer-modal nodrag nopan" role="dialog" aria-labelledby="curve-starer-title">
        <header className="cr-curve-starer-modal__header">
          <div className="cr-curve-starer-modal__title-row">
            <div className="cr-curve-starer-modal__title-block">
              <h2 id="curve-starer-title" className="cr-modal__title">
                CurveStarer · Training dynamics
              </h2>
            </div>
            <button type="button" className="cr-modal__btn cr-modal__btn--ghost cr-curve-starer-modal__close" onClick={onClose}>
              Close
            </button>
          </div>
          <div className="cr-curve-starer-modal__tabs" role="tablist" aria-label="CurveStarer mode">
            {TAB_OPTIONS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={
                  activeTab === tab.id
                    ? "cr-curve-starer-modal__tab cr-curve-starer-modal__tab--active"
                    : "cr-curve-starer-modal__tab"
                }
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          {activeTab === "phase-segmentation" ? (
            <div className="cr-curve-starer-modal__toolbar">
              <button
                type="button"
                className="cr-modal__btn cr-modal__btn--primary cr-curve-starer-modal__start"
                disabled={busy}
                title="Run LPD analysis on wired training and observable viz curves"
                onClick={onStartAnalyze}
              >
                {busy ? "Staring…" : "Start Staring"}
              </button>
              <label className="cr-curve-starer-modal__cols-field">
                <span className="cr-curve-starer-modal__cols-label">Per row</span>
                <input
                  type="number"
                  className="cr-modal__input cr-curve-starer-modal__cols-input"
                  min={MIN_GRID_COLS}
                  max={MAX_GRID_COLS}
                  step={1}
                  inputMode="numeric"
                  value={gridColsDraft}
                  aria-label="Curves per row"
                  onChange={(e) => setGridColsDraft(e.target.value)}
                  onBlur={commitGridColsDraft}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      commitGridColsDraft();
                      (e.currentTarget as HTMLInputElement).blur();
                    }
                  }}
                />
              </label>
              <DiscreteMultiSelect
                label="Rank by"
                options={CURVE_STARER_RANK_BY_OPTIONS}
                value={rankBy}
                singleSelect
                matchModalInput
                disabled={busy && entries.length === 0}
                ariaLabel="Rank observable curves"
                onCommit={(next) => {
                  const raw = Array.isArray(next) ? next[0] : next;
                  onRankByChange(isCurveStarerRankBy(raw) ? raw : "default");
                }}
              />
              <button
                type="button"
                className={
                  showCommonBoundaries
                    ? "cr-modal__btn cr-curve-starer-modal__common-boundary cr-curve-starer-modal__common-boundary--active"
                    : "cr-modal__btn cr-curve-starer-modal__common-boundary"
                }
                disabled={!hasCommonBoundaries}
                title={
                  hasCommonBoundaries
                    ? `${commonBoundaryClusters.length} shared boundary cluster(s) across curves`
                    : "Need at least two curves with similar phase boundaries"
                }
                aria-pressed={showCommonBoundaries}
                onClick={() => setShowCommonBoundaries((v) => !v)}
              >
                Common Boundary
              </button>
            </div>
          ) : (
            <div className="cr-curve-starer-modal__toolbar">
              <button
                type="button"
                className="cr-modal__btn cr-modal__btn--primary cr-curve-starer-modal__start"
                disabled={correlationBusy || !canStartCorrelationStaring}
                title="Rank observables by Ranking R² (co-change with target)"
                onClick={handleStartCorrelationStaring}
              >
                {correlationBusy ? "Staring…" : "Start Staring"}
              </button>
              <DiscreteMultiSelect
                label="Target curve"
                options={
                  targetOptions.length > 0
                    ? targetOptions
                    : [{ id: "", label: "Train first (≥ 2 logged steps per viz)" }]
                }
                value={effectiveTargetConfig?.entryId ?? ""}
                singleSelect
                matchModalInput
                disabled={targetOptions.length === 0}
                ariaLabel="Target curve for correlation analysis"
                onCommit={(next) => {
                  const raw = Array.isArray(next) ? next[0] : next;
                  if (typeof raw === "string" && raw) handleTargetChange(raw);
                }}
              />
              <DiscreteMultiSelect
                label="Objective"
                options={[...TARGET_OBJECTIVE_OPTIONS]}
                value={effectiveTargetConfig?.objective ?? "higher"}
                singleSelect
                matchModalInput
                disabled={targetOptions.length === 0}
                ariaLabel="Target optimization direction"
                onCommit={(next) => {
                  const raw = Array.isArray(next) ? next[0] : next;
                  if (raw === "higher" || raw === "lower") handleObjectiveChange(raw);
                }}
              />
              {targetOptions.length === 0 ? (
                <span className="cr-curve-starer-modal__target-hint">
                  Wire training / observable viz to Trainer, run training, then pick a target curve.
                </span>
              ) : null}
            </div>
          )}
        </header>
        <div className="cr-curve-starer-modal__body">
          {phaseSegmentationProgressLabel ? (
            <p className="cr-curve-starer-modal__status">{phaseSegmentationProgressLabel}</p>
          ) : null}
          {activeTab === "phase-segmentation" && !busy && statusMessage ? (
            <p className="cr-curve-starer-modal__status cr-curve-starer-modal__status--warn">{statusMessage}</p>
          ) : null}
          {activeTab === "phase-segmentation" ? (
            displayed.length > 0 ? (
              <div
                className="cr-curve-starer-grid"
                style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}
                onPointerLeave={(e) => {
                  const related = e.relatedTarget;
                  if (related instanceof Element && related.closest(".cr-curve-starer-grid")) return;
                  setSyncT(null);
                }}
              >
                {displayed.map((entry) => (
                  <CurveStarerCell
                    key={entry.entryId}
                    entry={entry}
                    breakdown={interestingnessByEntryId.get(entry.entryId)}
                    showCommonBoundaries={showCommonBoundaries}
                    commonBoundaryClusters={commonBoundaryClusters}
                    syncT={syncT}
                    onSyncTChange={setSyncT}
                  />
                ))}
              </div>
            ) : busy ? (
              <p className="cr-curve-starer-modal__empty">
                First curves usually appear within a few seconds; larger batches keep filling in as each LPD run
                finishes.
              </p>
            ) : !statusMessage ? (
              <p className="cr-curve-starer-modal__empty">
                Click <strong>Start Staring</strong> to analyze wired training and observable viz curves (train first;
                each series needs ≥ 5 logged steps).
              </p>
            ) : null
          ) : effectiveTargetConfig ? (
            <CorrelationFinderPanel
              entries={correlationEntries}
              targetConfig={effectiveTargetConfig}
              nodes={nodes}
              edges={edges}
              staring={correlationBusy}
              stareRequested={stareRequestId}
              onStareComplete={handleCorrelationStareComplete}
            />
          ) : (
            <p className="cr-curve-starer-modal__empty">Configure a target curve after training.</p>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
