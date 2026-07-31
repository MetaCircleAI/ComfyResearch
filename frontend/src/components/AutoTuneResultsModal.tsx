import { useCallback, useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";
import type { AutoTuneComparisonResult, AutoTuneRankedCurve } from "./nodes/trainerDefaults";
import { slFormatYTick } from "./nodes/scalarLineChartShared";

const CHART_W = 640;
const CHART_H = 280;
const PAD_L = 44;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 36;

function formatParamCell(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) {
    const x = v;
    if (Number.isInteger(x) || Math.abs(x - Math.round(x)) < 1e-6 * Math.max(1, Math.abs(x))) {
      return String(Math.round(x));
    }
    return x.toPrecision(4);
  }
  if (v === null || v === undefined) return "—";
  return String(v);
}

/** Column order: group by last path segment, then full key (node ids differ). */
function collectSortedParamKeys(ranked: { params: Record<string, unknown> }[]): string[] {
  const s = new Set<string>();
  for (const r of ranked) {
    for (const k of Object.keys(r.params)) s.add(k);
  }
  return [...s].sort((a, b) => {
    const la = a.includes(".") ? a.slice(a.lastIndexOf(".") + 1) : a;
    const lb = b.includes(".") ? b.slice(b.lastIndexOf(".") + 1) : b;
    const c = la.localeCompare(lb);
    if (c !== 0) return c;
    return a.localeCompare(b);
  });
}

function shortParamHeader(fullKey: string): string {
  const i = fullKey.lastIndexOf(".");
  if (i <= 0) return fullKey;
  return fullKey.slice(i + 1);
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

function boundsForSeriesList(
  series: { steps: number[]; vals: number[]; visible: boolean }[],
): { minX: number; maxX: number; minY: number; maxY: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const s of series) {
    if (!s.visible) continue;
    for (let i = 0; i < s.steps.length; i++) {
      const x = s.steps[i]!;
      const y = s.vals[i]!;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }
  const padY = (maxY - minY) * 0.06 || 0.05;
  return {
    minX,
    maxX: maxX === minX ? minX + 1 : maxX,
    minY: Math.max(0, minY - padY),
    maxY: maxY + padY,
  };
}

function buildPath(steps: number[], vals: number[], b: { minX: number; maxX: number; minY: number; maxY: number }): string {
  if (steps.length < 2 || vals.length !== steps.length) return "";
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const spanX = b.maxX - b.minX || 1;
  const spanY = b.maxY - b.minY || 1;
  return steps
    .map((s, i) => {
      const v = vals[i]!;
      const px = PAD_L + (innerW * (s - b.minX)) / spanX;
      const py = PAD_T + innerH * (1 - (v - b.minY) / spanY);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

type SeriesKey = "original" | "target" | "rank1" | "rank2" | "rank3" | "rank4";

const SERIES_META: { key: SeriesKey; label: string; color: string; dashed?: boolean }[] = [
  { key: "original", label: "Original (pre-tune)", color: "#9aa3b2" },
  { key: "target", label: "Target", color: "#5fd38d", dashed: true },
  { key: "rank1", label: "Best tuned (#1)", color: "#6eb6ff" },
  { key: "rank2", label: "Runner-up #2", color: "#ffb86b" },
  { key: "rank3", label: "Runner-up #3", color: "#c792ea" },
  { key: "rank4", label: "Runner-up #4", color: "#f78c6c" },
];

export function AutoTuneResultsModal({
  open,
  result,
  onClose,
}: {
  open: boolean;
  result: AutoTuneComparisonResult | null;
  onClose: () => void;
}) {
  const [vis, setVis] = useState<Record<SeriesKey, boolean>>(() => ({
    original: true,
    target: true,
    rank1: true,
    rank2: true,
    rank3: true,
    rank4: true,
  }));

  useEffect(() => {
    if (!open || !result) return;
    setVis({
      original: true,
      target: true,
      rank1: true,
      rank2: true,
      rank3: true,
      rank4: true,
    });
  }, [open, result]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  const seriesForBounds = useMemo(() => {
    if (!result) return [];
    const out: { steps: number[]; vals: number[]; visible: boolean }[] = [];
    const push = (steps: number[], vals: number[], key: SeriesKey) => {
      if (steps.length >= 2 && vals.length === steps.length) {
        out.push({ steps, vals, visible: vis[key] });
      }
    };
    push(result.baselineStepTicks, result.baselineLossHistory, "original");
    push(result.targetStepTicks, result.targetLossHistory, "target");
    (["rank1", "rank2", "rank3", "rank4"] as const).forEach((slot, i) => {
      const row = result.ranked[i];
      if (row) push(row.stepTicks, row.lossHistory, slot);
    });
    return out;
  }, [result, vis]);

  const b = useMemo(() => boundsForSeriesList(seriesForBounds), [seriesForBounds]);

  const xTicks = useMemo(() => {
    const span = b.maxX - b.minX;
    const step = niceStep(span / 5);
    const out: number[] = [];
    let t = Math.ceil(b.minX / step) * step;
    while (t <= b.maxX + step * 1e-9 && out.length < 10) {
      if (t >= b.minX - 1e-9) out.push(t);
      t += step;
    }
    return out.length ? out : [b.minX, b.maxX];
  }, [b]);

  const yTicks = useMemo(() => {
    const span = b.maxY - b.minY;
    const step = niceStep(span / 5);
    const out: number[] = [];
    let t = Math.ceil(b.minY / step) * step;
    while (t <= b.maxY + step * 1e-9 && out.length < 10) {
      if (t >= b.minY - 1e-9) out.push(t);
      t += step;
    }
    return out.length ? out : [b.minY, b.maxY];
  }, [b]);

  const rankedTableRows = useMemo(() => {
    if (!result) return [];
    const rows: { slot: "rank1" | "rank2" | "rank3" | "rank4"; label: string; color: string; row: AutoTuneRankedCurve }[] =
      [];
    (["rank1", "rank2", "rank3", "rank4"] as const).forEach((slot, i) => {
      const r = result.ranked[i];
      if (!r || r.stepTicks.length < 2) return;
      const meta = SERIES_META.find((m) => m.key === slot);
      rows.push({ slot, label: meta?.label ?? slot, color: meta?.color ?? "#fff", row: r });
    });
    return rows;
  }, [result]);

  const paramColumns = useMemo(() => {
    if (!result?.ranked.length) return [];
    return collectSortedParamKeys(result.ranked.filter((r) => r.stepTicks.length >= 2));
  }, [result]);

  if (!open || !result) return null;

  const innerBottom = PAD_T + (CHART_H - PAD_T - PAD_B);
  const innerRight = CHART_W - PAD_R;

  const pathFor = (steps: number[], vals: number[], key: SeriesKey) =>
    vis[key] ? buildPath(steps, vals, b) : "";

  const node = (
    <div className="cr-modal-backdrop" style={{ zIndex: 10080 }} onMouseDown={handleBackdropMouseDown}>
      <div className="cr-modal cr-auto-tune-results-modal" role="dialog" aria-modal="true">
        <h2 className="cr-modal__title">Auto-tune comparison</h2>
        <p className="cr-modal__hint">
          Toggle curves below. Best score:{" "}
          {Number.isFinite(result.bestScore) ? result.bestScore.toExponential(4) : "—"}
        </p>
        <div className="cr-auto-tune-results-layout">
          <div className="cr-auto-tune-results-legend nodrag nopan">
            {SERIES_META.map((meta) => {
              const rankIdx = { rank1: 0, rank2: 1, rank3: 2, rank4: 3 }[meta.key];
              const row =
                meta.key === "original" || meta.key === "target"
                  ? null
                  : rankIdx !== undefined
                    ? result.ranked[rankIdx]
                    : null;
              const hasData =
                meta.key === "original"
                  ? result.baselineStepTicks.length >= 2
                  : meta.key === "target"
                    ? result.targetStepTicks.length >= 2
                    : !!row && row.stepTicks.length >= 2;
              if (!hasData) return null;
              return (
                <label key={meta.key} className="cr-auto-tune-results-legend__row">
                  <input
                    type="checkbox"
                    checked={vis[meta.key]}
                    onChange={(e) => setVis((v) => ({ ...v, [meta.key]: e.target.checked }))}
                  />
                  <span className="cr-auto-tune-results-legend__swatch" style={{ background: meta.color }} />
                  <span className="cr-auto-tune-results-legend__text">
                    <span className="cr-auto-tune-results-legend__label">{meta.label}</span>
                    {row ? (
                      <span className="cr-auto-tune-results-legend__params">score {row.score.toExponential(3)}</span>
                    ) : null}
                  </span>
                </label>
              );
            })}
          </div>
          <svg className="cr-auto-tune-results-chart" viewBox={`0 0 ${CHART_W} ${CHART_H}`} width={CHART_W} height={CHART_H}>
            <rect
              x={PAD_L}
              y={PAD_T}
              width={CHART_W - PAD_L - PAD_R}
              height={CHART_H - PAD_T - PAD_B}
              rx={6}
              className="cr-auto-tune-results-chart__bg"
            />
            {yTicks.map((yt) => {
              const py = PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - (yt - b.minY) / (b.maxY - b.minY || 1));
              return (
                <line
                  key={`gy-${yt}`}
                  x1={PAD_L}
                  y1={py}
                  x2={innerRight}
                  y2={py}
                  className="cr-auto-tune-results-chart__grid"
                />
              );
            })}
            {xTicks.map((xt) => {
              const px = PAD_L + ((CHART_W - PAD_L - PAD_R) * (xt - b.minX)) / (b.maxX - b.minX || 1);
              return (
                <line
                  key={`gx-${xt}`}
                  x1={px}
                  y1={PAD_T}
                  x2={px}
                  y2={innerBottom}
                  className="cr-auto-tune-results-chart__grid"
                />
              );
            })}
            <line x1={PAD_L} y1={innerBottom} x2={innerRight} y2={innerBottom} className="cr-auto-tune-results-chart__axis" />
            <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={innerBottom} className="cr-auto-tune-results-chart__axis" />
            {xTicks.map((xt) => {
              const px = PAD_L + ((CHART_W - PAD_L - PAD_R) * (xt - b.minX)) / (b.maxX - b.minX || 1);
              return (
                <text
                  key={`xt-${xt}`}
                  x={px}
                  y={innerBottom + 14}
                  textAnchor="middle"
                  className="cr-auto-tune-results-chart__tick-label"
                >
                  {Math.abs(xt - Math.round(xt)) < 1e-6 ? String(Math.round(xt)) : xt.toFixed(1)}
                </text>
              );
            })}
            {yTicks.map((yt) => {
              const py = PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - (yt - b.minY) / (b.maxY - b.minY || 1));
              return (
                <text
                  key={`yt-${yt}`}
                  x={PAD_L - 8}
                  y={py}
                  dominantBaseline="middle"
                  textAnchor="end"
                  className="cr-auto-tune-results-chart__tick-label"
                >
                  {slFormatYTick(yt, false)}
                </text>
              );
            })}
            {pathFor(result.baselineStepTicks, result.baselineLossHistory, "original") ? (
              <path
                d={pathFor(result.baselineStepTicks, result.baselineLossHistory, "original")}
                fill="none"
                stroke="#9aa3b2"
                strokeWidth={2}
              />
            ) : null}
            {pathFor(result.targetStepTicks, result.targetLossHistory, "target") ? (
              <path
                d={pathFor(result.targetStepTicks, result.targetLossHistory, "target")}
                fill="none"
                stroke="#5fd38d"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            ) : null}
            {(["rank1", "rank2", "rank3", "rank4"] as const).map((slot, i) => {
              const row = result.ranked[i];
              if (!row || row.stepTicks.length < 2) return null;
              const key = slot;
              const color = SERIES_META.find((m) => m.key === key)?.color ?? "#fff";
              const d = pathFor(row.stepTicks, row.lossHistory, key);
              if (!d) return null;
              return <path key={slot} d={d} fill="none" stroke={color} strokeWidth={2} />;
            })}
            <text
              x={PAD_L + (CHART_W - PAD_L - PAD_R) / 2}
              y={CHART_H - 4}
              textAnchor="middle"
              className="cr-auto-tune-results-chart__axis-title"
            >
              step
            </text>
            <text
              x={10}
              y={PAD_T + (CHART_H - PAD_T - PAD_B) / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              transform={`rotate(-90, 10, ${PAD_T + (CHART_H - PAD_T - PAD_B) / 2})`}
              className="cr-auto-tune-results-chart__axis-title"
            >
              loss
            </text>
          </svg>
        </div>
        {rankedTableRows.length > 0 ? (
          <div className="cr-auto-tune-params-table-section nodrag nopan">
            <h3 className="cr-auto-tune-params-table-section__title">Hyperparameters by runner</h3>
            <div className="cr-auto-tune-params-table-wrap">
              <table className="cr-auto-tune-params-table">
                <thead>
                  <tr>
                    <th scope="col" className="cr-auto-tune-params-table__th cr-auto-tune-params-table__th--runner">
                      Runner
                    </th>
                    <th scope="col" className="cr-auto-tune-params-table__th cr-auto-tune-params-table__th--score">
                      Score
                    </th>
                    {paramColumns.map((pk) => (
                      <th
                        key={pk}
                        scope="col"
                        className="cr-auto-tune-params-table__th cr-auto-tune-params-table__th--param"
                        title={pk}
                      >
                        {shortParamHeader(pk)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rankedTableRows.map(({ slot, label, color, row }) => (
                    <tr key={slot} className="cr-auto-tune-params-table__tr">
                      <th
                        scope="row"
                        className="cr-auto-tune-params-table__td cr-auto-tune-params-table__td--runner"
                        style={{ borderLeft: `3px solid ${color}` }}
                      >
                        <span className="cr-auto-tune-params-table__runner-swatch" style={{ background: color }} />
                        {label}
                      </th>
                      <td className="cr-auto-tune-params-table__td cr-auto-tune-params-table__td--num">
                        {row.score.toExponential(4)}
                      </td>
                      {paramColumns.map((pk) => (
                        <td key={pk} className="cr-auto-tune-params-table__td cr-auto-tune-params-table__td--num">
                          {formatParamCell(row.params[pk])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
