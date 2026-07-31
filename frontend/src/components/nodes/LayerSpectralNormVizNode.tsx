import type { NodeProps } from "@xyflow/react";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { VizSocketsBar } from "./VizSocketsBar";
/* Theme-aware via tokens.css; classic values pin the exact legacy hex per position. */
const COLORS = ["var(--cr-chart-1)", "var(--cr-chart-2)", "var(--cr-chart-3)", "var(--cr-chart-4)", "var(--cr-chart-9)", "var(--cr-chart-6)"];
function rowsOf(raw: unknown): number[][] { return Array.isArray(raw) ? raw.filter(Array.isArray).map((r) => r.map(Number)) : []; }
/** Separate semantic view: these are layer norms, not Hessian eigenvalues. */
export function LayerSpectralNormVizNode({ id, data, selected }: NodeProps) {
  const d = (data ?? {}) as { pairedObservableId?: string; valueHistories?: unknown; stepTicks?: unknown; seriesLabels?: unknown };
  const rows = rowsOf(d.valueHistories); const steps = Array.isArray(d.stepTicks) ? d.stepTicks.map(Number) : [];
  const labels = Array.isArray(d.seriesLabels) ? d.seriesLabels.filter((x): x is string => typeof x === "string") : [];
  const pts = rows.flatMap((r, ri) => r.map((y, xi) => ({ x: steps[xi] ?? xi, y, ri })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)));
  const maxX = Math.max(1, ...pts.map((p) => p.x)); const minY = Math.min(0, ...pts.map((p) => p.y)); const maxY = Math.max(1, ...pts.map((p) => p.y));
  const xy = (x: number, y: number) => `${28 + 184 * x / maxX},${112 - 88 * (y - minY) / Math.max(1e-9, maxY - minY)}`;
  return <div className={`cr-node cr-node--observable-viz-layer-spectral-norm${selected ? " cr-node--selected" : ""}`}>
    <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title="Layer spectral norm" />
    <div className="cr-node__body cr-node__body--compact"><VizSocketsBar />
      <svg width="232" height="132" role="img" aria-label="Layer spectral norm curves"><line x1="28" y1="112" x2="212" y2="112" stroke="currentColor" opacity=".35" /><line x1="28" y1="12" x2="28" y2="112" stroke="currentColor" opacity=".35" />
        {rows.map((r, ri) => <path key={ri} d={r.map((y, xi) => Number.isFinite(y) ? `${xi === 0 ? "M" : "L"}${xy(steps[xi] ?? xi, y)}` : "").join(" ")} fill="none" stroke={COLORS[ri % COLORS.length]} strokeWidth="2" />)}
      </svg><div className="cr-observable-hint">{rows.length ? rows.map((_, i) => labels[i] ?? `Linear layer ${i + 1}`).join(" · ") : "Waiting for Linear-layer norm samples."}</div>
    </div></div>;
}
