import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";

type Point = { x: number; y: number };

const W = 520;
const H = 240;
const PAD = 24;

/** Above this many loaded points, collapse to `DEFAULT_ANCHOR_COUNT` for editing. */
const DENSE_CURVE_POINT_THRESHOLD = 24;
const DEFAULT_ANCHOR_COUNT = 16;
const MIN_ANCHORS = 2;
const MAX_ANCHORS = 64;

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function interpYFromSortedPoints(sorted: Point[], xq: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return Math.max(0, sorted[0]!.y);
  const xs = sorted.map((p) => p.x);
  const ys = sorted.map((p) => p.y);
  if (xq <= xs[0]!) return ys[0]!;
  if (xq >= xs[xs.length - 1]!) return ys[ys.length - 1]!;
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (xs[mid]! <= xq) lo = mid;
    else hi = mid;
  }
  const x0 = xs[lo]!;
  const x1 = xs[hi]!;
  const y0 = ys[lo]!;
  const y1 = ys[hi]!;
  const t = x1 === x0 ? 0 : (xq - x0) / (x1 - x0);
  return y0 + (y1 - y0) * t;
}

/** Evenly spaced x in [0, maxStep]; y from the polyline through sorted anchors. */
function resamplePolylineToAnchors(sorted: Point[], maxStep: number, n: number): Point[] {
  const nClamped = Math.max(MIN_ANCHORS, Math.min(MAX_ANCHORS, Math.round(n)));
  if (sorted.length < 2) return [...sorted];
  const out: Point[] = [];
  for (let i = 0; i < nClamped; i++) {
    const x = (i / (nClamped - 1)) * maxStep;
    out.push({ x, y: Math.max(0, interpYFromSortedPoints(sorted, x)) });
  }
  return out;
}

/** Map pointer position to SVG user space (0…W, 0…H) when the SVG is scaled via viewBox/CSS. */
function clientPointToSvgUser(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
): { x: number; y: number } {
  const r = svg.getBoundingClientRect();
  const rw = r.width || W;
  const rh = r.height || H;
  return {
    x: ((clientX - r.left) / rw) * W,
    y: ((clientY - r.top) / rh) * H,
  };
}

function sampleCurve(points: Point[], maxStep: number, count = 96): { steps: number[]; values: number[] } {
  if (points.length < 2) return { steps: [], values: [] };
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const outSteps: number[] = [];
  const outVals: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = (i / (count - 1)) * maxStep;
    outSteps.push(Math.round(x));
    outVals.push(Math.max(0, interpYFromSortedPoints(sorted, x)));
  }
  return { steps: outSteps, values: outVals };
}

export function TargetCurveModal({
  open,
  sourceSteps,
  sourceLoss,
  initialSteps,
  initialLoss,
  onCancel,
  onSave,
}: {
  open: boolean;
  sourceSteps: number[];
  sourceLoss: number[];
  initialSteps: number[];
  initialLoss: number[];
  onCancel: () => void;
  onSave: (steps: number[], values: number[]) => void;
}) {
  const maxStep = useMemo(() => {
    const src = sourceSteps.length ? Math.max(...sourceSteps) : 0;
    const init = initialSteps.length ? Math.max(...initialSteps) : 0;
    return Math.max(10, src, init);
  }, [initialSteps, sourceSteps]);
  const maxY = useMemo(() => {
    const vals = [...sourceLoss, ...initialLoss].filter((x) => Number.isFinite(x) && x >= 0);
    if (!vals.length) return 1;
    return Math.max(1e-6, Math.max(...vals) * 1.1);
  }, [initialLoss, sourceLoss]);
  const [points, setPoints] = useState<Point[]>([]);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [anchorTargetCount, setAnchorTargetCount] = useState(String(DEFAULT_ANCHOR_COUNT));
  const svgRef = useRef<SVGSVGElement | null>(null);

  const applyResample = useCallback(() => {
    const n = Math.round(Number.parseFloat(anchorTargetCount));
    if (!Number.isFinite(n)) return;
    setPoints((prev) => {
      const sorted = [...prev].sort((a, b) => a.x - b.x);
      if (sorted.length < 2) return prev;
      return resamplePolylineToAnchors(sorted, maxStep, n);
    });
  }, [anchorTargetCount, maxStep]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onCancel();
    },
    [onCancel],
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, open]);

  useEffect(() => {
    if (!open) return;
    if (initialSteps.length >= 2 && initialLoss.length === initialSteps.length) {
      const raw = initialSteps.map((s, i) => ({
        x: clamp01(s / maxStep) * maxStep,
        y: Math.max(0, initialLoss[i] ?? 0),
      }));
      const sorted = [...raw].sort((a, b) => a.x - b.x);
      if (sorted.length > DENSE_CURVE_POINT_THRESHOLD) {
        setPoints(resamplePolylineToAnchors(sorted, maxStep, DEFAULT_ANCHOR_COUNT));
        setAnchorTargetCount(String(DEFAULT_ANCHOR_COUNT));
      } else {
        setPoints(sorted);
        setAnchorTargetCount(String(Math.max(MIN_ANCHORS, Math.min(MAX_ANCHORS, sorted.length))));
      }
      return;
    }
    if (sourceSteps.length >= 2 && sourceLoss.length === sourceSteps.length) {
      const coarse = [0, 0.25, 0.5, 0.75, 1].map((r) => {
        const idx = Math.min(sourceSteps.length - 1, Math.round(r * (sourceSteps.length - 1)));
        return { x: sourceSteps[idx] ?? r * maxStep, y: sourceLoss[idx] ?? 0 };
      });
      setPoints(coarse);
      setAnchorTargetCount(String(coarse.length));
      return;
    }
    setPoints([
      { x: 0, y: maxY * 0.9 },
      { x: maxStep, y: maxY * 0.1 },
    ]);
    setAnchorTargetCount("2");
  }, [initialLoss, initialSteps, maxStep, maxY, open, sourceLoss, sourceSteps]);

  const toPx = (p: Point) => ({
    x: PAD + (p.x / maxStep) * (W - PAD * 2),
    y: PAD + (1 - p.y / maxY) * (H - PAD * 2),
  });
  const fromPx = (px: number, py: number): Point => ({
    x: clamp01((px - PAD) / (W - PAD * 2)) * maxStep,
    y: clamp01(1 - (py - PAD) / (H - PAD * 2)) * maxY,
  });

  if (!open) return null;

  const sorted = [...points].sort((a, b) => a.x - b.x);
  const targetPath = sorted
    .map((p, i) => {
      const pp = toPx(p);
      return `${i === 0 ? "M" : "L"}${pp.x.toFixed(2)},${pp.y.toFixed(2)}`;
    })
    .join(" ");
  const sourcePath =
    sourceSteps.length >= 2 && sourceLoss.length === sourceSteps.length
      ? sourceSteps
          .map((s, i) => {
            const pp = toPx({ x: s, y: sourceLoss[i] ?? 0 });
            return `${i === 0 ? "M" : "L"}${pp.x.toFixed(2)},${pp.y.toFixed(2)}`;
          })
          .join(" ")
      : "";

  const node = (
    <div className="cr-modal-backdrop" style={{ zIndex: 10080 }} onMouseDown={handleBackdropMouseDown}>
      <div className="cr-modal cr-target-curve-modal" role="dialog" aria-modal="true">
        <h2 className="cr-modal__title">Draw target curve</h2>
        <p className="cr-modal__hint">
          Click in the board to add points. Drag points to shape the curve. Gray = current source curve.
        </p>
        <div className="cr-target-curve-anchor-tools">
          <label className="cr-target-curve-anchor-tools__label">
            Anchor points
            <input
              className="cr-input cr-target-curve-anchor-tools__input"
              type="number"
              min={MIN_ANCHORS}
              max={MAX_ANCHORS}
              step={1}
              value={anchorTargetCount}
              onChange={(e) => setAnchorTargetCount(e.target.value)}
              aria-describedby="target-curve-anchor-help"
            />
          </label>
          <button
            type="button"
            className="cr-modal__btn"
            disabled={points.length < 2}
            onClick={() => applyResample()}
          >
            Resample
          </button>
          <span id="target-curve-anchor-help" className="cr-target-curve-anchor-tools__hint">
            Resample replaces anchors with N evenly spaced samples along x from the current shape.
          </span>
        </div>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          height="auto"
          preserveAspectRatio="xMidYMid meet"
          className="cr-target-curve-board"
          onMouseDown={(e) => {
            const svg = svgRef.current;
            if (!svg) return;
            const { x: ux, y: uy } = clientPointToSvgUser(svg, e.clientX, e.clientY);
            const near = sorted.findIndex((p) => {
              const t = toPx(p);
              return (t.x - ux) ** 2 + (t.y - uy) ** 2 < 9 ** 2;
            });
            if (near >= 0) {
              setDragIdx(near);
              return;
            }
            setPoints((prev) => [...prev, fromPx(ux, uy)]);
          }}
          onMouseMove={(e) => {
            if (dragIdx == null) return;
            const svg = svgRef.current;
            if (!svg) return;
            const { x: ux, y: uy } = clientPointToSvgUser(svg, e.clientX, e.clientY);
            const next = fromPx(ux, uy);
            setPoints((prev) => {
              const arr = [...prev].sort((a, b) => a.x - b.x);
              arr[dragIdx] = next;
              return arr;
            });
          }}
          onMouseUp={() => setDragIdx(null)}
          onMouseLeave={() => setDragIdx(null)}
        >
          <rect x={PAD} y={PAD} width={W - PAD * 2} height={H - PAD * 2} rx={6} className="cr-target-curve-bg" />
          {sourcePath ? <path d={sourcePath} className="cr-target-curve-source" fill="none" /> : null}
          {targetPath ? <path d={targetPath} className="cr-target-curve-target" fill="none" /> : null}
          {sorted.map((p, i) => {
            const pp = toPx(p);
            return <circle key={`${i}:${p.x}`} cx={pp.x} cy={pp.y} r={4.5} className="cr-target-curve-point" />;
          })}
        </svg>
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={() => setPoints([])}>
            Clear
          </button>
          <button type="button" className="cr-modal__btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="cr-modal__btn cr-modal__btn--primary"
            disabled={points.length < 2}
            onClick={() => {
              const sampled = sampleCurve(points, maxStep);
              if (sampled.steps.length < 2) return;
              onSave(sampled.steps, sampled.values);
            }}
          >
            Save target curve
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
