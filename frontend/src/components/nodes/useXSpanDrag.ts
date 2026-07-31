import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";
import type { SLBounds } from "./scalarLineChartShared";
import {
  SL_PAD_L,
  SL_PAD_T,
  slInnerBottom,
  slInnerRight,
  slPxToPlotX,
  slScalarRawXFromPlotX,
} from "./scalarLineChartShared";

const MIN_DRAG_PX = 6;

type DragPx = { x0: number; x1: number };

/**
 * Horizontal span selection in plot area: maps to raw step bounds via inverse transform.
 */
export function useXSpanDrag(args: {
  viewBounds: SLBounds;
  logX: boolean;
  /** When true with ``logX``: x is log10(raw sweep value), not log10(step+1). */
  plainLogX: boolean;
  stepDomain: { min: number; max: number };
  enabled: boolean;
  left: number;
  right: number;
  top: number;
  bottom: number;
  onCommit: (span: { stepMin: number; stepMax: number } | null) => void;
}) {
  const { viewBounds, logX, plainLogX, stepDomain, enabled, left, right, top, bottom, onCommit } = args;
  const [drag, setDrag] = useState<DragPx | null>(null);
  const dragRef = useRef<DragPx | null>(null);

  useEffect(() => {
    dragRef.current = drag;
  }, [drag]);

  const clampPx = useCallback(
    (px: number) => Math.min(right, Math.max(left, px)),
    [left, right],
  );

  const finish = useCallback(() => {
    const d = dragRef.current;
    dragRef.current = null;
    setDrag(null);
    if (!d) {
      return;
    }
    const w = Math.abs(d.x1 - d.x0);
    if (w < MIN_DRAG_PX) {
      return;
    }
    const xa = clampPx(Math.min(d.x0, d.x1));
    const xb = clampPx(Math.max(d.x0, d.x1));
    const txA = slPxToPlotX(xa, viewBounds);
    const txB = slPxToPlotX(xb, viewBounds);
    let a = slScalarRawXFromPlotX(txA, logX, plainLogX);
    let b = slScalarRawXFromPlotX(txB, logX, plainLogX);
    if (a > b) [a, b] = [b, a];
    a = Math.max(stepDomain.min, Math.min(stepDomain.max, a));
    b = Math.max(stepDomain.min, Math.min(stepDomain.max, b));
    if (!(b > a)) {
      return;
    }
    onCommit({ stepMin: a, stepMax: b });
  }, [clampPx, logX, plainLogX, onCommit, stepDomain.max, stepDomain.min, viewBounds]);

  const onMouseDown = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (!enabled || e.button !== 0) return;
      const px = clampPx(e.nativeEvent.offsetX);
      const py = e.nativeEvent.offsetY;
      if (px < left || px > right || py < top || py > bottom) return;
      e.preventDefault();
      e.stopPropagation();
      const next = { x0: px, x1: px };
      dragRef.current = next;
      setDrag(next);
    },
    [bottom, clampPx, enabled, left, right, top],
  );

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (!dragRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      const px = clampPx(e.nativeEvent.offsetX);
      setDrag((prev) => {
        if (!prev) return null;
        const next = { ...prev, x1: px };
        dragRef.current = next;
        return next;
      });
    },
    [clampPx],
  );

  const selectionRect = drag
    ? {
        x: Math.min(drag.x0, drag.x1),
        y: top,
        w: Math.abs(drag.x1 - drag.x0),
        h: bottom - top,
      }
    : null;

  return {
    onMouseDown,
    onMouseMove,
    onMouseUp: finish,
    onMouseLeave: finish,
    selectionRect,
    isDragging: drag !== null,
  };
}

export function slPlotInnerRect() {
  return {
    left: SL_PAD_L,
    right: slInnerRight(),
    top: SL_PAD_T,
    bottom: slInnerBottom(),
  };
}
