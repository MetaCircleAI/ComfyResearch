import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

export type LineChartBounds = { minX: number; maxX: number; minY: number; maxY: number };
type DragRect = { x0: number; y0: number; x1: number; y1: number };

export function useLineChartZoom(args: {
  bounds: LineChartBounds;
  enabled: boolean;
  left: number;
  right: number;
  top: number;
  bottom: number;
  initialZoom?: LineChartBounds | null;
  onZoomChange?: (next: LineChartBounds | null) => void;
}) {
  const { bounds, enabled, left, right, top, bottom, initialZoom = null, onZoomChange } = args;
  const [zoomBounds, setZoomBoundsState] = useState<LineChartBounds | null>(initialZoom);
  const [drag, setDrag] = useState<DragRect | null>(null);
  const dragRef = useRef<DragRect | null>(null);
  useEffect(() => {
    dragRef.current = drag;
  }, [drag]);
  const setZoomBounds = useCallback(
    (next: LineChartBounds | null) => {
      setZoomBoundsState(next);
      onZoomChange?.(next);
    },
    [onZoomChange],
  );

  const viewBounds = zoomBounds ?? bounds;
  const width = Math.max(1, right - left);
  const height = Math.max(1, bottom - top);

  const clampX = useCallback((x: number) => Math.min(right, Math.max(left, x)), [left, right]);
  const clampY = useCallback((y: number) => Math.min(bottom, Math.max(top, y)), [top, bottom]);

  const pxToX = useCallback(
    (px: number) => {
      const t = (clampX(px) - left) / width;
      return viewBounds.minX + t * (viewBounds.maxX - viewBounds.minX || 1);
    },
    [clampX, left, width, viewBounds],
  );

  const pxToY = useCallback(
    (py: number) => {
      const t = (clampY(py) - top) / height;
      return viewBounds.maxY - t * (viewBounds.maxY - viewBounds.minY || 1);
    },
    [clampY, top, height, viewBounds],
  );

  const onMouseDown = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (!enabled || e.button !== 0) return;
      const x = clampX(e.nativeEvent.offsetX);
      const y = clampY(e.nativeEvent.offsetY);
      if (x < left || x > right || y < top || y > bottom) return;
      e.preventDefault();
      e.stopPropagation();
      const next = { x0: x, y0: y, x1: x, y1: y };
      dragRef.current = next;
      setDrag(next);
    },
    [bottom, clampX, clampY, enabled, left, right, top],
  );

  const onMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (!dragRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      setDrag((prev) => {
        if (!prev) return null;
        const next = { ...prev, x1: clampX(e.nativeEvent.offsetX), y1: clampY(e.nativeEvent.offsetY) };
        dragRef.current = next;
        return next;
      });
    },
    [clampX, clampY],
  );

  const finishDrag = useCallback(() => {
    const d = dragRef.current;
    dragRef.current = null;
    setDrag(null);
    if (!d) return;
    const x0 = clampX(d.x0);
    const x1 = clampX(d.x1);
    const y0 = clampY(d.y0);
    const y1 = clampY(d.y1);
    if (Math.abs(x1 - x0) < 6 || Math.abs(y1 - y0) < 6) return;
    const minX = pxToX(Math.min(x0, x1));
    const maxX = pxToX(Math.max(x0, x1));
    const minY = pxToY(Math.max(y0, y1));
    const maxY = pxToY(Math.min(y0, y1));
    if (!(maxX > minX && maxY > minY)) return;
    setZoomBounds({ minX, maxX, minY, maxY });
  }, [clampX, clampY, pxToX, pxToY, setZoomBounds]);

  const resetZoom = useCallback(() => {
    setZoomBounds(null);
  }, [setZoomBounds]);

  const selectionRect = useMemo(() => {
    if (!drag) return null;
    const x = Math.min(clampX(drag.x0), clampX(drag.x1));
    const y = Math.min(clampY(drag.y0), clampY(drag.y1));
    const w = Math.abs(clampX(drag.x1) - clampX(drag.x0));
    const h = Math.abs(clampY(drag.y1) - clampY(drag.y0));
    return { x, y, w, h };
  }, [clampX, clampY, drag]);

  return {
    viewBounds,
    isZoomed: zoomBounds !== null,
    selectionRect,
    onMouseDown,
    onMouseMove,
    onMouseUp: finishDrag,
    onMouseLeave: finishDrag,
    resetZoom,
  };
}
