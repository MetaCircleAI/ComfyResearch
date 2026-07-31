/**
 * Plain-wheel canvas zoom (issue #170): gesture classification and scaling.
 *
 * The canvas wheel handler runs in the capture phase on `.cr-canvas-wrap`, so
 * it sees every wheel event first. Zooming may only steal the event when the
 * pointer is not over a surface with native wheel semantics: XYFlow `nowheel`
 * regions, form controls (Chrome scrolls values on focused number inputs),
 * and any actually-overflowing scrollable ancestor. A scrollable region never
 * releases the wheel to canvas zoom — not even at its top/bottom — so its
 * gesture semantics stay position-independent.
 */

const WHEEL_NATIVE_SELECTOR = [
  ".nowheel",
  "input",
  "textarea",
  "select",
  "[contenteditable='true']",
].join(", ");

const SCROLLABLE_OVERFLOW = /(auto|scroll|overlay)/;

function isScrollable(el: Element): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const style = getComputedStyle(el);
  return (
    (SCROLLABLE_OVERFLOW.test(style.overflowY) && el.scrollHeight > el.clientHeight) ||
    (SCROLLABLE_OVERFLOW.test(style.overflowX) && el.scrollWidth > el.clientWidth)
  );
}

/** May a plain (no Ctrl/Cmd) wheel event over `target` zoom the canvas? */
export function wheelTargetAllowsCanvasZoom(
  target: Element,
  boundary: Element,
  deltaX: number,
  deltaY: number,
): boolean {
  // Horizontal-dominant gestures are scroll/pan intent, never zoom.
  if (Math.abs(deltaX) > Math.abs(deltaY)) return false;
  if (target.closest(WHEEL_NATIVE_SELECTOR)) return false;
  for (let el: Element | null = target; el && el !== boundary; el = el.parentElement) {
    if (isScrollable(el)) return false;
  }
  return true;
}

/**
 * Apply the same exponential wheel scale used by XYFlow/d3-zoom. Pixel deltas
 * stay proportional, so a trackpad gesture is continuous instead of snapping
 * through toolbar zoom levels.
 */
export function getContinuousWheelZoom(
  currentZoom: number,
  deltaY: number,
  deltaMode: number,
  pinch: boolean,
  minZoom: number,
  maxZoom: number,
): number {
  const modeScale = deltaMode === 1 ? 0.05 : deltaMode ? 1 : 0.002;
  const gestureScale = pinch ? 10 : 1;
  const nextZoom = currentZoom * 2 ** (-deltaY * modeScale * gestureScale);
  return Math.min(maxZoom, Math.max(minZoom, nextZoom));
}
