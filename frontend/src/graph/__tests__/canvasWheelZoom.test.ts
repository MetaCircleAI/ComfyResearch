// @vitest-environment jsdom
/**
 * Issue #170: plain mouse wheel zooms the canvas (no Ctrl needed). The wheel
 * capture handler must keep native scrolling alive inside scrollable regions
 * (~39 overflow:auto surfaces, none of which use XYFlow's `nowheel`), while
 * every trackpad delta scales the canvas continuously.
 */
import { describe, expect, it } from "vitest";

import {
  getContinuousWheelZoom,
  wheelTargetAllowsCanvasZoom,
} from "../../components/canvasWheelZoom";

const SVG_NS = "http://www.w3.org/2000/svg";

function makeBoundary(): HTMLElement {
  const boundary = document.createElement("div");
  boundary.className = "cr-canvas-wrap";
  document.body.appendChild(boundary);
  return boundary;
}

function makeScrollable(overflowY: string, scrollHeight: number, clientHeight: number): HTMLElement {
  const el = document.createElement("div");
  el.style.overflowY = overflowY;
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight });
  Object.defineProperty(el, "clientHeight", { value: clientHeight });
  return el;
}

describe("wheelTargetAllowsCanvasZoom", () => {
  it("allows plain surfaces (node body, pane)", () => {
    const boundary = makeBoundary();
    const label = document.createElement("span");
    boundary.appendChild(label);
    expect(wheelTargetAllowsCanvasZoom(label, boundary, 0, 100)).toBe(true);
  });

  it("allows chart SVG children (real SVG rect)", () => {
    const boundary = makeBoundary();
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "cr-tviz-chart nodrag nopan");
    const rect = document.createElementNS(SVG_NS, "rect");
    svg.appendChild(rect);
    boundary.appendChild(svg);
    expect(wheelTargetAllowsCanvasZoom(rect, boundary, 0, 100)).toBe(true);
  });

  it("never releases a scrollable ancestor to canvas zoom, even at top or bottom", () => {
    const boundary = makeBoundary();
    // Scrollable content: overflow-y auto with real overflow. Scroll position
    // is deliberately NOT consulted — top/bottom must keep native semantics.
    const scroller = makeScrollable("auto", 500, 100);
    const inner = document.createElement("p");
    scroller.appendChild(inner);
    boundary.appendChild(scroller);
    expect(wheelTargetAllowsCanvasZoom(inner, boundary, 0, 100)).toBe(false);
    expect(wheelTargetAllowsCanvasZoom(inner, boundary, 0, -100)).toBe(false);
  });

  it("ignores non-overflowing overflow:auto shells", () => {
    const boundary = makeBoundary();
    const shell = makeScrollable("auto", 100, 100);
    const inner = document.createElement("p");
    shell.appendChild(inner);
    boundary.appendChild(shell);
    expect(wheelTargetAllowsCanvasZoom(inner, boundary, 0, 100)).toBe(true);
  });

  it("blocks horizontal-dominant wheel gestures", () => {
    const boundary = makeBoundary();
    const label = document.createElement("span");
    boundary.appendChild(label);
    expect(wheelTargetAllowsCanvasZoom(label, boundary, 120, 30)).toBe(false);
  });

  it("keeps nowheel and form controls native", () => {
    const boundary = makeBoundary();
    const nowheel = document.createElement("div");
    nowheel.className = "nowheel";
    const child = document.createElement("span");
    nowheel.appendChild(child);
    const textarea = document.createElement("textarea");
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    boundary.append(nowheel, textarea, editable);
    expect(wheelTargetAllowsCanvasZoom(child, boundary, 0, 100)).toBe(false);
    expect(wheelTargetAllowsCanvasZoom(textarea, boundary, 0, 100)).toBe(false);
    expect(wheelTargetAllowsCanvasZoom(editable, boundary, 0, 100)).toBe(false);
  });

  it("does not consult the boundary itself", () => {
    const boundary = makeScrollable("auto", 500, 100);
    boundary.className = "cr-canvas-wrap";
    document.body.appendChild(boundary);
    const label = document.createElement("span");
    boundary.appendChild(label);
    expect(wheelTargetAllowsCanvasZoom(label, boundary, 0, 100)).toBe(true);
  });
});

describe("continuous wheel zoom", () => {
  it("maps each pixel delta proportionally instead of waiting for a zoom level", () => {
    expect(getContinuousWheelZoom(1, 10, 0, false, 0.25, 4)).toBeCloseTo(2 ** -0.02);
    expect(getContinuousWheelZoom(1, 20, 0, false, 0.25, 4)).toBeCloseTo(2 ** -0.04);
  });

  it("zooms in for negative deltas and out for positive deltas", () => {
    expect(getContinuousWheelZoom(1, -10, 0, false, 0.25, 4)).toBeGreaterThan(1);
    expect(getContinuousWheelZoom(1, 10, 0, false, 0.25, 4)).toBeLessThan(1);
  });

  it("normalizes line/page modes and clamps the configured zoom range", () => {
    expect(getContinuousWheelZoom(1, 1, 1, false, 0.25, 4)).toBeCloseTo(2 ** -0.05);
    expect(getContinuousWheelZoom(1, 10, 2, false, 0.25, 4)).toBe(0.25);
    expect(getContinuousWheelZoom(1, -10, 2, false, 0.25, 4)).toBe(4);
  });

  it("uses the stronger pinch scale without quantizing the result", () => {
    expect(getContinuousWheelZoom(1, 1, 0, true, 0.25, 4)).toBeCloseTo(2 ** -0.02);
  });
});
