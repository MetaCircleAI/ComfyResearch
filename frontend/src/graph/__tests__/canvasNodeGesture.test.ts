// @vitest-environment jsdom
/**
 * Issue #169: pressing on a chart's drag-to-zoom surface must be an app
 * interaction, not a canvas-node drag gesture. The chart SVGs carry
 * `cr-tviz-chart nodrag nopan`, but the workspace redesign's manual `.nodrag`
 * drag fallback turned them back into drag handles — the interactive-press
 * selector is the layer that must exempt them.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  isInteractiveNodePressTarget,
  NODE_PRESS_INTERACTIVE_SELECTOR,
} from "../../components/canvasNodeGesture";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Minimal canvas-node DOM: wrapper > body > (children). */
function nodeBodyWith(child: Element): Element {
  const wrapper = document.createElement("div");
  wrapper.className = "react-flow__node";
  const body = document.createElement("div");
  body.className = "cr-node";
  wrapper.appendChild(body);
  body.appendChild(child);
  return child;
}

describe("isInteractiveNodePressTarget", () => {
  it("exempts presses inside a chart zoom surface (real SVG child, issue #169)", () => {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "cr-tviz-chart nodrag nopan");
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("class", "cr-tviz-chart__plot-bg");
    svg.appendChild(rect);
    nodeBodyWith(svg);
    // The actual pointerdown target is usually a child shape, not the svg root.
    expect(isInteractiveNodePressTarget(rect)).toBe(true);
    expect(isInteractiveNodePressTarget(svg)).toBe(true);
  });

  it("keeps plain node body presses as node gestures", () => {
    const label = document.createElement("span");
    nodeBodyWith(label);
    expect(isInteractiveNodePressTarget(label)).toBe(false);
  });

  it("keeps form controls interactive", () => {
    const input = document.createElement("input");
    nodeBodyWith(input);
    expect(isInteractiveNodePressTarget(input)).toBe(true);
  });

  it("leaves bare .nodrag label shells to the manual drag fallback", () => {
    // Parameter-label shells use `nodrag` solely to protect their inner input;
    // the shell itself must stay a node gesture (manualDragFallback contract).
    const shell = document.createElement("div");
    shell.className = "cr-comfy-widget nodrag nopan";
    const label = document.createElement("span");
    shell.appendChild(label);
    nodeBodyWith(shell);
    expect(isInteractiveNodePressTarget(label)).toBe(false);
  });
});

describe("ResearchCanvas wiring", () => {
  const canvasSource = readFileSync(
    resolve(process.cwd(), "src/components/ResearchCanvas.tsx"),
    "utf-8",
  );

  it("routes the press-time interactive check through the shared helper", () => {
    expect(canvasSource).toContain("isInteractiveNodePressTarget(");
  });

  it("no longer inlines the interactive selector", () => {
    // The pre-#169 inline selector; re-inlining it would let the chart-zoom
    // exemption silently drift out of the press path.
    expect(canvasSource).not.toMatch(/closest\(\s*["'`]input,\s*textarea/);
  });

  it("selector still covers the original interactive regions", () => {
    for (const part of [
      "input",
      "textarea",
      "select",
      "[contenteditable='true']",
      ".react-flow__handle",
      ".react-flow__resize-control",
      ".cr-node__header button",
      ".cr-tviz-chart",
    ]) {
      expect(NODE_PRESS_INTERACTIVE_SELECTOR).toContain(part);
    }
  });
});
