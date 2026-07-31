import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// vitest runs with cwd = frontend/
const tokensCss = readFileSync(path.resolve(process.cwd(), "src/tokens.css"), "utf8");

/** Extract `--name: value;` pairs from the given CSS block text. */
function parseVars(block: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[m[1]!] = m[2]!.trim();
  return out;
}

function blockOf(selectorRe: RegExp): string {
  const m = tokensCss.match(new RegExp(selectorRe.source + String.raw`\s*\{([\s\S]*?)\n\}`));
  if (!m) throw new Error(`selector not found: ${selectorRe}`);
  return m[1]!;
}

const studio = parseVars(blockOf(/:root/));
const classic = parseVars(blockOf(/:root\[data-cr-theme="classic"\]/));
const paper = parseVars(blockOf(/:root\[data-cr-theme="paper"\]/));

const SEMANTIC_COLOR = [
  "--cr-canvas", "--cr-canvas-dot", "--cr-surface-1", "--cr-surface-2", "--cr-surface-3",
  "--cr-hairline", "--cr-hairline-2", "--cr-hairline-strong",
  "--cr-text-1", "--cr-text-2", "--cr-text-3", "--cr-text-4",
  "--cr-accent", "--cr-accent-hover", "--cr-danger", "--cr-success",
  "--cr-accent-dataset", "--cr-accent-model", "--cr-accent-optimizer", "--cr-accent-loss",
  "--cr-accent-trainer", "--cr-accent-checkpoint", "--cr-accent-where", "--cr-accent-tensor",
  "--cr-accent-observable", "--cr-accent-hypothesis", "--cr-accent-comment",
  "--cr-chart-1", "--cr-chart-2", "--cr-chart-3", "--cr-chart-4", "--cr-chart-5", "--cr-chart-6",
  "--cr-chart-7", "--cr-chart-8", "--cr-chart-9",
];
const THEME_INDEPENDENT = [
  "--cr-font-ui", "--cr-font-display", "--cr-font-mono", "--cr-mono",
  "--cr-r-sm", "--cr-r-md", "--cr-r-lg", "--cr-shadow-rest", "--cr-shadow-float",
];
const LEGACY = [
  "--cr-bg", "--cr-panel", "--cr-border", "--cr-text", "--cr-muted",
  "--cr-fg", "--cr-text-muted", "--cr-btn", "--cr-btn-primary", "--cr-library-accent",
];

// -- WCAG helpers (sRGB) --
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as [number, number, number];
}
function luminance([r, g, b]: [number, number, number]): number {
  const f = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function contrast(fg: string, bg: string): number {
  const [l1, l2] = [luminance(hexToRgb(fg)), luminance(hexToRgb(bg))];
  const [hi, lo] = l1 > l2 ? [l1, l2] : [l2, l1];
  return (hi + 0.05) / (lo + 0.05);
}
/**
 * `color-mix(in srgb, top A%, transparent)` composited over an OPAQUE base
 * equals linear per-channel interpolation: top*A + base*(1-A). This helper
 * models the spec §4.2 band (13% accent over surface-1) and title tint
 * (45% accent into text-1) as they actually render.
 */
function compositeOver(top: string, base: string, alpha: number): string {
  const [tr, tg, tb] = hexToRgb(top);
  const [br, bg2, bb] = hexToRgb(base);
  const c = (x: number, y: number) => Math.round(x * alpha + y * (1 - alpha));
  return `#${[c(tr, br), c(tg, bg2), c(tb, bb)].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

describe("design tokens", () => {
  it("defines every semantic color token in the studio (:root) palette", () => {
    for (const t of SEMANTIC_COLOR) expect(studio[t], t).toBeTruthy();
  });
  it("overrides every semantic color token in the classic palette", () => {
    for (const t of SEMANTIC_COLOR) expect(classic[t], t).toBeTruthy();
  });
  it("defines theme-independent tokens once in :root and not in classic", () => {
    for (const t of THEME_INDEPENDENT) {
      expect(studio[t], `studio ${t}`).toBeTruthy();
      // --cr-mono is deliberately reset to `initial` in classic (it was a
      // fallback-only var pre-redesign); everything else must not fork.
      if (t === "--cr-mono") continue;
      expect(classic[t], `classic must not fork ${t}`).toBeUndefined();
    }
    expect(studio["--cr-mono"]).toBe("var(--cr-font-mono)");
  });
  it("defines every legacy alias in both palettes", () => {
    for (const t of LEGACY) {
      expect(studio[t], `studio ${t}`).toBeTruthy();
      expect(classic[t], `classic ${t}`).toBeTruthy();
    }
  });
  it("keeps every classic legacy token pixel-identical to the pre-redesign palette", () => {
    // Exact :root palette as of commit 23855fc (pre-redesign), full fixture.
    const PRE_REDESIGN: Record<string, string> = {
      "--cr-bg": "#1a1a1f",
      "--cr-panel": "#24242c",
      "--cr-border": "#3a3a44",
      "--cr-text": "#e8e8ee",
      "--cr-muted": "#9a9aa8",
      "--cr-accent-dataset": "#2dd4bf",
      "--cr-accent-model": "#60a5fa",
      "--cr-accent-optimizer": "#fb923c",
      "--cr-accent-loss": "#f87171",
      "--cr-accent-trainer": "#a78bfa",
      "--cr-accent-checkpoint": "#64748b",
      "--cr-accent-where": "#fbbf24",
      "--cr-accent-tensor": "#22d3ee",
      "--cr-accent-observable": "#f472b6",
      "--cr-accent-hypothesis": "#4ade80",
      "--cr-accent-comment": "#94a3b8",
      "--cr-btn": "#2f2f38",
      "--cr-btn-primary": "#3b82f6",
    };
    for (const [t, v] of Object.entries(PRE_REDESIGN)) {
      expect(classic[t], t).toBe(v);
    }
    // --cr-danger is newly defined in classic; its only pre-redesign call
    // site used fallback #f87171, so the classic value must equal it.
    expect(classic["--cr-danger"]).toBe("#f87171");
    // Chart tokens replace the legacy SERIES_COLORS hex arrays (sweepVizPlot/
    // curveSeriesPlot); classic must pin the EXACT legacy palette in order.
    const LEGACY_SERIES = [
      "#c084fc", "#60a5fa", "#34d399", "#fbbf24",
      "#f472b6", "#a78bfa", "#2dd4bf", "#fb923c",
    ];
    LEGACY_SERIES.forEach((hex, i) => {
      expect(classic[`--cr-chart-${i + 1}`], `--cr-chart-${i + 1}`).toBe(hex);
    });
    // chart-9 exists for the viz-node variant arrays' 5th position (#fb7185).
    expect(classic["--cr-chart-9"]).toBe("#fb7185");
  });
  it("resets previously fallback-only vars to `initial` in classic (pixel fidelity)", () => {
    // These names appeared in index.css only inside var(--x, fallback) or
    // bare var(--x) before the redesign. Defining them in classic would
    // repaint those call sites; `initial` re-activates each site's own
    // historical fallback behavior.
    const FALLBACK_ONLY = [
      "--cr-fg", "--cr-text-muted", "--cr-library-accent", "--cr-mono",
      "--cr-surface-1", "--cr-surface-2", "--cr-surface-3",
    ];
    for (const t of FALLBACK_ONLY) {
      expect(classic[t], t).toBe("initial");
    }
  });
  it("studio text tokens meet contrast on studio surfaces (spec §9)", () => {
    for (const surface of ["--cr-surface-1", "--cr-surface-2", "--cr-surface-3"]) {
      for (const text of ["--cr-text-1", "--cr-text-2", "--cr-text-3"]) {
        expect(contrast(studio[text]!, studio[surface]!), `${text} on ${surface}`)
          .toBeGreaterThanOrEqual(4.5);
      }
    }
    // spec §9: text-4 (hints/placeholders/disabled) exempt at >= 2.5.
    expect(contrast(studio["--cr-text-4"]!, studio["--cr-surface-1"]!)).toBeGreaterThanOrEqual(2.5);
  });
  it("paper palette defines every semantic color token and passes contrast", () => {
    for (const t of SEMANTIC_COLOR) expect(paper[t], t).toBeTruthy();
    for (const surface of ["--cr-surface-1", "--cr-surface-2", "--cr-surface-3"]) {
      for (const text of ["--cr-text-1", "--cr-text-2", "--cr-text-3"]) {
        expect(contrast(paper[text]!, paper[surface]!), `${text} on ${surface}`)
          .toBeGreaterThanOrEqual(4.5);
      }
    }
    expect(contrast(paper["--cr-text-4"]!, paper["--cr-surface-1"]!)).toBeGreaterThanOrEqual(2.5);
    const accents = SEMANTIC_COLOR.filter(
      (t) => t.startsWith("--cr-accent-") && t !== "--cr-accent-hover",
    );
    for (const t of accents) {
      const band = compositeOver(paper[t]!, paper["--cr-surface-1"]!, 0.13);
      const title = compositeOver(paper[t]!, paper["--cr-text-1"]!, 0.45);
      expect(contrast(title, band), t).toBeGreaterThanOrEqual(4.5);
    }
  });
  it("studio header-band title tints meet 4.5:1 on their tinted bands (spec §4.2)", () => {
    const accents = SEMANTIC_COLOR.filter(
      (t) => t.startsWith("--cr-accent-") && t !== "--cr-accent-hover",
    );
    for (const t of accents) {
      const band = compositeOver(studio[t]!, studio["--cr-surface-1"]!, 0.13);
      const title = compositeOver(studio[t]!, studio["--cr-text-1"]!, 0.45);
      expect(contrast(title, band), t).toBeGreaterThanOrEqual(4.5);
    }
  });
});
