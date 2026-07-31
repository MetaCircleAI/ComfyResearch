// @vitest-environment jsdom
/**
 * v1 trim seam: the primary rail ships exactly these sections. PVI, LMech and
 * Test (quiz) were removed for the first release (archive/pre-v1-trim keeps
 * the pre-trim tree); Observables deliberately stays. Re-adding or re-removing
 * a rail entry must be an explicit decision that updates this list.
 *
 * PR-1 of the UI redesign re-adds a single footer button (theme settings
 * popover) — an intentional exception recorded here.
 */
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it } from "vitest";

import { LeftNavRail } from "../../components/LeftNavRail";
import { ThemeProvider } from "../../themeContext";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/**
 * Node >= 22 injects an experimental, file-less `localStorage` global that
 * shadows jsdom's working implementation inside vitest. Install a
 * deterministic in-memory Storage instead (same stub as theme.test.ts).
 */
class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? this.map.get(key)! : null;
  }
  key(index: number): string | null {
    return [...this.map.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, String(value));
  }
}

beforeEach(() => {
  Object.defineProperty(window, "localStorage", {
    value: new MemoryStorage(),
    configurable: true,
  });
  delete document.documentElement.dataset.crTheme;
});

const V1_RAIL_LABELS = ["Nodes", "Observables", "Templates"];

function renderRail(host: HTMLElement) {
  const root = createRoot(host);
  act(() => {
    root.render(
      createElement(
        ThemeProvider,
        null,
        createElement(LeftNavRail, {
          activeSection: "nodes",
          onSelectSection: () => undefined,
        }),
      ),
    );
  });
  return root;
}

describe("LeftNavRail v1 composition", () => {
  it("renders exactly the v1 primary sections, in order", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = renderRail(host);
    const labels = [...host.querySelectorAll(".cr-rail__main .cr-rail__label")].map(
      (el) => el.textContent,
    );
    expect(labels).toEqual(V1_RAIL_LABELS);
    const footerButtons = [...host.querySelectorAll(".cr-rail__footer > button")];
    expect(footerButtons.map((b) => b.getAttribute("aria-label"))).toEqual(["Theme settings"]);
    act(() => root.unmount());
    host.remove();
  });

  it("theme popover switches and persists the theme", () => {
    document.documentElement.dataset.crTheme = "studio";
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = renderRail(host);
    const toggle = host.querySelector<HTMLButtonElement>('[aria-label="Theme settings"]')!;
    act(() => toggle.click());
    const options = [...host.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]')];
    expect(options.map((o) => o.textContent)).toEqual(["Studio", "Paper", "Classic"]);
    act(() => options[2]!.click());
    expect(document.documentElement.dataset.crTheme).toBe("classic");
    expect(window.localStorage.getItem("comfyresearch.theme")).toBe("classic");
    act(() => root.unmount());
    host.remove();
  });
});
