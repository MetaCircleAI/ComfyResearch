// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  initTheme,
  normalizeTheme,
  readStoredTheme,
} from "../../theme";
import { ThemeProvider, useTheme } from "../../themeContext";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/**
 * Node >= 22 injects an experimental, file-less `localStorage` global that
 * shadows jsdom's working implementation inside vitest (its methods throw or
 * are missing). Install a deterministic in-memory Storage instead.
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

describe("theme module", () => {
  it("defaults to studio with empty storage", () => {
    expect(readStoredTheme()).toBe("studio");
    expect(DEFAULT_THEME).toBe("studio");
  });
  it("normalizes unknown stored values to the default", () => {
    expect(normalizeTheme("classic")).toBe("classic");
    expect(normalizeTheme("paper")).toBe("paper");
    expect(normalizeTheme("neon")).toBe("studio");
    expect(normalizeTheme(null)).toBe("studio");
  });
  it("initTheme applies the stored theme to <html>", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "classic");
    expect(initTheme()).toBe("classic");
    expect(document.documentElement.dataset.crTheme).toBe("classic");
  });
});

describe("ThemeProvider / useTheme", () => {
  function Probe() {
    const { theme, setTheme } = useTheme();
    return createElement(
      "button",
      { "data-theme": theme, onClick: () => setTheme(theme === "studio" ? "classic" : "studio") },
      theme,
    );
  }
  it("applies the stored theme to <html> on mount without initTheme()", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "classic");
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() => {
      root.render(createElement(ThemeProvider, null, createElement(Probe)));
    });
    expect(document.documentElement.dataset.crTheme).toBe("classic");
    act(() => root.unmount());
    host.remove();
  });
  it("provides the active theme and switches + persists on setTheme", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() => {
      root.render(createElement(ThemeProvider, null, createElement(Probe)));
    });
    const btn = host.querySelector("button")!;
    expect(btn.textContent).toBe("studio");
    act(() => btn.click());
    expect(btn.textContent).toBe("classic");
    expect(document.documentElement.dataset.crTheme).toBe("classic");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("classic");
    act(() => btn.click());
    expect(document.documentElement.dataset.crTheme).toBe("studio");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("studio");
    act(() => root.unmount());
    host.remove();
  });
});
