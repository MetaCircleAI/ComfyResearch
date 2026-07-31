// @vitest-environment jsdom
/** ParamSection invariants: classic renders children with NO wrapper DOM;
 * studio wraps, toggles, and forceOpen pins the body visible. */
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it } from "vitest";
import { ParamSection } from "../../components/nodes/ParamSection";
import { ThemeProvider } from "../../themeContext";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(k: string): string | null {
    return this.map.get(k) ?? null;
  }
  key(i: number): string | null {
    return [...this.map.keys()][i] ?? null;
  }
  removeItem(k: string): void {
    this.map.delete(k);
  }
  setItem(k: string, v: string): void {
    this.map.set(k, String(v));
  }
}

beforeEach(() => {
  Object.defineProperty(window, "localStorage", { value: new MemoryStorage(), configurable: true });
  delete document.documentElement.dataset.crTheme;
});

function mount(theme: "studio" | "classic", props: { defaultOpen?: boolean; forceOpen?: boolean }) {
  window.localStorage.setItem("comfyresearch.theme", theme);
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      createElement(
        ThemeProvider,
        null,
        createElement(
          ParamSection,
          { title: "Advanced settings", ...props },
          createElement("span", { className: "probe-child" }, "grad clip"),
        ),
      ),
    );
  });
  return { host, root };
}

describe("ParamSection", () => {
  it("classic: renders children directly, no wrapper DOM", () => {
    const { host, root } = mount("classic", {});
    expect(host.querySelector(".cr-param-section")).toBeNull();
    expect(host.querySelector(".probe-child")?.textContent).toBe("grad clip");
    act(() => root.unmount());
    host.remove();
  });

  it("studio: collapsed by default when defaultOpen=false, toggles open", () => {
    const { host, root } = mount("studio", { defaultOpen: false });
    expect(host.querySelector(".probe-child")).toBeNull();
    const head = host.querySelector<HTMLButtonElement>(".cr-param-section__head")!;
    expect(head.getAttribute("aria-expanded")).toBe("false");
    act(() => head.click());
    expect(host.querySelector(".probe-child")).toBeTruthy();
    expect(head.getAttribute("aria-expanded")).toBe("true");
    act(() => root.unmount());
    host.remove();
  });

  it("studio: forceOpen keeps the body visible and untogglable-closed", () => {
    const { host, root } = mount("studio", { defaultOpen: false, forceOpen: true });
    expect(host.querySelector(".probe-child")).toBeTruthy();
    const head = host.querySelector<HTMLButtonElement>(".cr-param-section__head")!;
    act(() => head.click());
    expect(host.querySelector(".probe-child"), "forceOpen section must not collapse").toBeTruthy();
    act(() => root.unmount());
    host.remove();
  });
});
