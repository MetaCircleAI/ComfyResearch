// @vitest-environment jsdom
/**
 * Studio-theme number steppers on ComfyIntField: present under studio,
 * absent under classic (classic keeps the exact legacy DOM), stepping acts
 * on the last COMMITTED value and never commits in-progress text.
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ComfyIntField } from "../../components/nodes/comfyNumberFields";
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

function mount(theme: "studio" | "classic", onCommit: (n: number) => void): { host: HTMLElement; root: Root } {
  window.localStorage.setItem("comfyresearch.theme", theme);
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      createElement(
        ThemeProvider,
        null,
        createElement(ComfyIntField, {
          label: "steps",
          value: 2000,
          onCommit,
          min: 0,
          ariaLabel: "steps",
        }),
      ),
    );
  });
  return { host, root };
}

describe("ComfyIntField studio steppers", () => {
  it("renders hover steppers under studio and steps the committed value", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("studio", onCommit);
    const inc = host.querySelector<HTMLButtonElement>('[aria-label="steps increase"]');
    const dec = host.querySelector<HTMLButtonElement>('[aria-label="steps decrease"]');
    expect(inc).toBeTruthy();
    expect(dec).toBeTruthy();
    expect(inc!.type).toBe("button");
    act(() => inc!.click());
    expect(onCommit).toHaveBeenCalledWith(2001);
    act(() => dec!.click());
    expect(onCommit).toHaveBeenCalledWith(1999);
    act(() => root.unmount());
    host.remove();
  });

  it("steps from the committed value, discarding uncommitted typed text", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("studio", onCommit);
    const input = host.querySelector<HTMLInputElement>('input[aria-label="steps"]')!;
    // Type without blurring (no commit yet).
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
      setter.call(input, "77");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const inc = host.querySelector<HTMLButtonElement>('[aria-label="steps increase"]')!;
    act(() => inc.click());
    // Steps from committed 2000, not typed 77; text resyncs to the new value.
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(2001);
    expect(input.value).toBe("2001");
    act(() => root.unmount());
    host.remove();
  });

  it("respects min clamping without emitting a no-op commit", () => {
    const onCommit = vi.fn();
    window.localStorage.setItem("comfyresearch.theme", "studio");
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() => {
      root.render(
        createElement(
          ThemeProvider,
          null,
          createElement(ComfyIntField, { label: "n", value: 0, onCommit, min: 0, ariaLabel: "n" }),
        ),
      );
    });
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="n decrease"]')!.click());
    expect(onCommit).not.toHaveBeenCalled();
    act(() => root.unmount());
    host.remove();
  });

  it("does not commit in-progress text when focus moves to a stepper (Tab path)", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("studio", onCommit);
    const input = host.querySelector<HTMLInputElement>('input[aria-label="steps"]')!;
    const inc = host.querySelector<HTMLButtonElement>('[aria-label="steps increase"]')!;
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
      setter.call(input, "77");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    // Simulate Tab: focusout with relatedTarget = the stepper button.
    act(() => {
      input.dispatchEvent(new FocusEvent("focusout", { bubbles: true, relatedTarget: inc }));
    });
    expect(onCommit).not.toHaveBeenCalled(); // 77 was NOT committed
    expect(input.value).toBe("2000"); // reverted to committed value
    act(() => inc.click());
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(2001);
    act(() => root.unmount());
    host.remove();
  });

  it("bare render (no provider) under classic html attr keeps legacy DOM", () => {
    document.documentElement.dataset.crTheme = "classic";
    const onCommit = vi.fn();
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() => {
      root.render(
        createElement(ComfyIntField, { label: "n", value: 5, onCommit, ariaLabel: "n" }),
      );
    });
    expect(host.querySelector(".cr-num-step")).toBeNull();
    expect(host.querySelector(".cr-num-wrap")).toBeNull();
    act(() => root.unmount());
    host.remove();
  });

  it("renders the legacy DOM (no steppers) under classic", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("classic", onCommit);
    expect(host.querySelector(".cr-num-step")).toBeNull();
    expect(host.querySelector(".cr-num-wrap")).toBeNull();
    expect(host.querySelector('input[aria-label="steps"]')).toBeTruthy();
    act(() => root.unmount());
    host.remove();
  });
});
