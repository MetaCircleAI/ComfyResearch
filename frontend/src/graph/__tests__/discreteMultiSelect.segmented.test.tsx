// @vitest-environment jsdom
/**
 * Segmented presentation of DiscreteMultiSelect: studio-only DOM (radiogroup
 * of role=radio segments), classic keeps the dropdown trigger exactly.
 */
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscreteMultiSelect } from "../../components/nodes/DiscreteMultiSelect";
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

const OPTIONS = [
  { id: "cpu", label: "CPU" },
  { id: "mps", label: "MPS" },
  { id: "cuda", label: "CUDA (local)" },
] as const;
type Id = (typeof OPTIONS)[number]["id"];

function mount(theme: "studio" | "classic", onCommit: (v: unknown) => void) {
  window.localStorage.setItem("comfyresearch.theme", theme);
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      createElement(
        ThemeProvider,
        null,
        createElement(DiscreteMultiSelect<Id>, {
          label: "device",
          options: [...OPTIONS],
          value: "cpu",
          singleSelect: true,
          presentation: "segmented",
          segmentLabels: { cuda: "CUDA" },
          ariaLabel: "Device",
          onCommit,
        }),
      ),
    );
  });
  return { host, root };
}

describe("DiscreteMultiSelect segmented", () => {
  it("renders a radiogroup under studio and commits single ids", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("studio", onCommit);
    const group = host.querySelector('[role="radiogroup"]');
    expect(group).toBeTruthy();
    const segs = [...host.querySelectorAll<HTMLButtonElement>('[role="radio"]')];
    expect(segs.map((s) => s.textContent)).toEqual(["CPU", "MPS", "CUDA"]);
    expect(segs.map((s) => s.getAttribute("aria-checked"))).toEqual(["true", "false", "false"]);
    expect(segs[2]!.title).toBe("CUDA (local)"); // full label preserved
    act(() => segs[1]!.click());
    expect(onCommit).toHaveBeenCalledWith("mps");
    act(() => segs[0]!.click()); // clicking the active segment is a no-op
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(host.querySelector(".cr-discrete-multi-dd__btn")).toBeNull();
    act(() => root.unmount());
    host.remove();
  });

  it("keeps the dropdown trigger DOM under classic", () => {
    const onCommit = vi.fn();
    const { host, root } = mount("classic", onCommit);
    expect(host.querySelector('[role="radiogroup"]')).toBeNull();
    expect(host.querySelector(".cr-discrete-multi-dd__btn")).toBeTruthy();
    act(() => root.unmount());
    host.remove();
  });
});
