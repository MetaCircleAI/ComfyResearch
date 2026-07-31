// @vitest-environment jsdom

import { ReactFlowProvider, useStoreApi } from "@xyflow/react";
import { act, createElement, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { GraphToolbar } from "../../components/ResearchCanvas";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  document.body.replaceChildren();
});

function renderToolbar() {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  const noop = () => undefined;
  let store: ReturnType<typeof useStoreApi> | null = null;

  function StoreProbe() {
    const storeApi = useStoreApi();
    useEffect(() => {
      store = storeApi;
    }, [storeApi]);
    return null;
  }

  act(() => {
    root.render(
      <ReactFlowProvider>
        <StoreProbe />
        {createElement(GraphToolbar, {
          onSaveToServer: async () => undefined,
          onLoadFromServer: async () => undefined,
          onSaveGraphToSourceFile: noop,
          canSaveGraphToSourceFile: false,
          onSaveGraphToFileTier: noop,
          onSaveGraphAsLibrary: noop,
          onExportCanvasPng: noop,
          onExportCanvasPdf: noop,
          onOpenGraphCompare: noop,
          onGraphFileLoaded: noop,
          onGraphFileError: noop,
          onAutoLayoutCanvas: noop,
          onAutoConnectCanvas: noop,
          onClearCanvas: noop,
          loading: false,
          error: null,
          notice: null,
        })}
      </ReactFlowProvider>,
    );
  });
  return { host, root, getStore: () => store };
}

describe("GraphToolbar canvas controls", () => {
  it("places fit-view in the canvas group and invokes the real fit action", () => {
    const { host, root, getStore } = renderToolbar();
    const controls = host.querySelector("[aria-label='Canvas controls']");
    const labels = [...(controls?.querySelectorAll("button") ?? [])].map((button) =>
      button.getAttribute("aria-label"),
    );

    expect(labels).toEqual([
      "Auto layout graph",
      "Auto-connect trainer wiring",
      "Clear canvas",
      "Fit graph to view",
      "Zoom out",
      "Zoom in",
    ]);

    const fitButton = controls?.querySelector<HTMLButtonElement>(
      "button[aria-label='Fit graph to view']",
    );
    act(() => fitButton?.click());
    expect(getStore()?.getState().fitViewQueued).toBe(true);
    expect(getStore()?.getState().fitViewOptions).toMatchObject({
      padding: 0.12,
      duration: 320,
    });

    act(() => root.unmount());
  });
});
