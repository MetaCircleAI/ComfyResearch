// @vitest-environment jsdom

import { ReactFlowProvider, NodeProps } from "@xyflow/react";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { ResearchGraphProvider } from "../../context/ResearchGraphContext";
import { NODE_REGISTRY } from "../nodeRegistry";
import { nodeRegistryDefaults } from "../nodeRegistrySpec";

const noop = () => undefined;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function propsForNodeType(nodeType: string, data: Record<string, unknown> = {}): NodeProps {
  return {
    id: `${nodeType}-smoke`,
    type: nodeType,
    data: { ...(nodeRegistryDefaults(nodeType) ?? {}), ...data },
    selected: false,
    dragging: false,
    zIndex: 0,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
  } as NodeProps;
}

describe("node component smoke", () => {
  it("mounts every registry component with default data inside React Flow context", () => {
    const failures: string[] = [];

    for (const [nodeType, spec] of Object.entries(NODE_REGISTRY)) {
      const container = document.createElement("div");
      document.body.appendChild(container);
      const root = createRoot(container);

      try {
        act(() => {
          root.render(
            <ReactFlowProvider>
              <ResearchGraphProvider value={{ addNode: noop }}>
                {createElement(spec.component, propsForNodeType(nodeType))}
              </ResearchGraphProvider>
            </ReactFlowProvider>,
          );
        });
      } catch (error) {
        failures.push(`${nodeType}: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        act(() => {
          root.unmount();
        });
        container.remove();
      }
    }

    expect(failures).toEqual([]);
  });

  it("keeps IDNNs Trainer controls and Advanced summary coherent", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    act(() => {
      root.render(
        <ReactFlowProvider>
          <ResearchGraphProvider value={{ addNode: noop }}>
            {createElement(
              NODE_REGISTRY.trainer.component,
              propsForNodeType("trainer", { logSchedule: "idnns_logspace", logSamples: 123 }),
            )}
          </ResearchGraphProvider>
        </ReactFlowProvider>,
      );
    });

    expect(container.querySelector('[aria-label="Log frequency (steps)"]')).toBeNull();
    expect(container.querySelector("summary.cr-trainer-advanced__summary")?.textContent?.trim()).toBe("Advanced");
    expect(container.querySelector(".cr-trainer-advanced__status")).toBeNull();
    expect(
      container.querySelector('.cr-comfy-widget__label[title*="selected minibatch policy"]'),
    ).not.toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
  });
});
