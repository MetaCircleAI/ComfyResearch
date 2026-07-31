// @vitest-environment jsdom

import { ReactFlowProvider, type NodeProps } from "@xyflow/react";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { SchemaNode, renderFieldControl, nodeTypeClass } from "../../components/nodes/SchemaNode";
import { NODE_SPEC_REGISTRY, nodeRegistryDefaults } from "../nodeRegistrySpec";

(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

function propsForNodeType(nodeType: string): NodeProps {
  return {
    id: `${nodeType}-schema`,
    type: nodeType,
    data: nodeRegistryDefaults(nodeType) ?? {},
    selected: false,
    dragging: false,
    zIndex: 0,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
  } as NodeProps;
}

function mount(nodeType: string): { container: HTMLDivElement; unmount: () => void } {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <ReactFlowProvider>{createElement(SchemaNode, propsForNodeType(nodeType))}</ReactFlowProvider>,
    );
  });
  return { container, unmount: () => act(() => root.unmount()) };
}

describe("SchemaNode", () => {
  it("derives a hyphenated node class from the type", () => {
    expect(nodeTypeClass("adam_optimizer")).toBe("cr-node--adam-optimizer");
    expect(nodeTypeClass("shampoo_optimizer")).toBe("cr-node--shampoo-optimizer");
  });

  const OPTIMIZER_TYPES = [
    "adam_optimizer",
    "adamw_optimizer",
    "sgd_optimizer",
    "signsgd_optimizer",
    "muon_optimizer",
    "shampoo_optimizer",
    "soap_optimizer",
  ] as const;

  it.each(OPTIMIZER_TYPES)("renders stable field containers for %s in spec order", (nodeType) => {
    const { container, unmount } = mount(nodeType);
    const keys = [...container.querySelectorAll("[data-schema-field-key]")].map((el) =>
      el.getAttribute("data-schema-field-key"),
    );
    expect(keys).toEqual(NODE_SPEC_REGISTRY[nodeType].fields?.map((f) => f.key));
    unmount();
  });

  it("preserves the exact per-field aria-labels", () => {
    const { container, unmount } = mount("adam_optimizer");
    const labels = [...container.querySelectorAll("[aria-label]")].map((el) => el.getAttribute("aria-label"));
    for (const expected of ["Learning rate", "Adam beta1", "Adam beta2", "Adam epsilon", "Adam weight decay"]) {
      expect(labels).toContain(expected);
    }
    unmount();
  });

  it("throws loud on an unsupported field kind", () => {
    expect(() =>
      renderFieldControl({ kind: "int", key: "x", label: "x", defaultValue: 1 }, {}, () => undefined),
    ).toThrow(/does not support field kind: int/);
  });
});
