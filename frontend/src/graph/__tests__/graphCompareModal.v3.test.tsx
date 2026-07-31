// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import {
  GraphCompareModal,
  type GraphCompareTarget,
} from "../../components/GraphCompareModal";
import type { GraphDocument } from "../../types/graph";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const emptyDocument: GraphDocument = {
  version: 1,
  nodes: [],
  edges: [],
  viewport: null,
};

afterEach(() => {
  document.body.replaceChildren();
});

function renderCompareModal(targets: GraphCompareTarget[]) {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      createElement(GraphCompareModal, {
        open: true,
        onClose: () => undefined,
        sourceLabel: "Baseline / Main",
        sourceDocument: emptyDocument,
        targets,
      }),
    );
  });
  return root;
}

describe("GraphCompareModal workspace v3 copy", () => {
  it("describes project-only comparison when no target exists", () => {
    const root = renderCompareModal([]);

    expect(document.querySelector(".cr-graph-compare__intro")?.textContent).not.toContain(
      "Duplicated " + "canvases",
    );
    expect(document.querySelector(".cr-modal__hint")?.textContent).toBe(
      "Add another project to compare against.",
    );

    act(() => root.unmount());
  });

  it("asks the user to choose a project", () => {
    const root = renderCompareModal([
      {
        key: "comparison",
        projectTitle: "Comparison",
        canvasTitle: "Main",
        document: emptyDocument,
      },
    ]);

    expect(document.querySelector("option")?.textContent).toBe("Choose a project…");

    act(() => root.unmount());
  });
});
