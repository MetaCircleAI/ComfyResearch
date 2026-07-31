import type { Node as RFNode } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { buildNodeDefinitionPython, shouldOmitNotebookCell } from "../nodeDefinitionCode";
import { NODE_SPEC_REGISTRY, nodeRegistryDefaults } from "../nodeRegistrySpec";

function makeDefaultNode(nodeType: string): RFNode {
  return {
    id: `codegen_entry_${nodeType}`,
    type: nodeType,
    data: nodeRegistryDefaults(nodeType) ?? {},
    position: { x: 0, y: 0 },
  } as RFNode;
}

describe("registry codegen entries", () => {
  it("runs every default non-visual node headlessly", () => {
    const problems: string[] = [];
    const nodeTypes = Object.keys(NODE_SPEC_REGISTRY).sort((a, b) => a.localeCompare(b));

    for (const nodeType of nodeTypes) {
      if (shouldOmitNotebookCell(nodeType)) continue;

      const node = makeDefaultNode(nodeType);
      const ctx = { nodes: [node], edges: [] };
      let code = "";
      try {
        code = buildNodeDefinitionPython(node, ctx);
      } catch (error) {
        problems.push(`${nodeType}: threw ${error instanceof Error ? error.message : String(error)}`);
        continue;
      }

      if (!code.trim()) problems.push(`${nodeType}: empty generated cell`);
      if (code.includes("undefined")) problems.push(`${nodeType}: emitted literal undefined`);
      if (code.includes("[object Object]")) problems.push(`${nodeType}: emitted literal [object Object]`);
      if (code.includes("UNKNOWN_DATASET_CODEGEN_FALLBACK")) {
        problems.push(`${nodeType}: hit unknown dataset fallback`);
      }
      if (/raise RuntimeError\("Replace this stub/.test(code)) {
        problems.push(`${nodeType}: emitted removed generic RuntimeError stub`);
      }
      if (/return nn\.Identity\(\)/.test(code)) {
        problems.push(`${nodeType}: emitted removed nn.Identity stub`);
      }
    }

    expect(problems).toEqual([]);
  });
});
