/**
 * Bundled with esbuild (CSS → empty) and executed by node to verify Code-tab codegen for every research node type.
 */
import type { Node as RFNode } from "@xyflow/react";
import { researchNodeTypes } from "../src/components/nodeTypes";
import { buildNodeDefinitionPython, shouldOmitNotebookCell } from "../src/graph/nodeDefinitionCode";

function makeNode(type: string, id: string): RFNode {
  return {
    id,
    type,
    data: {},
    position: { x: 0, y: 0 },
  } as RFNode;
}

const keys = Object.keys(researchNodeTypes).sort((a, b) => a.localeCompare(b));
const problems: string[] = [];

for (const nodeType of keys) {
  if (shouldOmitNotebookCell(nodeType)) continue;

  const id = `verify_${nodeType}_a`;
  const n = makeNode(nodeType, id);
  const ctx = { nodes: [n], edges: [] as { source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null }[] };

  let src: string;
  try {
    src = buildNodeDefinitionPython(n, ctx);
  } catch (e) {
    problems.push(`${nodeType}: buildNodeDefinitionPython threw: ${e}`);
    continue;
  }

  if (!src.trim()) {
    problems.push(`${nodeType}: empty generated cell`);
    continue;
  }

  if (/raise RuntimeError\("Replace this stub/.test(src)) {
    problems.push(`${nodeType}: still uses removed generic RuntimeError stub`);
  }
  if (/return nn\.Identity\(\)/.test(src)) {
    problems.push(`${nodeType}: still uses nn.Identity layer stub`);
  }

  if (src.includes("UNKNOWN_DATASET_CODEGEN_FALLBACK")) {
    problems.push(`${nodeType}: hits unknown-dataset fallback (missing from buildRoutedDatasetTorch / exportLoadersDatasets?)`);
  }
}

if (problems.length) {
  console.error("Research node codegen verification FAILED:\n" + problems.map((p) => "  - " + p).join("\n"));
  process.exit(1);
}

console.log(`OK: ${keys.length} research node types; ${keys.filter((k) => !shouldOmitNotebookCell(k)).length} require Code cells — all pass.`);
