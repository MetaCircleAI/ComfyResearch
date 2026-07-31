import type { Node as RFNode } from "@xyflow/react";
import { pySlugForNode, type CodegenContext, type CodegenEdge } from "./codegenContext";
import { defaultInstanceTitleBase, readInstanceTitle } from "./nodeInstanceTitle";
import { nodeRegistryCodegen } from "./nodeRegistrySpec";
import { buildCatchAllNotebookCell } from "./auxNotebookNodeCodegen";

export type { CodegenContext, CodegenEdge } from "./codegenContext";
export { pySlugForNode } from "./codegenContext";

function isDatasetNodeType(t: string): boolean {
  return t.endsWith("_dataset") || t === "random_input_distribution" || t === "input_sampler";
}

function buildUnknownDatasetFallback(pySym: string, title: string, nodeType: string): string {
  return `# === ${title} (${nodeType}) ===
# UNKNOWN_DATASET_CODEGEN_FALLBACK — add a registry codegen entry backed by \`extraDatasetCodegen.ts\` and \`exportLoadersDatasets\`.
# No dedicated Code exporter for this dataset id yet — use Train on the canvas or add a generator in \`extraDatasetCodegen.ts\`.
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    x = torch.randn(512, 8, device=device)
    y = torch.randn(512, 1, device=device)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x[:128], y[:128]), batch_size=batch_size)
    return train_loader, test_loader
`;
}

/** Graph-only visualization nodes: do not auto-append a Code notebook cell when they are created. */
export function shouldOmitNotebookCell(nodeType: string): boolean {
  const t = String(nodeType ?? "");
  if (t === "training_visualization" || t === "observable_viz" || t === "visualize_kan") return true;
  if (t === "image_dataset_displayer") return true;
  if (t === "protein_structure_displayer" || t === "protein_structure_comparison_viz") return true;
  if (t === "table_viz" || t.startsWith("tensor_viz")) return true;
  if (t === "comment" || t === "url_node" || t === "graph_assist_failure_overlay") return true;
  if (t === "sweep_data_table" || t === "curve_annotator") return true;
  return false;
}

/**
 * Runnable Python cells for Code mode. Pass \`ctx\` with the full canvas \`nodes\` so slugs match trainer wiring
 * and duplicate node types get \`_2\`, \`_3\`, … suffixes.
 */
export function buildNodeDefinitionPython(node: RFNode, ctx?: CodegenContext): string {
  const nodeType = String(node.type ?? "unknown");
  const allNodes = ctx?.nodes?.length ? ctx.nodes : [node];
  const pySym = pySlugForNode(node.id, allNodes);
  const title = readInstanceTitle(node.data, defaultInstanceTitleBase(nodeType));
  const raw = (node.data ?? {}) as Record<string, unknown>;
  const registryCodegen = nodeRegistryCodegen({ pySym, title, nodeId: node.id, nodeType, data: raw, ctx });
  if (registryCodegen != null) return registryCodegen;

  if (isDatasetNodeType(nodeType)) {
    return buildUnknownDatasetFallback(pySym, title, nodeType);
  }
  return buildCatchAllNotebookCell(pySym, title, nodeType);
}

/** Short cell appended when Train is clicked (matches \`fn_<slug>_run\` / \`fn_<slug>_plot\` in the trainer definition). */
export function buildTrainerRunnerPython(trainerNodeId: string, allNodes: RFNode[]): string {
  const py = pySlugForNode(trainerNodeId, allNodes);
  return `# Auto-added when you clicked Train on this canvas
r = fn_${py}_run()
fn_${py}_plot(r)
`;
}

const PYTHON_FILE_HEADER =
  "# ComfyResearch — standalone Python export of a canvas. Matches Code tab cell order.\n" +
  "# Dependencies: pip install torch matplotlib numpy\n" +
  "# Headless plots: MPLBACKEND=Agg (set below in __main__ when auto-run is appended).\n\n";

function buildMainRunnerBlock(nodes: RFNode[]): string {
  const trainers = nodes.filter((n) => !n.parentId && n.type === "trainer");
  if (!trainers.length) return "";
  const lines: string[] = [];
  lines.push("\n\nif __name__ == \"__main__\":\n");
  lines.push("    import os\n");
  lines.push('    os.environ.setdefault("MPLBACKEND", "Agg")\n');
  lines.push("    import matplotlib\n");
  lines.push('    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))\n');
  for (const tr of trainers) {
    const runner = buildTrainerRunnerPython(tr.id, nodes);
    for (const ln of runner.split("\n")) {
      lines.push(ln === "" ? "\n" : `    ${ln}\n`);
    }
  }
  return lines.join("");
}

/**
 * Runnable script: same per-node generators as Code mode (seed order), then an ``__main__`` block
 * that runs each ``trainer`` (``fn_<slug>_run`` / ``fn_<slug>_plot``) with a headless-safe Matplotlib backend.
 */
export function buildCanvasStandalonePython(nodes: RFNode[], edges: CodegenEdge[]): string {
  const topLevel = nodes.filter((n) => String(n.type ?? "") !== "graph_assist_failure_overlay" && !n.parentId);
  const chunks: string[] = [];
  for (const n of topLevel) {
    const nt = String(n.type ?? "unknown");
    if (shouldOmitNotebookCell(nt)) continue;
    chunks.push(buildNodeDefinitionPython(n, { nodes, edges }));
  }
  return PYTHON_FILE_HEADER + chunks.join("\n\n# ----------\n\n") + buildMainRunnerBlock(nodes);
}
