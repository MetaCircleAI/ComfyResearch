/**
 * Runnable (or explicitly documented) Code cells for nodes that are not datasets / models / optimizers / losses.
 */
export function buildGraphAuxCell(pySym: string, title: string, nodeType: string, note: string): string {
  return `# === ${title} (${nodeType}) ===
# ${note}


def fn_${pySym}_describe():
    return ${JSON.stringify(nodeType)}
`;
}

/** Tensor graph, observables, checkpoints, and other canvas-only wiring: no ``RuntimeError`` — see server sources. */
export function buildCatchAllNotebookCell(pySym: string, title: string, nodeType: string): string {
  return buildGraphAuxCell(
    pySym,
    title,
    nodeType,
    "Executed as part of the wired graph (tensor pipeline, observables, or /api/train). See comfy_research/engine (trainer_run, activation_collect, tensor graph builders) for exact formulas.",
  );
}
