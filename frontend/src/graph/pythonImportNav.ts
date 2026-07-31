/**
 * Discover ``from comfy_research.... import ...`` lines in notebook / spec Python for "go to source" links.
 */

export type ComfyResearchImportNav = {
  /** Full dotted module, e.g. ``comfy_research.engine.transformer_encoder_custom``. */
  module: string;
  /** Shorter label for chips, e.g. ``engine.transformer_encoder_custom``. */
  shortLabel: string;
};

/**
 * First occurrence per module, in source order (top of file first).
 */
export function listComfyResearchImportModules(pythonSource: string): ComfyResearchImportNav[] {
  const lines = pythonSource.split(/\r?\n/);
  const seen = new Set<string>();
  const out: ComfyResearchImportNav[] = [];
  for (const line of lines) {
    const m = line.match(/^\s*from\s+([\w.]+)\s+import\s+/);
    if (!m) continue;
    const mod = m[1]!.trim();
    if (!mod.startsWith("comfy_research.")) continue;
    if (seen.has(mod)) continue;
    seen.add(mod);
    const shortLabel = mod.startsWith("comfy_research.") ? mod.slice("comfy_research.".length) : mod;
    out.push({ module: mod, shortLabel });
  }
  return out;
}
