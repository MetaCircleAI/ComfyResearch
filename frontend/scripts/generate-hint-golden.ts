/** Resolved hints from nodeRegistryHint, snapshotted by node type. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { NODE_SPEC_REGISTRY, nodeRegistryHint } from "../src/graph/nodeRegistrySpec";

export const HINT_GOLDEN_PATH = resolve(process.cwd(), "src/graph/__tests__/__snapshots__/hintGolden.json");

export function computeHintGolden(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const t of Object.keys(NODE_SPEC_REGISTRY).sort()) {
    const resolved = nodeRegistryHint(t);
    if (resolved != null) out[t] = resolved;
  }
  return out;
}

const isMain = process.argv[1]?.endsWith(".mjs") || process.argv[1]?.endsWith("generate-hint-golden.ts");
if (isMain) {
  writeFileSync(HINT_GOLDEN_PATH, JSON.stringify(computeHintGolden(), null, 1) + "\n");
  console.log("wrote hint golden:", Object.keys(computeHintGolden()).length, "hints");
}
