import type { VisionDatasetKind, VisionDatasetNodeData } from "../../components/nodes/visionDatasetDefaults";

function esc(s: string): string {
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

export const DEFAULT_VISION_DATASET_SPEC_NAME = "visionDatasetSpec";

export function generateVisionDatasetSpecCode(
  kind: VisionDatasetKind,
  d: VisionDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const lines: string[] = [];
  lines.push(`# === ${esc(specName)} (${kind}) ===`);
  lines.push(`from comfy_research.engine.vision_datasets_runtime import build_vision_numpy_arrays`);
  lines.push(`import json`);
  lines.push(`import numpy as np`);
  lines.push(`from comfy_research.schemas.graph import NodeKind`);
  lines.push("");
  lines.push(`def ${esc(specName)}():`);
  lines.push(`    rng = np.random.default_rng(int(${JSON.stringify(scalar(d.initSeed ?? d.seed ?? 0))}))`);
  lines.push(`    data = json.loads(r"""${esc(jsonDataBlock(d, order))}""")`);
  lines.push(
    `    x_tr, y_tr, x_te, y_te = build_vision_numpy_arrays(NodeKind("${kind}"), data, int(data["trainSize"]), int(data["testSize"]), rng)`,
  );
  lines.push(`    if bool(data.get("flattenOutput", False)):`);
  lines.push(`        def _fl(x):`);
  lines.push(`            if x is None:`);
  lines.push(`                return None`);
  lines.push(`            return x.reshape(int(x.shape[0]), -1).astype(np.float32)`);
  lines.push(`        x_tr, x_te = _fl(x_tr), _fl(x_te)`);
  lines.push(`    return {"x_train": x_tr, "y_train": y_tr, "x_test": x_te, "y_test": y_te}`);
  return lines.join("\n");
}

function scalar(v: unknown): number {
  if (Array.isArray(v)) return typeof v[0] === "number" ? (v[0] as number) : 0;
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function jsonDataBlock(d: VisionDatasetNodeData, order: string[]): string {
  const o: Record<string, unknown> = {};
  for (const k of order) {
    const v = (d as Record<string, unknown>)[k];
    if (v !== undefined) o[k] = v;
  }
  o.trainSize = scalar(d.trainSize);
  o.testSize = scalar(d.testSize);
  o.initSeed = scalar(d.initSeed ?? d.seed);
  if (d.seed !== undefined) o.seed = scalar(d.seed);
  return JSON.stringify(o);
}
