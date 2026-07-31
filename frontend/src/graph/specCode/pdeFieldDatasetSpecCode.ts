import type { PdeFieldDatasetKind, PdeFieldDatasetNodeData } from "../../components/nodes/pdeFieldDatasetDefaults";
import { defaultPdeFieldDatasetData } from "../../components/nodes/pdeFieldDatasetDefaults";

function firstScalar(v: unknown, fallback: unknown): unknown {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return v.length ? v[0] : fallback;
  return v;
}

export const DEFAULT_PDE_FIELD_DATASET_SPEC_NAME = "PdeFieldDataset";

export function defaultParamOrderForPdeKind(kind: PdeFieldDatasetKind): string[] {
  const common = [
    "contextFrames",
    "channels",
    "gridSize",
    "trainSize",
    "testSize",
    "warmupSteps",
    "dt",
    "icScale",
    "initSeed",
  ] as const;
  if (kind === "diffusion_pde_dataset") {
    return [...common.slice(0, 6), "diffusionCoeff", ...common.slice(6)];
  }
  if (kind === "reaction_diffusion_dataset") {
    return [...common.slice(0, 6), "diffusionCoeff", "reactionRate", ...common.slice(6)];
  }
  return [...common.slice(0, 6), "velocityX", "velocityY", ...common.slice(6)];
}

export function generatePdeFieldDatasetSpecCode(
  kind: PdeFieldDatasetKind,
  d: PdeFieldDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const defs = defaultPdeFieldDatasetData(kind);
  const merged = { ...defs, ...d };
  const nm = specName.trim() || DEFAULT_PDE_FIELD_DATASET_SPEC_NAME;
  const keys = (order.length ? order : defaultParamOrderForPdeKind(kind)).filter(Boolean);
  const lines: string[] = [
    `import numpy as np`,
    `from comfy_research.schemas.graph import NodeKind`,
    `from comfy_research.engine.pde_field_dataset_runtime import build_pde_field_arrays`,
    ``,
    `def sample_${nm}_arrays(train_size: int, test_size: int, init_seed: int = 0):`,
    `    data = {`,
  ];
  for (const k of keys) {
    const v = (merged as Record<string, unknown>)[k];
    lines.push(`        "${k}": ${JSON.stringify(firstScalar(v, (defs as Record<string, unknown>)[k]))},`);
  }
  lines.push(`        "samplingMode": ${JSON.stringify(String(firstScalar(merged.samplingMode, "fixed")))},`);
  lines.push(`    }`);
  lines.push(`    rng = np.random.default_rng(int(init_seed))`);
  const nk =
    kind === "diffusion_pde_dataset"
      ? "NodeKind.diffusion_pde_dataset"
      : kind === "reaction_diffusion_dataset"
        ? "NodeKind.reaction_diffusion_dataset"
        : "NodeKind.advection_dataset";
  lines.push(`    x_train, y_train, x_test, y_test = build_pde_field_arrays(${nk}, rng, data, data, train_size, test_size)`);
  lines.push(`    return x_train, y_train, x_test, y_test`);
  return lines.join("\n");
}
