import type { MppSpatiotemporalModelNodeData } from "../../components/nodes/mppSpatiotemporalModelDefaults";
import { defaultMppSpatiotemporalModelData } from "../../components/nodes/mppSpatiotemporalModelDefaults";

function firstScalar(v: unknown, fallback: unknown): unknown {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return v.length ? v[0] : fallback;
  return v;
}

const KEYS = new Set([
  "contextFrames",
  "channels",
  "gridSize",
  "inputDim",
  "outputDim",
  "patchSize",
  "embedDim",
  "depth",
  "numHeads",
  "ffRatio",
  "dropout",
  "seed",
]);

export const DEFAULT_MPP_SPATIOTEMPORAL_SPEC_NAME = "MppSpatiotemporalModel";

export const DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER: (keyof MppSpatiotemporalModelNodeData)[] = [
  "contextFrames",
  "channels",
  "gridSize",
  "inputDim",
  "outputDim",
  "patchSize",
  "embedDim",
  "depth",
  "numHeads",
  "ffRatio",
  "dropout",
  "seed",
];

export function generateMppSpatiotemporalModelSpecCode(
  d: MppSpatiotemporalModelNodeData,
  order: string[],
  specName: string,
): string {
  const merged = { ...defaultMppSpatiotemporalModelData(), ...d };
  const nm = specName.trim() || DEFAULT_MPP_SPATIOTEMPORAL_SPEC_NAME;
  const keys = (order.length ? order : DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER).filter((k) => KEYS.has(k));
  const lines: string[] = [
    `import torch`,
    `from comfy_research.engine.mpp_spatiotemporal_model import mpp_spatiotemporal_from_canvas_md`,
    ``,
    `def build_${nm}() -> torch.nn.Module:`,
    `    data = {`,
  ];
  for (const k of keys) {
    const ck = k as keyof MppSpatiotemporalModelNodeData;
    lines.push(`        "${String(ck)}": ${JSON.stringify(firstScalar(merged[ck], defaultMppSpatiotemporalModelData()[ck]))},`);
  }
  lines.push(`    }`);
  lines.push(`    return mpp_spatiotemporal_from_canvas_md(data)`);
  return lines.join("\n");
}
