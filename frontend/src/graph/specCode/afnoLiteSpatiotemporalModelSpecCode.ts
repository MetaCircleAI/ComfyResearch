import type { AfnoLiteSpatiotemporalModelNodeData } from "../../components/nodes/afnoLiteSpatiotemporalModelDefaults";
import { defaultAfnoLiteSpatiotemporalModelData } from "../../components/nodes/afnoLiteSpatiotemporalModelDefaults";

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
  "numSpectralBlocks",
  "maxFrequencyModes",
  "spectralShrinkFactor",
  "seed",
]);

export const DEFAULT_AFNO_LITE_SPATIOTEMPORAL_SPEC_NAME = "AfnoLiteSpatiotemporalModel";

export const DEFAULT_AFNO_LITE_SPATIOTEMPORAL_PARAM_ORDER: (keyof AfnoLiteSpatiotemporalModelNodeData)[] = [
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
  "numSpectralBlocks",
  "maxFrequencyModes",
  "spectralShrinkFactor",
  "seed",
];

export function generateAfnoLiteSpatiotemporalModelSpecCode(
  d: AfnoLiteSpatiotemporalModelNodeData,
  order: string[],
  specName: string,
): string {
  const merged = { ...defaultAfnoLiteSpatiotemporalModelData(), ...d };
  const nm = specName.trim() || DEFAULT_AFNO_LITE_SPATIOTEMPORAL_SPEC_NAME;
  const keys = (order.length ? order : DEFAULT_AFNO_LITE_SPATIOTEMPORAL_PARAM_ORDER).filter((k) => KEYS.has(k));
  const lines: string[] = [
    `import torch`,
    `from comfy_research.engine.afno_lite_spatiotemporal_model import afno_lite_spatiotemporal_from_canvas_md`,
    ``,
    `def build_${nm}() -> torch.nn.Module:`,
    `    data = {`,
  ];
  for (const k of keys) {
    const ck = k as keyof AfnoLiteSpatiotemporalModelNodeData;
    lines.push(
      `        "${String(ck)}": ${JSON.stringify(firstScalar(merged[ck], defaultAfnoLiteSpatiotemporalModelData()[ck]))},`,
    );
  }
  lines.push(`    }`);
  lines.push(`    return afno_lite_spatiotemporal_from_canvas_md(data)`);
  return lines.join("\n");
}

