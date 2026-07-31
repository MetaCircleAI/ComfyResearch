import type { AfnoAtomicLayerNodeData } from "../../components/nodes/afnoAtomicLayerDefaults";
import { defaultAfnoAtomicLayerData } from "../../components/nodes/afnoAtomicLayerDefaults";

export type AfnoAtomicLayerKind =
  | "afno_patch_embed_layer"
  | "afno_spectral_mixer_layer"
  | "afno_encoder_block_layer"
  | "afno_patch_decode_layer";

const BUILDER_BY_KIND: Record<AfnoAtomicLayerKind, string> = {
  afno_patch_embed_layer: "afno_patch_embed_layer_from_canvas_md",
  afno_spectral_mixer_layer: "afno_spectral_mixer_layer_from_canvas_md",
  afno_encoder_block_layer: "afno_encoder_block_layer_from_canvas_md",
  afno_patch_decode_layer: "afno_patch_decode_layer_from_canvas_md",
};

const DEFAULT_SPEC_NAME_BY_KIND: Record<AfnoAtomicLayerKind, string> = {
  afno_patch_embed_layer: "AfnoPatchEmbedLayer",
  afno_spectral_mixer_layer: "AfnoSpectralMixerLayer",
  afno_encoder_block_layer: "AfnoEncoderBlockLayer",
  afno_patch_decode_layer: "AfnoPatchDecodeLayer",
};

const KNOWN_KEYS = new Set([
  "contextFrames",
  "channels",
  "gridSize",
  "inputDim",
  "outputDim",
  "patchSize",
  "embedDim",
  "numHeads",
  "ffRatio",
  "dropout",
  "numSpectralBlocks",
  "maxFrequencyModes",
  "spectralShrinkFactor",
  "seed",
]);

export const DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER: (keyof AfnoAtomicLayerNodeData)[] = [
  "contextFrames",
  "channels",
  "gridSize",
  "inputDim",
  "outputDim",
  "patchSize",
  "embedDim",
  "numHeads",
  "ffRatio",
  "dropout",
  "numSpectralBlocks",
  "maxFrequencyModes",
  "spectralShrinkFactor",
  "seed",
];

function firstScalar(v: unknown, fallback: unknown): unknown {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return v.length ? v[0] : fallback;
  return v;
}

export function defaultAfnoAtomicSpecName(kind: AfnoAtomicLayerKind): string {
  return DEFAULT_SPEC_NAME_BY_KIND[kind];
}

export function generateAfnoAtomicLayerSpecCode(
  kind: AfnoAtomicLayerKind,
  d: AfnoAtomicLayerNodeData,
  order: string[],
  specName: string,
): string {
  const merged = { ...defaultAfnoAtomicLayerData(), ...d };
  const nm = specName.trim() || defaultAfnoAtomicSpecName(kind);
  const builder = BUILDER_BY_KIND[kind];
  const keys = (order.length ? order : DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  const lines: string[] = [
    `import torch`,
    `from comfy_research.engine.afno_lite_spatiotemporal_model import ${builder}`,
    ``,
    `def build_${nm}() -> torch.nn.Module:`,
    `    data = {`,
  ];
  for (const k of keys) {
    const ck = k as keyof AfnoAtomicLayerNodeData;
    lines.push(`        "${String(ck)}": ${JSON.stringify(firstScalar(merged[ck], defaultAfnoAtomicLayerData()[ck]))},`);
  }
  lines.push(`    }`);
  lines.push(`    return ${builder}(data)`);
  return lines.join("\n");
}

