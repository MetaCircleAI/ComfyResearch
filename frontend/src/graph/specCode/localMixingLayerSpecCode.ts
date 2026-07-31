import type { LocalMixingLayerNodeData } from "../../components/nodes/localMixingLayerDefaults";
import { defaultLocalMixingLayerData } from "../../components/nodes/localMixingLayerDefaults";
import { camelToSnakeCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["modelDim", "kernelSize", "seed"]);

export const DEFAULT_LOCAL_MIXING_LAYER_SPEC_NAME = "LocalMixingLayer";

export const DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER: (keyof LocalMixingLayerNodeData)[] = [
  "modelDim",
  "kernelSize",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyIntDefault(key: keyof LocalMixingLayerNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "number") return String(Math.floor(s));
  return JSON.stringify(s);
}

/** Match server atomic_layer_chain.build_atomic_layer_module clamping. */
function clampKernelSize(ks: number): number {
  let k = Math.max(3, Math.floor(ks));
  if (k % 2 === 0) k += 1;
  return k;
}

/**
 * Self-contained PyTorch shown in Code notebook — mirrors
 * ``comfy_research.engine.local_mixing`` (``CausalDepthwiseConv1d`` + residual).
 */
export function generateLocalMixingLayerSpecCode(
  d: LocalMixingLayerNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_LOCAL_MIXING_LAYER_SPEC_NAME;
  const merged = { ...defaultLocalMixingLayerData(), ...d };
  const keys = (order.length ? order : DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER).filter((k) =>
    KNOWN_KEYS.has(k),
  );

  const lines: string[] = [
    `import torch`,
    `import torch.nn as nn`,
    `import torch.nn.functional as F`,
    ``,
    `# Mirrors comfy_research/engine/local_mixing.py — trainer builds this via`,
    `# atomic_layer_chain.build_atomic_layer_module for graph node local_mixing_layer.`,
    ``,
    `class CausalDepthwiseConv1d(nn.Module):`,
    `    """Depthwise causal conv on sequence axis; layout [batch, T, channels]."""`,
    ``,
    `    def __init__(self, channels: int, kernel_size: int) -> None:`,
    `        super().__init__()`,
    `        c = int(channels)`,
    `        ks = int(kernel_size)`,
    `        if c < 1:`,
    `            raise ValueError("channels must be >= 1")`,
    `        if ks < 3 or ks % 2 != 1:`,
    `            raise ValueError("kernel_size must be odd and >= 3")`,
    `        self.channels = c`,
    `        self.kernel_size = ks`,
    `        self.conv = nn.Conv1d(c, c, ks, groups=c, bias=True)`,
    ``,
    `    def forward(self, x: torch.Tensor) -> torch.Tensor:`,
    `        if x.dim() != 3:`,
    `            raise ValueError(f"expected [batch, seq, channels], got {tuple(x.shape)}")`,
    `        z = x.transpose(1, 2)`,
    `        z = F.pad(z, (self.kernel_size - 1, 0))`,
    `        z = self.conv(z)`,
    `        return z.transpose(1, 2)`,
    ``,
    `class ${name}(nn.Module):`,
    `    """Residual causal depthwise mixing: y = x + conv(x) (same as CausalLocalMixingResidual)."""`,
    ``,
    `    def __init__(`,
    `        self,`,
  ];

  for (const k of keys) {
    const ck = k as keyof LocalMixingLayerNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: int = ${formatPyIntDefault(ck, merged[ck])},`);
  }
  lines.push(`    ) -> None:`);
  lines.push(`        super().__init__()`);
  lines.push(`        md = max(1, int(model_dim))`);
  lines.push(`        ks_raw = int(kernel_size)`);
  lines.push(`        if ks_raw < 3:`);
  lines.push(`            ks = 3`);
  lines.push(`        elif ks_raw % 2 == 0:`);
  lines.push(`            ks = ks_raw + 1`);
  lines.push(`        else:`);
  lines.push(`            ks = ks_raw`);
  lines.push(`        self.model_dim = md`);
  lines.push(`        self.kernel_size = int(ks)`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        self.mix = CausalDepthwiseConv1d(self.model_dim, self.kernel_size)`);

  const md = Math.max(1, Math.floor(Number(firstScalar(merged.modelDim, 64)) || 64));
  const ksDisplay = clampKernelSize(Number(firstScalar(merged.kernelSize, 5)) || 5);
  lines.push(
    ``,
    `        # First sweep from node: model_dim=${md}, kernel_size clamped to ${ksDisplay} (same rules as server)`,
  );

  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        """[B, T, C] -> same; rank-2 [B, C] is treated as T=1."""`);
  lines.push(`        if x.dim() == 2:`);
  lines.push(`            y = x.unsqueeze(1) + self.mix(x.unsqueeze(1))`);
  lines.push(`            return y.squeeze(1)`);
  lines.push(`        if x.dim() != 3:`);
  lines.push(`            raise ValueError(f"local_mixing_layer expects rank 2 or 3, got {tuple(x.shape)}")`);
  lines.push(`        return x + self.mix(x)`);
  return lines.join("\n");
}
