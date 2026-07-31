import { defaultAbsolutePosEmbedLayerData, type AbsolutePosEmbedLayerNodeData } from "../components/nodes/absolutePosEmbedLayerDefaults";
import { defaultActivationLayerData, type ActivationLayerNodeData } from "../components/nodes/activationLayerDefaults";
import { defaultAfnoAtomicLayerData, type AfnoAtomicLayerNodeData } from "../components/nodes/afnoAtomicLayerDefaults";
import { defaultEmbeddingLayerData, type EmbeddingLayerNodeData } from "../components/nodes/embeddingLayerDefaults";
import { defaultLayerNormLayerData, type LayerNormLayerNodeData } from "../components/nodes/layerNormLayerDefaults";
import { defaultLinearLayerData, type LinearLayerNodeData } from "../components/nodes/linearLayerDefaults";
import { defaultLocalMixingLayerData, type LocalMixingLayerNodeData } from "../components/nodes/localMixingLayerDefaults";
import { defaultRotaryEmbedLayerData, type RotaryEmbedLayerNodeData } from "../components/nodes/rotaryEmbedLayerDefaults";
import { defaultRmsNormLayerData, type RmsNormLayerNodeData } from "../components/nodes/rmsNormLayerDefaults";
import { defaultUnembeddingLayerData, type UnembeddingLayerNodeData } from "../components/nodes/unembeddingLayerDefaults";
import { defaultCausalMaskNodeData, type CausalMaskNodeData } from "../components/nodes/causalMaskDefaults";
import { defaultEinsumNodeData, type EinsumNodeData } from "../components/nodes/einsumDefaults";
import { defaultFlattenNodeData, readFlattenExceptDim, type FlattenNodeData } from "../components/nodes/flattenDefaults";
import { defaultReshapeNodeData, type ReshapeNodeData } from "../components/nodes/reshapeDefaults";
import { defaultSoftmaxNodeData, type SoftmaxNodeData } from "../components/nodes/softmaxDefaults";
import {
  generateAbsolutePosEmbedLayerSpecCode,
  DEFAULT_ABSOLUTE_POS_EMBED_LAYER_PARAM_ORDER,
} from "./specCode/absolutePosEmbedLayerSpecCode";
import { generateActivationLayerSpecCode, DEFAULT_ACTIVATION_LAYER_PARAM_ORDER } from "./specCode/activationLayerSpecCode";
import { generateEmbeddingLayerSpecCode, DEFAULT_EMBEDDING_LAYER_PARAM_ORDER } from "./specCode/embeddingLayerSpecCode";
import { generateLayerNormLayerSpecCode, DEFAULT_LAYER_NORM_LAYER_PARAM_ORDER } from "./specCode/layerNormLayerSpecCode";
import { generateLinearLayerSpecCode, DEFAULT_LINEAR_LAYER_PARAM_ORDER } from "./specCode/linearLayerSpecCode";
import {
  generateLocalMixingLayerSpecCode,
  DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER,
} from "./specCode/localMixingLayerSpecCode";
import {
  generateRotaryEmbedLayerSpecCode,
  DEFAULT_ROTARY_EMBED_LAYER_PARAM_ORDER,
} from "./specCode/rotaryEmbedLayerSpecCode";
import { generateRmsNormLayerSpecCode, DEFAULT_RMS_NORM_LAYER_PARAM_ORDER } from "./specCode/rmsNormLayerSpecCode";
import { generateUnembeddingLayerSpecCode, DEFAULT_UNEMBEDDING_LAYER_PARAM_ORDER } from "./specCode/unembeddingLayerSpecCode";
import {
  DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER,
  generateAfnoAtomicLayerSpecCode,
  type AfnoAtomicLayerKind,
} from "./specCode/afnoAtomicLayerSpecCode";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

function isLayerFrozen(raw: Record<string, unknown>): boolean {
  const d = raw ?? {};
  if (d["freeze"] === true || d["freeze"] === "true") return true;
  if (d["trainable"] === false || d["trainable"] === "false") return true;
  if (d["requiresGrad"] === false || d["requiresGrad"] === "false") return true;
  if (d["requires_grad"] === false || d["requires_grad"] === "false") return true;
  return false;
}

function wrapLayerModule(
  title: string,
  nodeType: string,
  pySym: string,
  className: string,
  specBody: string,
  seed?: number,
  frozen?: boolean,
): string {
  const seedLine = typeof seed === "number" && Number.isFinite(seed) ? `    torch.manual_seed(int(${JSON.stringify(seed)}))\n` : "";
  if (frozen) {
    const factoryBody = `    import torch\n${seedLine}    m = ${className}()\n    m.requires_grad_(False)\n    return m`;
    return `# === ${title} (${nodeType}) [FROZEN] ===
${specBody}


def fn_${pySym}_module() -> ${className}:
${factoryBody}
`;
  }
  const factoryBody = `    import torch\n${seedLine}    return ${className}()`;
  return `# === ${title} (${nodeType}) ===
${specBody}


def fn_${pySym}_module() -> ${className}:
${factoryBody}
`;
}

export function buildLinearLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultLinearLayerData();
  const d = { ...defs, ...(raw as Partial<LinearLayerNodeData>) } as LinearLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_LINEAR_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateLinearLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "linear_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

export function buildActivationLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultActivationLayerData();
  const d = { ...defs, ...(raw as Partial<ActivationLayerNodeData>) } as ActivationLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_ACTIVATION_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateActivationLayerSpecCode(d, order, className);
  return wrapLayerModule(title, "activation_layer", pySym, className, body, 0, isLayerFrozen(raw));
}

export function buildLayerNormLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultLayerNormLayerData();
  const d = { ...defs, ...(raw as Partial<LayerNormLayerNodeData>) } as LayerNormLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_LAYER_NORM_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateLayerNormLayerSpecCode(d, order, className);
  return wrapLayerModule(title, "layer_norm_layer", pySym, className, body, undefined, isLayerFrozen(raw));
}

export function buildRmsNormLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultRmsNormLayerData();
  const d = { ...defs, ...(raw as Partial<RmsNormLayerNodeData>) } as RmsNormLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_RMS_NORM_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateRmsNormLayerSpecCode(d, order, className);
  return wrapLayerModule(title, "rms_norm_layer", pySym, className, body, undefined, isLayerFrozen(raw));
}

export function buildEmbeddingLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultEmbeddingLayerData();
  const d = { ...defs, ...(raw as Partial<EmbeddingLayerNodeData>) } as EmbeddingLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_EMBEDDING_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateEmbeddingLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "embedding_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

export function buildUnembeddingLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultUnembeddingLayerData();
  const d = { ...defs, ...(raw as Partial<UnembeddingLayerNodeData>) } as UnembeddingLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_UNEMBEDDING_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateUnembeddingLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "unembedding_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

export function buildAbsolutePosEmbedLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultAbsolutePosEmbedLayerData();
  const d = { ...defs, ...(raw as Partial<AbsolutePosEmbedLayerNodeData>) } as AbsolutePosEmbedLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_ABSOLUTE_POS_EMBED_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateAbsolutePosEmbedLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "absolute_pos_embed_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

export function buildRotaryEmbedLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultRotaryEmbedLayerData();
  const d = { ...defs, ...(raw as Partial<RotaryEmbedLayerNodeData>) } as RotaryEmbedLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_ROTARY_EMBED_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateRotaryEmbedLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "rotary_embed_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

export function buildLocalMixingLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultLocalMixingLayerData();
  const d = { ...defs, ...(raw as Partial<LocalMixingLayerNodeData>) } as LocalMixingLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_LOCAL_MIXING_LAYER_PARAM_ORDER];
  const className = `CrLayer_${pySym}`;
  const body = generateLocalMixingLayerSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapLayerModule(title, "local_mixing_layer", pySym, className, body, seed, isLayerFrozen(raw));
}

function buildAfnoAtomicLayerModule(
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
  kind: AfnoAtomicLayerKind,
): string {
  const defs = defaultAfnoAtomicLayerData();
  const d = { ...defs, ...(raw as Partial<AfnoAtomicLayerNodeData>) } as AfnoAtomicLayerNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_AFNO_ATOMIC_LAYER_PARAM_ORDER];
  const specName = `CrLayer_${pySym}`;
  const body = generateAfnoAtomicLayerSpecCode(kind, d, order, specName);
  const seed = firstScalar(d.seed, defs.seed as number);
  const frozen = isLayerFrozen(raw);
  const freezeLine = frozen ? "    m.requires_grad_(False)\n" : "";
  const freezeComment = frozen ? " [FROZEN]" : "";
  if (frozen) {
    return `# === ${title} (${kind})${freezeComment} ===
${body}


def fn_${pySym}_module() -> torch.nn.Module:
    import torch
    torch.manual_seed(int(${JSON.stringify(seed)}))
    m = build_${specName}()
    m.requires_grad_(False)
    return m
`;
  }
  return `# === ${title} (${kind}) ===
${body}


def fn_${pySym}_module() -> torch.nn.Module:
    import torch
    torch.manual_seed(int(${JSON.stringify(seed)}))
    return build_${specName}()
`;
}

export function buildAfnoPatchEmbedLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  return buildAfnoAtomicLayerModule(pySym, title, raw, "afno_patch_embed_layer");
}

export function buildAfnoSpectralMixerLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  return buildAfnoAtomicLayerModule(pySym, title, raw, "afno_spectral_mixer_layer");
}

export function buildAfnoEncoderBlockLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  return buildAfnoAtomicLayerModule(pySym, title, raw, "afno_encoder_block_layer");
}

export function buildAfnoPatchDecodeLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  return buildAfnoAtomicLayerModule(pySym, title, raw, "afno_patch_decode_layer");
}

export function buildSoftmaxLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultSoftmaxNodeData();
  const d = { ...defs, ...(raw as Partial<SoftmaxNodeData>) } as SoftmaxNodeData;
  const dim = Math.floor(firstScalar(d.dimension, defs.dimension as number));
  return `# === ${title} (softmax) ===
import torch
import torch.nn as nn


class CrLayer_${pySym}(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = nn.Softmax(dim=int(${JSON.stringify(dim)}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sm(x)


def fn_${pySym}_module() -> CrLayer_${pySym}:
    return CrLayer_${pySym}()
`;
}

export function buildFlattenLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultFlattenNodeData();
  const d = { ...defs, ...(raw as Partial<FlattenNodeData>) } as FlattenNodeData;
  void readFlattenExceptDim(d.exceptDim);
  return `# === ${title} (flatten) ===
# exceptDim on the canvas is handled in the server combiner; this export uses \`torch.nn.Flatten(start_dim=1)\` (most MLP stacks).
import torch
import torch.nn as nn


class CrLayer_${pySym}(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat(x)


def fn_${pySym}_module() -> CrLayer_${pySym}:
    return CrLayer_${pySym}()
`;
}

export function buildReshapeLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultReshapeNodeData();
  const d = { ...defs, ...(raw as Partial<ReshapeNodeData>) } as ReshapeNodeData;
  const rule = String(firstScalar(d.reshapeRule, defs.reshapeRule as string));
  return `# === ${title} (reshape) ===
# Rule from node: ${rule.replace(/\\/g, "\\\\")}
# Full einops-style reshape is resolved in the server sequential builder; this cell keeps tensors unchanged so imports stay runnable.
import torch
import torch.nn as nn


class CrLayer_${pySym}(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def fn_${pySym}_module() -> CrLayer_${pySym}:
    return CrLayer_${pySym}()
`;
}

export function buildEinsumLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultEinsumNodeData();
  const d = { ...defs, ...(raw as Partial<EinsumNodeData>) } as EinsumNodeData;
  const eq = String(firstScalar(d.equation, defs.equation as string));
  return `# === ${title} (einsum) ===
# Equation: ${eq.replace(/\\/g, "\\\\")}
import torch
import torch.nn as nn


class CrLayer_${pySym}(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum(${JSON.stringify(eq)}, a, b)


def fn_${pySym}_module() -> CrLayer_${pySym}:
    return CrLayer_${pySym}()
`;
}

export function buildCausalMaskLayerModule(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultCausalMaskNodeData();
  const d = { ...defs, ...(raw as Partial<CausalMaskNodeData>) } as CausalMaskNodeData;
  const diag = Math.floor(firstScalar(d.diagonalOffset, defs.diagonalOffset as number));
  const t = 2048;
  return `# === ${title} (causal_mask) ===
# Upper-triangular boolean mask with diagonal offset from the node (fixed T=${t} for export).
import torch
import torch.nn as nn


class CrLayer_${pySym}(nn.Module):
    def __init__(self):
        super().__init__()
        t = int(${JSON.stringify(t)})
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=int(${JSON.stringify(diag)})),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask


def fn_${pySym}_module() -> CrLayer_${pySym}:
    return CrLayer_${pySym}()
`;
}
