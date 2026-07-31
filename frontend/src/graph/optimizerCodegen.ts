/** Optimizer notebook-codegen fns(自 nodeRegistrySpec.ts 抽出,供
 * generatedNodeSpecTypes 的 CODEGEN_ADAPTERS 引用——直接从 nodeRegistrySpec
 * export 会成环, crlServerSideStub 教训)。fn 本体不持 schema 真相。 */
import { defaultAdamOptimizerData, type AdamOptimizerNodeData } from "../components/nodes/adamOptimizerDefaults";
import { defaultLrScheduleData, type LrScheduleNodeData } from "../components/nodes/lrScheduleDefaults";
import { defaultMupLrScheduleData, type MupLrScheduleNodeData } from "../components/nodes/mupLrScheduleDefaults";
import { defaultAdamWOptimizerData, type AdamWOptimizerNodeData } from "../components/nodes/adamWOptimizerDefaults";
import { defaultMuonOptimizerData, type MuonOptimizerNodeData } from "../components/nodes/muonOptimizerDefaults";
import { defaultSgdOptimizerData, type SgdOptimizerNodeData } from "../components/nodes/sgdOptimizerDefaults";
import { defaultSignSgdOptimizerData, type SignSgdOptimizerNodeData } from "../components/nodes/signSgdOptimizerDefaults";
import { defaultShampooOptimizerData, type ShampooOptimizerNodeData } from "../components/nodes/shampooOptimizerDefaults";
import { defaultSoapOptimizerData, type SoapOptimizerNodeData } from "../components/nodes/soapOptimizerDefaults";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

export function buildAdamTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultAdamOptimizerData();
  const d = { ...defs, ...(raw as Partial<AdamOptimizerNodeData>) } as AdamOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const b1 = firstScalar(d.beta1, defs.beta1 as number);
  const b2 = firstScalar(d.beta2, defs.beta2 as number);
  const eps = firstScalar(d.epsilon, defs.epsilon as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  return `# === ${title} (adam_optimizer) ===
import torch


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return torch.optim.Adam(
        params,
        lr=float(${lr}) if lr is None else float(lr),
        betas=(${b1}, ${b2}),
        eps=float(${eps}),
        weight_decay=float(${wd}),
    )
`;
}

export function buildSgdTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultSgdOptimizerData();
  const d = { ...defs, ...(raw as Partial<SgdOptimizerNodeData>) } as SgdOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const mom = firstScalar(d.momentum, defs.momentum as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  return `# === ${title} (sgd_optimizer) ===
import torch


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return torch.optim.SGD(
        params,
        lr=float(${lr}) if lr is None else float(lr),
        momentum=float(${mom}),
        weight_decay=float(${wd}),
    )
`;
}

export function buildMuonTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMuonOptimizerData();
  const d = { ...defs, ...(raw as Partial<MuonOptimizerNodeData>) } as MuonOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const mom = firstScalar(d.momentum, defs.momentum as number);
  return `# === ${title} (muon_optimizer) ===
# Muon is not in core PyTorch; this cell uses SGD + momentum as a practical stand-in for a standalone notebook.
import torch


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return torch.optim.SGD(params, lr=float(${lr}) if lr is None else float(lr), momentum=float(${mom}))
`;
}

export function buildAdamWTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultAdamWOptimizerData();
  const d = { ...defs, ...(raw as Partial<AdamWOptimizerNodeData>) } as AdamWOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const b1 = firstScalar(d.beta1, defs.beta1 as number);
  const b2 = firstScalar(d.beta2, defs.beta2 as number);
  const eps = firstScalar(d.epsilon, defs.epsilon as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  return `# === ${title} (adamw_optimizer) ===
import torch


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return torch.optim.AdamW(
        params,
        lr=float(${lr}) if lr is None else float(lr),
        betas=(${b1}, ${b2}),
        eps=float(${eps}),
        weight_decay=float(${wd}),
    )
`;
}

export function buildSignSgdTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultSignSgdOptimizerData();
  const d = { ...defs, ...(raw as Partial<SignSgdOptimizerNodeData>) } as SignSgdOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  return `# === ${title} (signsgd_optimizer) ===
import torch


class _SignSGD(torch.optim.Optimizer):
    def __init__(self, params, *, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__(params, {"lr": float(lr), "weight_decay": float(weight_decay)})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr0 = float(group["lr"])
            wd0 = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd0 != 0.0:
                    g = g.add(p, alpha=wd0)
                p.add_(g.sign(), alpha=-lr0)
        return loss


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return _SignSGD(params, lr=float(${lr}) if lr is None else float(lr), weight_decay=float(${wd}))
`;
}

export function buildShampooTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultShampooOptimizerData();
  const d = { ...defs, ...(raw as Partial<ShampooOptimizerNodeData>) } as ShampooOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const mom = firstScalar(d.momentum, defs.momentum as number);
  const eps = firstScalar(d.epsilon, defs.epsilon as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  const freq = Math.max(1, Math.round(firstScalar(d.preconditionFrequency, defs.preconditionFrequency as number)));
  const maxDim = Math.max(1, Math.round(firstScalar(d.maxPreconditionerDim, defs.maxPreconditionerDim as number)));
  return `# === ${title} (shampoo_optimizer) ===
from comfy_research.engine.matrix_preconditioner_optimizers import Shampoo


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return Shampoo(
        params,
        lr=float(${lr}) if lr is None else float(lr),
        momentum=float(${mom}),
        eps=float(${eps}),
        weight_decay=float(${wd}),
        precondition_frequency=int(${freq}),
        max_preconditioner_dim=int(${maxDim}),
    )
`;
}

export function buildSoapTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultSoapOptimizerData();
  const d = { ...defs, ...(raw as Partial<SoapOptimizerNodeData>) } as SoapOptimizerNodeData;
  const lr = firstScalar(d.learningRate, defs.learningRate as number);
  const b1 = firstScalar(d.beta1, defs.beta1 as number);
  const b2 = firstScalar(d.beta2, defs.beta2 as number);
  const eps = firstScalar(d.epsilon, defs.epsilon as number);
  const wd = firstScalar(d.weightDecay, defs.weightDecay as number);
  const freq = Math.max(1, Math.round(firstScalar(d.preconditionFrequency, defs.preconditionFrequency as number)));
  const maxDim = Math.max(1, Math.round(firstScalar(d.maxPreconditionerDim, defs.maxPreconditionerDim as number)));
  return `# === ${title} (soap_optimizer) ===
from comfy_research.engine.matrix_preconditioner_optimizers import SOAP


def fn_${pySym}_optimizer(params, *, lr: float | None = None):
    return SOAP(
        params,
        lr=float(${lr}) if lr is None else float(lr),
        betas=(${b1}, ${b2}),
        eps=float(${eps}),
        weight_decay=float(${wd}),
        precondition_frequency=int(${freq}),
        max_preconditioner_dim=int(${maxDim}),
    )
`;
}

export function buildLrScheduleCell(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultLrScheduleData();
  const d = { ...defs, ...(raw as Partial<LrScheduleNodeData>) } as LrScheduleNodeData;
  const kind = String(firstScalar(d.lrSchedule, defs.lrSchedule));
  const warm = Math.floor(firstScalar(d.lrWarmupSteps, defs.lrWarmupSteps));
  const floor = Number(firstScalar(d.cosineLrMinFraction, defs.cosineLrMinFraction));
  return `# === ${title} (lr_schedule) ===
# LR multipliers match the trainer's schedule hooks (constant / cosine / stable-stable-decay).


def fn_${pySym}_lr_factor(step: int, total_steps: int) -> float:
    import math
    kind = ${JSON.stringify(kind)}
    warm = int(${JSON.stringify(warm)})
    floor = float(${JSON.stringify(floor)})
    s = int(max(0, step))
    t = int(max(1, total_steps))
    if warm > 0 and s < warm:
        return float(s + 1) / float(max(1, warm))
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        p = float(s - warm) / float(max(1, t - warm))
        p = min(1.0, max(0.0, p))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    # stable_stable_decay: simplified two-phase multiplier; canvas uses full trainer schedule.
    return 1.0 if s < (t // 2) else 0.5
`;
}

export function buildMupLrScheduleCell(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMupLrScheduleData();
  const d = { ...defs, ...(raw as Partial<MupLrScheduleNodeData>) } as MupLrScheduleNodeData;
  const e = Number(firstScalar(d.mupEmbedLrMult, defs.mupEmbedLrMult));
  const h = Number(firstScalar(d.mupHiddenLrMult, defs.mupHiddenLrMult));
  const o = Number(firstScalar(d.mupOutputLrMult, defs.mupOutputLrMult));
  return `# === ${title} (mup_lr_schedule) ===
# μP LR multipliers from the node (applied in server trainer wiring).


def fn_${pySym}_mup_lr_mults() -> dict:
    return {
        "mup_embed_lr_mult": float(${JSON.stringify(e)}),
        "mup_hidden_lr_mult": float(${JSON.stringify(h)}),
        "mup_output_lr_mult": float(${JSON.stringify(o)}),
    }
`;
}
