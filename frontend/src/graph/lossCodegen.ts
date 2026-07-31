/** Loss notebook-codegen fns(自 nodeRegistrySpec.ts 抽出;从 spec
 * export 会成环, 教训)。fn 本体不持 schema 真相。 */
import { defaultCrossEntropyLossData, type CrossEntropyLossNodeData } from "../components/nodes/crossEntropyLossDefaults";
import { defaultMseLossData, type MseLossNodeData } from "../components/nodes/mseLossDefaults";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

export function buildMseTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMseLossData();
  const d = { ...defs, ...(raw as Partial<MseLossNodeData>) } as MseLossNodeData;
  const scale = firstScalar(d.lossScale, defs.lossScale as number);
  return `# === ${title} (mse_loss) ===
import torch
import torch.nn as nn


def fn_${pySym}_criterion():
    base = nn.MSELoss()
    scale = float(${scale})

    class ScaledMSE(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return scale * base(pred, target)

    return ScaledMSE()
`;
}

export function buildCrossEntropyTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultCrossEntropyLossData();
  const d = { ...defs, ...(raw as Partial<CrossEntropyLossNodeData>) } as CrossEntropyLossNodeData;
  const scale = firstScalar(d.lossScale, defs.lossScale as number);
  return `# === ${title} (cross_entropy_loss) ===
import torch
import torch.nn as nn


def fn_${pySym}_criterion():
    base = nn.CrossEntropyLoss()

    class ScaledCE(nn.Module):
        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return float(${scale}) * base(logits, target)

    return ScaledCE()
`;
}

export function buildDiffusionMseLossNotebookPython(pySym: string, title: string): string {
  return `# === ${title} (diffusion_mse_loss) ===
# ${"─".repeat(70)}
# Diffusion training loss (noise MSE)
#
# On the server, trainer_run samples timestep t uniformly in [0, T-1], Gaussian ε, builds x_t from the schedule,
# runs the score MLP, and applies this node's lossScale to MSE(ε̂, ε). This cell is only a short stub.
#
# Loss scale matches trainer (applied to MSE(ε̂, ε)); training loop samples t and noise on the server.
# Local stub: identity scale for documentation.

def fn_${pySym}_loss_scale() -> float:
    return 1.0
`;
}

export function buildKanRegTorch(pySym: string, title: string): string {
  return `# === ${title} (kan_reg) ===
import torch
import torch.nn as nn


def fn_${pySym}_criterion():
    """Lightweight surrogate regularizer for standalone runs (graph may use richer KAN penalties)."""

    class KanRegSurrogate(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return 0.01 * pred.abs().mean()

    return KanRegSurrogate()
`;
}
