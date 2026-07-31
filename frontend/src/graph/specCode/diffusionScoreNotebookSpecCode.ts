import {
  defaultDiffusionScoreModelData,
  type DiffusionScoreModelNodeData,
} from "../../components/nodes/diffusionScoreModelDefaults";
import { readmeDiffusionScore } from "./notebookImplementationReadme";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

const TIMESTEP_EMBED = `def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    import math
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
`;

export function buildDiffusionScoreNotebookPython(
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
): string {
  const defs = defaultDiffusionScoreModelData();
  const d = { ...defs, ...(raw as Partial<DiffusionScoreModelNodeData>) } as DiffusionScoreModelNodeData;
  const dataDim = Math.max(1, Math.floor(Number(firstScalar(d.inputDim, 8))));
  const hiddenDim = Math.max(8, Math.floor(Number(firstScalar(d.hiddenDim, 128))));
  const depth = Math.max(1, Math.floor(Number(firstScalar(d.depth, 3))));
  const timeEmbedDim = Math.max(8, Math.floor(Number(firstScalar(d.timeEmbedDim, 64))));
  const maxTimesteps = Math.max(2, Math.floor(Number(firstScalar(d.diffusionTimesteps, 100))));
  const seed = Math.floor(Number(firstScalar(d.seed, 0)));
  const className = `CrDiffusionScore_${pySym}`;
  const readme = readmeDiffusionScore(dataDim, hiddenDim, depth, timeEmbedDim, maxTimesteps);

  return `# === ${title} (diffusion_score_model) ===
${readme}
# ε_θ(x_t, t) MLP — matches comfy_research/engine/diffusion_score_model.py (DDPM-style noise prediction).
import torch
import torch.nn as nn
import torch.nn.functional as F

${TIMESTEP_EMBED}


class ${className}(nn.Module):
    """Predicts noise ε̂ given noisy data x_t and discrete diffusion index t (same layout as server)."""

    def __init__(self) -> None:
        super().__init__()
        self.data_dim = int(${dataDim})
        self.hidden_dim = int(${hiddenDim})
        self.max_timesteps = max(2, int(${maxTimesteps}))
        te = max(8, int(${timeEmbedDim}))
        self._te_dim = te
        # Map sinusoidal time features → hidden (SiLU MLP); time_in is a learned linear preprocess.
        self.time_embed = nn.Sequential(
            nn.Linear(te, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_in = nn.Linear(te, te)
        dep = max(1, int(${depth}))
        layers = [
            nn.Linear(self.data_dim + self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(dep - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(self.hidden_dim, self.data_dim))
        self.net = nn.Sequential(*layers)

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        raw_e = _timestep_embedding(t.clamp(min=0, max=self.max_timesteps - 1), self._te_dim)
        return self.time_embed(self.time_in(raw_e))

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_noisy: [B, data_dim], t: Long[B]; returns ε̂ with same shape as x_noisy.
        te = self.embed_time(t)
        h = torch.cat([x_noisy, te], dim=-1)
        return self.net(h)


def fn_${pySym}_model() -> ${className}:
    torch.manual_seed(${seed})
    return ${className}()
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
