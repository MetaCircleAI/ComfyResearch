/**
 * Self-contained PyTorch for Code notebook cells (matches comfy_research/engine token bundles).
 * Inlines CausalDepthwiseConv1d so notebooks run without the ComfyResearch package on PYTHONPATH.
 */
import {
  defaultAlternativeArchTokenLmData,
  type AlternativeArchTokenLmNodeData,
  type ArchLmKind,
} from "../../components/nodes/alternativeArchModelDefaults";
import { readmeAlternativeArchTokenLm, type AltArchReadmeParams } from "./notebookImplementationReadme";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

function oddKernel(k: number): number {
  let ks = Math.max(0, Math.floor(k));
  if (ks >= 3 && ks % 2 === 0) ks += 1;
  return ks;
}

const CAUSAL_DEPTHWISE_CONV = `class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal conv along sequence [B, T, C] — same as comfy_research/engine/local_mixing.py."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        c = int(channels)
        ks = int(kernel_size)
        if c < 1:
            raise ValueError("channels must be >= 1")
        if ks < 3 or ks % 2 != 1:
            raise ValueError("kernel_size must be odd and >= 3")
        self.channels = c
        self.kernel_size = ks
        self.conv = nn.Conv1d(c, c, ks, groups=c, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, seq, channels], got {tuple(x.shape)}")
        z = x.transpose(1, 2)
        z = F.pad(z, (self.kernel_size - 1, 0))
        z = self.conv(z)
        return z.transpose(1, 2)
`;

function localMixBlock(localKernel: number, modelDim: number): string {
  const lk = oddKernel(localKernel);
  if (lk < 3) {
    return `        self.local_mix = None`;
  }
  return `        self.local_mix = CausalDepthwiseConv1d(${modelDim}, ${lk})`;
}

function localMixForward(): string {
  return `        if self.local_mix is not None:
            h = h + self.local_mix(h)`;
}

export function buildAlternativeArchTokenLmNotebookPython(
  pySym: string,
  title: string,
  nodeType: ArchLmKind,
  raw: Record<string, unknown>,
): string {
  const defs = defaultAlternativeArchTokenLmData(nodeType);
  const d = { ...defs, ...(raw as Partial<AlternativeArchTokenLmNodeData>) } as AlternativeArchTokenLmNodeData;
  const vocab = Math.max(2, Math.floor(Number(firstScalar(d.vocabSize, 100))));
  const modelDim = Math.max(1, Math.floor(Number(firstScalar(d.embedDim, 32))));
  const contextLength = Math.max(1, Math.floor(Number(firstScalar(d.contextLength, 8))));
  const seed = Math.floor(Number(firstScalar(d.seed, 0)));
  const localK = Math.floor(Number(firstScalar(d.localMixingKernel, 0)));
  const className = `CrModel_${pySym}`;

  const causalStr = String(firstScalar(d.causalAttention, "yes")).toLowerCase();
  const causal = !(causalStr === "no" || causalStr === "false" || causalStr === "0");

  const numHeadsAll = Math.max(1, Math.floor(Number(firstScalar(d.numHeads, 4))));
  const numLayersAll = Math.max(1, Math.floor(Number(firstScalar(d.numLayers, 2))));
  const depthAll = Math.max(1, Math.floor(Number(firstScalar(d.depth, 2))));
  let convKAll = Math.floor(Number(firstScalar(d.convKernel, 7)));
  if (convKAll >= 3 && convKAll % 2 === 0) convKAll += 1;
  if (convKAll < 3) convKAll = 3;
  const ffMulAll = Math.max(1, Math.floor(Number(firstScalar(d.ffMult, 2))));
  const numSlotsAll = Math.max(1, Math.floor(Number(firstScalar(d.numSlots, 4))));
  const slotItersAll = Math.max(1, Math.floor(Number(firstScalar(d.slotIters, 3))));
  const readmeParams: AltArchReadmeParams = {
    vocab,
    modelDim,
    contextLength,
    localMixKernel: localK,
    numHeads: numHeadsAll,
    causal,
    numLayers: numLayersAll,
    depth: depthAll,
    convKernel: convKAll,
    ffMult: ffMulAll,
    numSlots: numSlotsAll,
    slotIters: slotItersAll,
  };
  const readme = readmeAlternativeArchTokenLm(nodeType, readmeParams);

  let extraClasses = "";

  if (nodeType === "linear_attention_model") {
    const numHeads = numHeadsAll;
    extraClasses = `
class _LinearAttentionCore(nn.Module):
    """Performs φ(Q)φ(K)^T V style mixing with ELU+1 features; causal branch uses cumulative sums (no L×L softmax)."""

    def __init__(self, model_dim: int, context_length: int, num_heads: int, *, causal: bool = True) -> None:
        super().__init__()
        if model_dim < 1 or context_length < 1 or num_heads < 1:
            raise ValueError("model_dim, context_length, num_heads must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.context_length = int(context_length)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.causal = bool(causal)
        dd = self.model_dim
        self.w_q = nn.Linear(dd, dd, bias=True)
        self.w_k = nn.Linear(dd, dd, bias=True)
        self.w_v = nn.Linear(dd, dd, bias=True)
        self.w_o = nn.Linear(dd, dd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, dd = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0
        if self.causal:
            kv = torch.cumsum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
            num = torch.einsum("bhld,bhldm->bhlm", phi_q, kv)
            den = torch.einsum("bhld,bhld->bhl", phi_q, torch.cumsum(phi_k, dim=2)).clamp(min=1e-6)
            mixed = num / den.unsqueeze(-1)
        else:
            sum_kv = torch.sum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=2, keepdim=True)
            sum_k = torch.sum(phi_k, dim=2, keepdim=True).clamp(min=1e-6)
            num = torch.einsum("bhld,bh1dm->bhlm", phi_q, sum_kv)
            den = torch.einsum("bhld,bh1d->bhl", phi_q, sum_k).clamp(min=1e-6)
            mixed = num / den.unsqueeze(-1)
        y = mixed.transpose(1, 2).reshape(b, l, dd)
        return self.w_o(y)


class ${className}(nn.Module):
    """Token LM: embed [B,L] → optional local DW conv → linear attention on [B,L,D] → logits [B,V] last pos."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = ${vocab}
        self.model_dim = ${modelDim}
        self.context_length = ${contextLength}
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
${localMixBlock(localK, modelDim)}
        self.block = _LinearAttentionCore(self.model_dim, self.context_length, ${numHeads}, causal=${causal ? "True" : "False"})
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long[B, L] with L == context_length; returns Float[B, V] next-token logits at L-1.
        h = self.embedding(token_ids.long())
${localMixForward()}
        h = self.block(h)
        return self.lm_head(h[:, -1, :])
`;
  } else if (nodeType === "diagonal_ssm_token_model") {
    const numLayers = numLayersAll;
    extraClasses = `
class _DiagonalSsmCore(nn.Module):
    """Diagonal recurrence h ← exp(A_t)⊙h + B_t⊙x_t with A_t,B_t from the current input x_t."""

    def __init__(self, model_dim: int, context_length: int) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.context_length = int(context_length)
        dd = self.model_dim
        self.proj_a = nn.Linear(dd, dd, bias=True)
        self.proj_b = nn.Linear(dd, dd, bias=True)
        self.out_proj = nn.Linear(dd, dd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, dd = x.shape
        h = torch.zeros(b, dd, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(l):
            xt = x[:, t, :]
            a_t = -F.softplus(self.proj_a(xt))
            b_t = self.proj_b(xt)
            h = torch.exp(a_t) * h + b_t * xt
            outs.append(self.out_proj(h))
        return torch.stack(outs, dim=1)


class ${className}(nn.Module):
    """Token LM: stacked diagonal SSM cores with residual + LN + LM head on last token."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = ${vocab}
        self.model_dim = ${modelDim}
        self.context_length = ${contextLength}
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
${localMixBlock(localK, modelDim)}
        self.layers = nn.ModuleList(
            [_DiagonalSsmCore(self.model_dim, self.context_length) for _ in range(${numLayers})]
        )
        self.ln = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long[B, L]; logits: Float[B, V] for final position.
        h = self.embedding(token_ids.long())
${localMixForward()}
        for layer in self.layers:
            h = h + layer(h)
        h = self.ln(h)
        return self.lm_head(h[:, -1, :])
`;
  } else if (nodeType === "rwkv_time_mix_token_model") {
    const depth = depthAll;
    extraClasses = `
class _RwkvLiteBlock(nn.Module):
    """Time-mix (scalar decay gate over recurrent state) + gated GELU feed-forward."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.w_k = nn.Linear(dim, dim, bias=True)
        self.w_v = nn.Linear(dim, dim, bias=True)
        self.w_r = nn.Linear(dim, dim, bias=True)
        self.decay = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ff_gate = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.ln1(x)
        b, t, d = z.shape
        h = torch.zeros(b, d, device=z.device, dtype=z.dtype)
        outs = []
        for i in range(t):
            zi = z[:, i, :]
            kk = torch.tanh(self.w_k(zi))
            vv = self.w_v(zi)
            rr = torch.sigmoid(self.w_r(zi))
            dec = torch.sigmoid(self.decay(zi))
            payload = kk * vv
            h = dec * h + (1.0 - dec) * payload
            outs.append(rr * h + (1.0 - rr) * zi)
        tm = torch.stack(outs, dim=1)
        x = residual + self.o_proj(tm)
        residual = x
        z2 = self.ln2(x)
        g, up = self.ff_gate(z2).chunk(2, dim=-1)
        x = residual + torch.sigmoid(g) * F.gelu(up)
        return x


class ${className}(nn.Module):
    """Token LM: gated recurrence over time + gated FFN blocks → last-step logits."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = ${vocab}
        self.model_dim = ${modelDim}
        self.context_length = ${contextLength}
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
${localMixBlock(localK, modelDim)}
        self.blocks = nn.ModuleList([_RwkvLiteBlock(self.model_dim) for _ in range(${depth})])
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long[B, L]; logits: Float[B, V] at last index.
        h = self.embedding(token_ids.long())
${localMixForward()}
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(h[:, -1, :])
`;
  } else if (nodeType === "hyena_like_conv_model") {
    const depth = depthAll;
    const ks = convKAll;
    const ffMult = ffMulAll;
    extraClasses = `
class _HyenaLikeBlock(nn.Module):
    """Pre-norm + residual causal depthwise conv + gated pointwise FFN (Hyena-style block)."""

    def __init__(self, dim: int, kernel_size: int, ff_mult: int = 2) -> None:
        super().__init__()
        ks = int(kernel_size)
        if ks < 3 or ks % 2 == 0:
            ks = ks + (1 - ks % 2)
            if ks < 3:
                ks = 3
        self.ln = nn.LayerNorm(dim)
        self.conv = CausalDepthwiseConv1d(dim, ks)
        self.norm_conv = nn.LayerNorm(dim)
        hidden = max(1, int(dim * ff_mult))
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.ln(x)
        z = z + self.conv(z)
        z = self.norm_conv(z)
        g = torch.sigmoid(self.gate(z))
        z = g * self.ff(z)
        return residual + z


class ${className}(nn.Module):
    """Token LM: depthwise-causal conv + gated FFN blocks (Hyena-style) → LN → last-step logits."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = ${vocab}
        self.model_dim = ${modelDim}
        self.context_length = ${contextLength}
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
${localMixBlock(localK, modelDim)}
        self.blocks = nn.ModuleList(
            [_HyenaLikeBlock(self.model_dim, ${ks}, ff_mult=${ffMult}) for _ in range(${depth})]
        )
        self.out_ln = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long[B, L]; logits: Float[B, V] last position.
        h = self.embedding(token_ids.long())
${localMixForward()}
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.lm_head(h[:, -1, :])
`;
  } else if (nodeType === "slot_attention_token_model") {
    const numSlots = numSlotsAll;
    const slotIters = slotItersAll;
    extraClasses = `
class _SlotAttention(nn.Module):
    """Iterative dot-product attention from K slots to L tokens; GRU + MLP refine slots each round."""

    def __init__(self, dim: int, num_slots: int, iters: int = 3, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.iters = max(1, int(iters))
        self.eps = float(eps)
        self.scale = dim**-0.5
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim) * 0.1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, _, d = inputs.shape
        slots = self.slots_mu.expand(b, -1, -1).contiguous()
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            attn = torch.einsum("bqd,bkd->bqk", q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = attn / (torch.sum(attn, dim=-2, keepdim=True) + self.eps)
            updates = torch.einsum("bqk,bkd->bqd", attn, v)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(slots)
        return slots


class ${className}(nn.Module):
    """Token LM: slot attention pools the sequence to slots, mean-pool → MLP → vocab logits (set readout)."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = ${vocab}
        self.model_dim = ${modelDim}
        self.context_length = ${contextLength}
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
${localMixBlock(localK, modelDim)}
        self.slot_attn = _SlotAttention(self.model_dim, num_slots=${numSlots}, iters=${slotIters})
        self.post = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, self.model_dim), nn.GELU())
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long[B, L]; logits: Float[B, V] summarizing the whole sequence (not per-step).
        h = self.embedding(token_ids.long())
${localMixForward()}
        slots = self.slot_attn(h)
        pooled = slots.mean(dim=1)
        return self.lm_head(self.post(pooled))
`;
  } else {
    const _bad: never = nodeType;
    throw new Error(`unsupported alternative arch node type: ${_bad}`);
  }

  return `# === ${title} (${nodeType}) ===
${readme}
# Self-contained token LM bundle (matches comfy_research/engine). Last-token logits [B, vocab].
import torch
import torch.nn as nn
import torch.nn.functional as F

${CAUSAL_DEPTHWISE_CONV}
${extraClasses}


def fn_${pySym}_model() -> ${className}:
    torch.manual_seed(${seed})
    return ${className}()
`;
}
