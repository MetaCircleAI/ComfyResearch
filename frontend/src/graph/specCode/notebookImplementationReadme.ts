/** Long-form # comments prepended to generated notebook / “open spec” Python for readability. */

import type { ArchLmKind } from "../../components/nodes/alternativeArchModelDefaults";

export type AltArchReadmeParams = {
  vocab: number;
  modelDim: number;
  contextLength: number;
  numHeads?: number;
  causal?: boolean;
  numLayers?: number;
  depth?: number;
  convKernel?: number;
  ffMult?: number;
  numSlots?: number;
  slotIters?: number;
  localMixKernel: number;
};

const ENGINE: Record<ArchLmKind, string> = {
  linear_attention_model: "comfy_research/engine/linear_attention_model.py (LinearAttentionTokenPredictBundle)",
  diagonal_ssm_token_model: "comfy_research/engine/diagonal_ssm_token_model.py (DiagonalSsmTokenPredictBundle)",
  rwkv_time_mix_token_model: "comfy_research/engine/rwkv_time_mix_token_model.py (RwkvTimeMixTokenPredictBundle)",
  hyena_like_conv_model: "comfy_research/engine/hyena_like_conv_model.py (HyenaLikeConvTokenPredictBundle)",
  slot_attention_token_model: "comfy_research/engine/slot_attention_token_model.py (SlotAttentionTokenPredictBundle)",
};

export function readmeAlternativeArchTokenLm(kind: ArchLmKind, p: AltArchReadmeParams): string {
  const lk = p.localMixKernel;
  const mix =
    lk >= 3
      ? `# Optional local mixing: CausalDepthwiseConv1d (odd kernel ${lk}) adds past-only per-channel blur before the main block.`
      : `# Optional local mixing is off (local_mixing_kernel < 3). Same field as on attention_only_model.`;
  const head = [
    `# `,
    `# ${"─".repeat(70)}`,
    `# Token LM bundle: ${kind}`,
    `# `,
    `# Server twin: ${ENGINE[kind]}`,
    `# Trainer wires dataset → this module with cross_entropy_loss on last-token logits.`,
    `# `,
    `# Hyperparameters: vocab_size=${p.vocab}, model_dim=${p.modelDim}, context_length=${p.contextLength}.`,
    mix,
    `# `,
    `# Common layout: Embedding(V→D) → [optional local conv] → sequence mixer → Linear(D→V) on LAST timestep only.`,
    `# Input: LongTensor token_ids with shape [batch, L] and L fixed to context_length.`,
    `# Output: FloatTensor logits with shape [batch, V] (one next-token distribution for position L-1).`,
    `# `,
  ];
  const body: Record<ArchLmKind, string[]> = {
    linear_attention_model: [
      `# --- How linear attention differs from standard attention ---`,
      `# Standard attention materializes an L×L score matrix per head (expensive when L is large).`,
      `# Here φ(q)=ELU(q)+1 and φ(k)=ELU(k)+1 are positive; causal mixing uses cumulative sums so each`,
      `# position only attends to past keys/values without forming the full softmax matrix.`,
      `# Still uses learned Q,K,V,O linear maps like a transformer block, but aggregation is linearized.`,
      `# `,
    ],
    diagonal_ssm_token_model: [
      `# --- Diagonal SSM core ---`,
      `# Each timestep t gets state h_t ∈ R^D updated as h_t = exp(A_t) ⊙ h_{t-1} + B_t ⊙ x_t where`,
      `# A_t, B_t come from linear projections of the current input x_t (input-dependent diagonal).`,
      `# Several such cores are stacked with residuals; LayerNorm + LM head on the final sequence.`,
      `# `,
    ],
    rwkv_time_mix_token_model: [
      `# --- RWKV-lite time-mix + channel-mix ---`,
      `# For each timestep, a small hidden state h mixes past with projected keys/values; decay and`,
      `# “receptance” gates control how much history vs fresh input is kept (pure PyTorch loop over T).`,
      `# A gated FFN (SiLU + GELU) follows, similar in spirit to RWKV channel-mix blocks.`,
      `# `,
    ],
    hyena_like_conv_model: [
      `# --- Hyena-like convolutional mixer ---`,
      `# Each block: LayerNorm → residual causal DEPTHWISE conv along time (per-channel, past-only) →`,
      `# LayerNorm → gated pointwise MLP (GELU) → residual. Repeats “depth” times, then LM head.`,
      `# This is a convolution-first sequence model alternative to attention (long kernels = wide receptive field).`,
      `# `,
    ],
    slot_attention_token_model: [
      `# --- Slot attention readout ---`,
      `# Learned slot vectors iteratively attend to ALL token positions (softmax over L), GRU updates,`,
      `# small MLP; after “slot_iters” rounds you get K slot tensors. We mean-pool slots → one vector →`,
      `# post MLP → vocab logits (set-level prediction: whole sequence summarized, not per-step logits).`,
      `# `,
    ],
  };
  const tail: string[] = [];
  if (kind === "linear_attention_model") {
    tail.push(
      `# Causal flag: ${p.causal ? "past-only (causal)" : "bidirectional (full sequence)"}; num_heads=${p.numHeads ?? "?"}.`,
    );
  }
  if (kind === "diagonal_ssm_token_model") {
    tail.push(`# num_layers=${p.numLayers ?? "?"}.`);
  }
  if (kind === "rwkv_time_mix_token_model" || kind === "hyena_like_conv_model") {
    tail.push(
      `# depth=${p.depth ?? "?"}` +
        (kind === "hyena_like_conv_model"
          ? `, conv_kernel≈${p.convKernel ?? "?"}, ff_mult=${p.ffMult ?? "?"}`
          : "") +
        `.`,
    );
  }
  if (kind === "slot_attention_token_model") {
    tail.push(`# num_slots=${p.numSlots ?? "?"}, slot_iters=${p.slotIters ?? "?"}.`);
  }
  tail.push(`# `);
  return [...head, ...body[kind], ...tail].join("\n");
}

export function readmeDiffusionScore(dataDim: number, hidden: number, depth: number, te: number, tmax: number): string {
  return [
    `# `,
    `# ${"─".repeat(70)}`,
    `# Diffusion score network ε_θ(x_t, t)`,
    `# `,
    `# Server twin: comfy_research/engine/diffusion_score_model.py (DiffusionScoreMLP + ddpm_schedule_*).`,
    `# Training samples discrete timestep t, Gaussian noise ε, forms noisy x_t, then minimizes MSE(ε̂, ε).`,
    `# `,
    `# Shapes: x_noisy is Float[B, data_dim=${dataDim}]; t is Long[B] with values in [0, T-1], T=${tmax}.`,
    `# Output ε̂ has the same shape as x_noisy. Time is sinusoidally embedded to width ${te}, projected`,
    `# to hidden_dim=${hidden}, concatenated with x_noisy, then an MLP of depth=${depth} (SiLU activations).`,
    `# `,
    `# This cell is the score MLP only; the trainer owns noise sampling, schedules, and the diffusion_mse_loss node.`,
    `# `,
  ].join("\n");
}

export function readmeAttentionOnlyCore(embedDim: number, contextLength: number, numHeads: number): string {
  return [
    `# `,
    `# ${"─".repeat(70)}`,
    `# Attention-only core (multi-head softmax self-attention on a fixed-length sequence)`,
    `# `,
    `# Server twin: comfy_research/engine/attention_only_model.py (AttentionOnlyModel).`,
    `# On the canvas, “token CE” training wraps this block with Embedding + optional local conv + LM head;`,
    `# this cell is the inner mixer only: float tensor in [B, L, D] → same shape out.`,
    `# `,
    `# Shapes: batch B, length L = context_length (${contextLength}), model width D = embed_dim (${embedDim}),`,
    `# heads H = num_heads (${numHeads}), head_dim = D // H.`,
    `# `,
    `# Forward steps:`,
    `#   1) Linear projections produce Q, K, V per head (each [B, H, L, head_dim]).`,
    `#   2) Attention scores = (Q @ K^T) * scale; apply causal mask if causal_attention=yes.`,
    `#   3) Softmax over keys → mix values; merge heads; output projection w_o.`,
    `# `,
  ].join("\n");
}

export function readmeTransformerToken(vocab: number, dim: number, layers: number, ctx: number): string {
  return [
    `# `,
    `# ${"─".repeat(70)}`,
    `# Transformer token LM (decoder-style stack + LM head)`,
    `# `,
    `# Server twin: comfy_research/engine (TransformerTokenPredictBundle / trainer_run token branch).`,
    `# `,
    `# Shapes: token ids [B, L] with L=${ctx}; last-step logits [B, vocab=${vocab}]; model_dim=${dim};`,
    `# num_layers=${layers} blocks (self-attn + FFN each, with causal masking).`,
    `# `,
    `# Each layer: LayerNorm → multi-head causal self-attention (residual) → LayerNorm → feed-forward (residual).`,
    `# Tie/un-tie embedding vs LM head matches your node’s tieEmbeddingLmHead setting.`,
    `# `,
  ].join("\n");
}

export function readmeMlpToken(nodeLabel: string, vocab: number, embed: number, tokensPerInput: number): string {
  return [
    `# `,
    `# ${"─".repeat(70)}`,
    `# ${nodeLabel}`,
    `# `,
    `# Server twin: comfy_research/engine (MLP token bundles in trainer_run).`,
    `# `,
    `# Idea: embed each of L=${tokensPerInput} tokens into D=${embed}, flatten to a single wide vector,`,
    `# run a deep MLP, then map back to logits over vocab V=${vocab} (often last-token CE).`,
    `# `,
  ].join("\n");
}

export function readmeMlpVector(inDim: number, outDim: number, depth: number): string {
  return [
    `# `,
    `# ${"─".repeat(70)}`,
    `# Vector MLP regressor / classifier`,
    `# `,
    `# Server twin: comfy_research/engine (mlp stack for tabular / vector datasets).`,
    `# `,
    `# Maps x ∈ R^${inDim} → y ∈ R^${outDim} through depth=${depth} hidden layers (see width in class __init__).`,
    `# `,
  ].join("\n");
}
