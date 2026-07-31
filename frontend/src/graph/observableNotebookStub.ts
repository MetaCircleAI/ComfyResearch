/**
 * Python-oriented documentation shown when the user opens "observable note" in the Code notebook.
 * Mirrors `comfy_research/engine/trainer_run.py` (inner `record()` and helpers ~L1207+); not executed by the trainer.
 */

const TRAINER_SOURCE = `#
# Primary implementation: comfy_research/engine/trainer_run.py
#   - Helpers: _weight_l2_norm, _gradient_l2_norm_*, _activation_mean_std_bucketed, _softmax_attention_probs_* ...
#   - Each log step: inner function record(step, train_val, x_log=..., y_log=...) appends to observable_metric_histories.
# Constants: HESSIAN_PARAM_LIMIT=2048, HESSIAN_FORCE_MAX_PARAMS=50_000_000
#`;

function banner(title: string, graphNodeType: string): string {
  const t = (title || graphNodeType).replace(/\r?\n/g, " ").trim() || graphNodeType;
  return `# Observable implementation notes - ${t}
# graph_node_type: ${graphNodeType}
${TRAINER_SOURCE}
`;
}

/** Trainer-side logic for one observable `type` string (same as canvas node type). */
function trainerImplementationDoc(kind: string): string {
  const docs: Record<string, string> = {
    observable_weight_l2: `# --- observable_weight_l2 ---
def _weight_l2_norm(module: torch.nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().float().pow(2).sum().item())
    return s ** 0.5

# Logged each record(): global series on node id; normAggregation "top_level_module" / "tensor" mirrors gradient-norm key layout (weights).
`,

    observable_weight_l1: `# --- observable_weight_l1 ---
def _weight_l1_norm(module: torch.nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().float().abs().sum().item())
    return s

# Logged: append(_weight_l1_norm(model))
`,

    observable_capacity: `# --- observable_capacity (bits-style capacity from CE loss) ---
# Uses logging batch yr (targets) to infer vocab_size = max(yr)+1 for integer labels.
# train_val is the current mean train loss (cross-entropy nats) at this log step.
# c_bits = (log(vocab_size) - train_val) * batch_size / log(2)
# If vocab_size <= 1 -> NaN
`,

    observable_hessian_eigenvalues: `# --- observable_hessian_eigenvalues ---
# _hessian_loss_eigenvalues(model, criterion, xr, yr, loss_scale, trainer_task, top_k, order, kan_regs, max_params)
# Builds exact Hessian of scalar loss w.r.t. all trainable params (O(P^2) memory), eigvalsh, take top_k.
# Node data: topK (default 5), order ascending|descending.
# If hessian_oversized_mode == "skip": NaNs. Else max_params is HESSIAN_PARAM_LIMIT or HESSIAN_FORCE_MAX_PARAMS when forced.
# Stores primary series on node id; rank i also mirrored to key "{node_id}::{i}" for multi-line viz.
`,

    observable_user: `# --- observable_user ---
# comfy_research/engine/observable_user_eval.py - eval_observable_user_mean_abs(...)
# Resolves definition_code from node data or saved user observable; evaluates your tensor path on the logging batch.
# Logged scalar is implementation-defined (typically a mean |·| style summary); runs under torch.no_grad().
`,

    observable_relu_nonlinear_count: `# --- observable_relu_nonlinear_count ---
# _count_nonlinear_relu_neurons(model, xr, hidden_layer_index=hidx, depth=depth)
# Expects nn.Sequential MLP: hook post-ReLU layer at index 2*hidx+1, forward on prepared batch.
# Counts hidden units where (act>0).any(batch) & (act==0).any(batch) over dim 0 ("nonlinear" ReLU units).
`,

    observable_accuracy: `# --- observable_accuracy ---
# Not used for trainer_task == diffusion_noise (NaN train + test).
# Train: pred = _forward_reg(model, xr); top1 = argmax(pred, -1); acc = mean(top1 == yr.long()).
# Test curve (if test batch exists): same on x_test_log, y_test_log; stored under key "{node_id}::test".
`,

    observable_gradient_norm: `# --- observable_gradient_norm ---
# Global: _gradient_l2_norm_global(model, normalize=gradientNormNormalized) - L2 norm of grads (None skipped).
# gradientNormNormalized default True: divide by sqrt(# scalar params with grad in that scope).
# normAggregation: "top_level_module" -> per-top keys "{node_id}::top::{segment}" (canonical list, NaN-padded vs global).
# normAggregation: "tensor" -> per-parameter grad L2 keys "{node_id}::tensor::{sanitized_name}" (capped count).
`,

    observable_train_test_gap: `# --- observable_train_test_gap ---
# When test_loss_history aligns with loss_history: append(test_loss_history[-1] - loss_history[-1]); else NaN.
`,

    observable_activation_stats: `# --- observable_activation_stats ---
# _activation_mean_std_bucketed: hooks nn.Linear / Conv*d; groups outputs by ".layers.{i}." in module name else bucket "rest".
# Per bucket: mean of module means, mean of module stds (unbiased=False each). Global = equal average across buckets.
# activationStatsLayers "all_layers" -> histories node_id::layer_mean::{seg} and ::layer_std::{seg}; viz emits paired rows per bucket.
# Primary series still bucket-average mean/std; ::std key remains global std aggregate alongside mean series id.
`,

    observable_sink_attention_mass: `# --- observable_sink_attention_mass ---
# _softmax_attention_probs_all_layers_or_none -> list[[B,H,L,L]] per encoder layer (same supported cores as last-layer helper).
# sinkAttentionMassLayers: "global" (default) -> mean of per-layer sink masses to history[node_id]; "all_layers" -> series node_id::layer::<i>.
# sinkTokenIndex (default 0). _attn_sink_mass(attn, sink_idx) = mean over B,H,L of attn[..., sink_idx].
`,

    observable_attention_entropy_mean: `# --- observable_attention_entropy_mean ---
# observableEncoderLayers "global" (default): last-layer softmax from _softmax_attention_probs_or_none; "all_layers": per-layer list + mean to primary + node_id::layer::<i>.
# attn = _softmax_attention_probs_or_none(...); if None -> NaN else _attn_entropy_mean:
#   p = attn.clamp(min=1e-9); ent = -(p * p.log()).sum(-1); return mean(ent).
`,

    observable_attention_max_weight_mean: `# --- observable_attention_max_weight_mean ---
# observableEncoderLayers: same as attention entropy (global = last layer; all_layers = per encoder layer + mean).
# Mean over batch/heads/positions of max softmax mass to any key: attn.max(dim=-1).values.mean().
`,

    observable_attention_head_sink_max: `# --- observable_attention_head_sink_max ---
# observableEncoderLayers: same as attention entropy. sinkTokenIndex for sink key.
# Per-head mean mass on sink key index; then max over heads: per_h = attn[:,:,:,si].mean((0,2)); float(per_h.max()).
`,

    observable_attention_position_bias_ratio: `# --- observable_attention_position_bias_ratio ---
# observableEncoderLayers: same as attention entropy.
# pos = attn.mean(dim=(0,1,2))  # length L vector; ratio = pos[0] / (mean(pos)+1e-12) - early vs average key mass.
`,

    observable_activation_norm_mean: `# --- observable_activation_norm_mean ---
# observableEncoderLayers "all_layers": _activation_norm_mean_and_outlier_per_bucket -> primary = mean of bucket means; series node_id::layer::<bucket>.
# _activation_norm_mean_and_outlier_ratio (global): hooks Linear/Conv; dim>=2 uses mean(||t||_2 over last dim); logs mean of per-module norms.
`,

    observable_activation_outlier_ratio: `# --- observable_activation_outlier_ratio ---
# observableEncoderLayers "all_layers": same bucket pass as norm mean; primary = mean of per-bucket outlier ratios; series node_id::layer::<bucket>.
# Global: same single forward as norm mean; logs global max|·|/mean|·|.
`,

    observable_embedding_effective_rank: `# --- observable_embedding_effective_rank ---
# observableEncoderLayers "all_layers": max effective rank per .layers.{i}. 2D weight bucket (SVD budget); primary = mean over buckets.
# "global": requires _EMBEDDING_OBSERVABLE_MODEL_TYPES + observable_numpy_arrays()["embedding"]; effective_rank_from_matrix(emb).
`,

    observable_embedding_feature_drift: `# --- observable_embedding_feature_drift ---
# observableEncoderLayers "all_layers": 1 - cosine between consecutive flattened 2D weights concatenated per .layers.{i}.; primary = mean over layers.
# "global": embedding matrix only (same as before); drift = 1 - cos(prev, cur) on flattened embedding.
`,

    observable_embedding_evolution: `# --- observable_embedding_evolution ---
# TOKEN_LM bundles only. emb, emb0 from model.observable_numpy_arrays() vs init snapshot attention_arrays_init.
# Logged: ||emb - emb0||_2 / (||emb0||_2 + 1e-12).
`,

    observable_embedding_trajectory: `# --- observable_embedding_trajectory ---
# TOKEN_LM bundles: each log appends full embedding matrix as nested Python lists into observable_embedding_histories[node_id]
# (not the scalar metric_histories dict). Frontend trajectory viz projects/plots these snapshots.
`,

    kan_reg: `# --- kan_reg (pykan MultKAN regularization scalar) ---
# _kan_reg_loss_term(model, node): lamb * model.get_reg(metric, lambL1, lambEntropy, lambCoef, lambCoefDiff)
#   metric from node data regMetric (pykan get_reg name family).
# Logged in record(): model.train(); forward with _prepare_x...(_regression_batch_for_model(model, xr)); no_grad; float(term).
# Also folded into total loss during training steps when kan_reg nodes are present (MSE + sum of reg terms).
`,

    observable_viz: `# --- observable_viz (mirror; no trainer scalar) ---
# This node type only displays histories streamed from the backend; see paired observable above for trainer math.
`,
  };

  return docs[kind] ?? `# --- ${kind} ---
# No dedicated stub text; see trainer_run.py record() and search for this node type.
`;
}

const VIZ_PANEL = `# =============================================================================
# Observable viz (frontend)
# =============================================================================
# Implementation: frontend/src/components/nodes/ObservableViz*.tsx + ObservableVizNode (data.vizVariant).
# Subscribes to trainer SSE / progress payloads and copies paired observable series into node data (histories, stepTicks).
# No PyTorch code on the canvas node itself; it only plots what the trainer already logged.
`;

export function buildObservableNotebookStub(
  graphNodeType: string,
  title: string,
  pairedTrainerObservableType?: string,
): string {
  const head = banner(title, graphNodeType);

  if (graphNodeType === "observable_viz") {
    const paired =
      pairedTrainerObservableType &&
      pairedTrainerObservableType !== "observable_viz" &&
      pairedTrainerObservableType.trim()
        ? pairedTrainerObservableType.trim()
        : null;
    const pairedBlock = paired
      ? `
# =============================================================================
# Paired trainer observable: ${paired}
# =============================================================================
${trainerImplementationDoc(paired)}
`
      : `
# Pair this viz node to a trainer observable source to see the exact trainer_run.py branch for that metric.
`;
    return `${head}
${VIZ_PANEL}
${pairedBlock}
# (Cell is documentation; trainer does not execute this file.)
pass
`;
  }

  return `${head}
${trainerImplementationDoc(graphNodeType)}

# (Cell is documentation; trainer does not execute this file.)
pass
`;
}
