/** shape 传播/上游张量解析的**功能支持子集**（并非与 family 恒等——
 * 这是能力边界,不是分类;支持面扩展属功能开发,不属重构)。
 *
 * 有意保留的能力差异：
 * - fakeTensorShapePropagation 的 atomic 集曾漏 rms_norm_layer(shape 规则
 *   在 558 行早已存在）；纳入集合后，rms_norm 的 shape 传播恢复工作，并与
 *   resolveUpstreamTensor 使用同一集合。
 * - afno ×4 层无 shape 规则实现 → 显式排除(实现后从例外移除即可)。
 * - FULL_MODEL 支持 16 型(28 型 canvas 全集的子集)——意图关系 guard 见
 *   canvasShapeSupport.seam.test.ts。
 */
import { nodeRegistryTypesWithFamily } from "./nodeRegistrySpec";

export const AFNO_SHAPE_UNSUPPORTED: ReadonlySet<string> = new Set([
  "afno_patch_embed_layer",
  "afno_spectral_mixer_layer",
  "afno_encoder_block_layer",
  "afno_patch_decode_layer",
]);

export const SHAPE_ATOMIC_LAYER_TYPES: ReadonlySet<string> = new Set(
  nodeRegistryTypesWithFamily("atomic_layer_model").filter((t) => !AFNO_SHAPE_UNSUPPORTED.has(t)),
);

export const SHAPE_FULL_MODEL_TYPES: ReadonlySet<string> = new Set([
  "mlp_model",
  "gated_mlp_model",
  "moe_mlp_model",
  "kan_model",
  "residual_ln_model",
  "attention_only_model",
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
  "diffusion_score_model",
  "numeric_transformer_model",
  "numeric_hyena_model",
  "transformer_token_model",
  "transformer_multi_token_model",
]);
