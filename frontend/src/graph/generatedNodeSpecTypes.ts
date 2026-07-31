import { buildInformationBottleneckDatasetLoaders, buildLinearLikeDatasetTorch, buildPdeFieldDatasetLoaders, buildToyLanguageDatasetLoaders } from "./extraDatasetCodegen";
import { buildVisionDatasetTorch } from "./visionDatasetDefinitionCode";
import {
  buildBigramLowRankDatasetLoaders,
  buildCircleRandomWalkDatasetLoaders,
  buildCircularMotionDatasetLoaders,
  buildInContextAssociativeRecallDatasetLoaders,
  buildKepler2dDatasetLoaders,
  buildModularAdditionDatasetTorch,
  buildSymbolicFuncDatasetLoaders,
  buildTokenPredictionDatasetLoaders,
  buildUniformLinearMotionDatasetLoaders,
  buildUnigramDatasetTorch,
} from "./extraDatasetCodegen";
import {
  buildDatasetMixerBPlaceholder,
  buildDatasetMixerLoaders,
  buildInputSamplerLoaders,
  buildRandomInputDistributionLoaders,
  buildTeacherDatasetLoaders,
  crlServerSideStub,
} from "./extraDatasetCodegen";
import type { CodegenContext } from "./codegenContext";
import { buildAttentionOnlyTorch, buildMlpTokenTorch, buildMlpTorch, buildTransformerTokenTorch } from "./coreModelCodegen";
import { buildAfnoLiteSpatiotemporalModelTorch, buildKanModelTorch, buildMppSpatiotemporalModelTorch, buildNumericHyenaModelTorch, buildNumericTransformerModelTorch, buildTransformerMultiTokenModelTorch } from "./extraModelCodegen";
import { buildAlternativeArchTokenLmNotebookPython } from "./specCode/alternativeArchNotebookSpecCode";
import { buildGatedMlpModelTorch, buildMoeMlpModelTorch } from "./extraModelCodegen";
import { buildResnetModelNotebookPython, buildVitModelNotebookPython } from "./specCode/visionModelSpecCode";
import { buildDiffusionScoreNotebookPython } from "./specCode/diffusionScoreNotebookSpecCode";
import { buildCombinedModelTorch, buildResidualLnModelTorch } from "./extraModelCodegen";
import { buildCrossEntropyTorch, buildDiffusionMseLossNotebookPython, buildKanRegTorch, buildMseTorch } from "./lossCodegen";
import { buildMupInitializationCell, buildSaxeInitializationCell, buildSymmetrizedMlpInitCell } from "./initializationCodegen";
import { buildTrainerTorch } from "./trainerCodegen";
import { buildGraphAuxCell } from "./auxNotebookNodeCodegen";
import {
  buildAdamTorch,
  buildLrScheduleCell,
  buildMupLrScheduleCell,
  buildAdamWTorch,
  buildMuonTorch,
  buildSgdTorch,
  buildShampooTorch,
  buildSignSgdTorch,
  buildSoapTorch,
} from "./optimizerCodegen";
import {
  generateAdamOptimizerSpecCode,
  generateAdamWOptimizerSpecCode,
  generateMuonOptimizerSpecCode,
  generateSgdOptimizerSpecCode,
  generateShampooOptimizerSpecCode,
  generateSignSgdOptimizerSpecCode,
  generateSoapOptimizerSpecCode,
} from "./specCode/optimizerSpecCode";
import {
  buildAbsolutePosEmbedLayerModule,
  buildCausalMaskLayerModule,
  buildEinsumLayerModule,
  buildFlattenLayerModule,
  buildReshapeLayerModule,
  buildSoftmaxLayerModule,
  buildActivationLayerModule,
  buildAfnoEncoderBlockLayerModule,
  buildAfnoPatchDecodeLayerModule,
  buildAfnoPatchEmbedLayerModule,
  buildAfnoSpectralMixerLayerModule,
  buildEmbeddingLayerModule,
  buildLayerNormLayerModule,
  buildLinearLayerModule,
  buildLocalMixingLayerModule,
  buildRmsNormLayerModule,
  buildRotaryEmbedLayerModule,
  buildUnembeddingLayerModule,
} from "./layerModulesCodegen";
// type-only imports — no runtime cycle with the generated file
import type { FieldDef, NodeCodegenArgs, NodeFamily, NodeRegistrySpec, NodeUiSpec } from "./nodeRegistrySpec";

/** Shape of one entry in the checked-in generated file (comfy_research/nodes/generate.py). */
export type GeneratedNodeSpec = {
  label: string;
  category: string;
  hint?: string;
  family?: readonly NodeFamily[];
  /** Plain default-data object; specFromGenerated wraps it in a fresh-copy closure. */
  defaults?: Record<string, unknown>;
  fields?: readonly FieldDef[];
  observable?: {
    infoMarkdown?: string;
    vizTitle?: string;
    vizVariant?: string;
    spawnsVizNode?: boolean;
  };
  /** Exact per-node defaultData recipe used when spawning. */
  spawn?: {
    kind: "hessian_topk" | "user_scalar" | "neuron_trajectory" | "scalar_series" | "embedding_trajectory" | "attention_map";
    /** user_scalar: spawn viz header name when it differs from vizTitle. */
    title?: string;
    fixedTopK?: number;
    topKFromField?: string;
    /** user_scalar 三态:omitted → builder 不收第四参,vizYAxisLabel 键不存在。 */
    unit?: string;
    /** hessian_topk:源 data[key]==="ascending" 才升序,其余降序。 */
    orderFromField?: string;
    /** hessian_topk:附加 seriesLabels 数组(activation_stats 的 mean/std)。 */
    seriesLabels?: string[];
    /** user_scalar/embedding_trajectory:data[key] trim 为标题,空/缺回退 title ?? vizTitle。 */
    titleFromField?: string;
  };
  /** Frontend adapters only — never schema facts (core rule). */
  frontend?: {
    /** Resolved via GENERATED_COMPONENT_ADAPTERS (nodeRegistry.ts). */
    componentKey?: string;
    /** Type-only metadata; the codegen path is not wired. */
    codegenKey?: string;
    /** SchemaNode show-code 面板;经 SPEC_CODE_ADAPTERS 解析。 */
    specCodeKey?: string;
  };
  /** SchemaNode 泛型 UI 块(accent/socketRows/codeKind/info)。 */
  ui?: NodeUiSpec;
  /** 声明式入口端口。声明即 return-style 全权接管该 target 的连接判定。
   * ioMode 仅在 readNodeCanvasIoMode(source.data) 等于声明值时接受。 */
  ports?: {
    in: ReadonlyArray<{
      id: string;
      accepts: ReadonlyArray<{
        family?: string;
        type?: string;
        ioMode?: "model" | "input-output";
        handles: readonly string[];
      }>;
    }>;
  };
  /** false → NodeSpec.resize=false(buildNodeTypesFromRegistry 不包 withNodeResize)。 */
  resizable?: boolean;
};

/** codegenKey → codegen body(TS-only truth;Python 只发 key)。Custom exporter 的
 * adapter 条目=在此加一行,无 schema facts。 */
export const CODEGEN_ADAPTERS: Record<string, (args: NodeCodegenArgs) => string> = {
  observable_weight_l2: ({ pySym, title }) => `# === ${title} (observable_weight_l2) ===
# Matches ComfyResearch /api train logging: √(Σ w²) over all parameters.
import torch


def fn_${pySym}_weight_l2_norm(module: torch.nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().float().pow(2).sum().item())
    return s**0.5
`,
  // dataset codegen adapters:函数本体留在原文件(组件仍直调),这里只做
  // codegenKey → builder 的 1 行包装,无 schema facts。
  advection_dataset: ({ pySym, title, data }) =>
    buildPdeFieldDatasetLoaders(pySym, title, "advection_dataset", data),
  diffusion_pde_dataset: ({ pySym, title, data }) =>
    buildPdeFieldDatasetLoaders(pySym, title, "diffusion_pde_dataset", data),
  reaction_diffusion_dataset: ({ pySym, title, data }) =>
    buildPdeFieldDatasetLoaders(pySym, title, "reaction_diffusion_dataset", data),
  linear_dataset: ({ pySym, title, data }) => buildLinearLikeDatasetTorch(pySym, title, "linear_dataset", data),
  random_noise_dataset: ({ pySym, title, data }) => buildLinearLikeDatasetTorch(pySym, title, "random_noise_dataset", data),
  memorization_a_dataset: ({ pySym, title, data }) => buildLinearLikeDatasetTorch(pySym, title, "memorization_a_dataset", data),
  memorization_b_dataset: ({ pySym, title, data }) => buildLinearLikeDatasetTorch(pySym, title, "memorization_b_dataset", data),
  biography_lm_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("biography_lm_dataset", pySym, title, data),
  cogs_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("cogs_dataset", pySym, title, data),
  dyck_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("dyck_dataset", pySym, title, data),
  formal_language_suite_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("formal_language_suite_dataset", pySym, title, data),
  listops_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("listops_dataset", pySym, title, data),
  multi_hop_fact_chain_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("multi_hop_fact_chain_dataset", pySym, title, data),
  ngram_language_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("ngram_language_dataset", pySym, title, data),
  pcfg_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("pcfg_dataset", pySym, title, data),
  phi1_style_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("phi1_style_dataset", pySym, title, data),
  relation_tuple_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("relation_tuple_dataset", pySym, title, data),
  scan_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("scan_dataset", pySym, title, data),
  synthetic_playground_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("synthetic_playground_dataset", pySym, title, data),
  tinystories_dataset: ({ pySym, title, data }) => buildToyLanguageDatasetLoaders("tinystories_dataset", pySym, title, data),
  mnist_dataset: ({ pySym, title, data }) => buildVisionDatasetTorch(pySym, title, "mnist_dataset", data),
  cifar10_dataset: ({ pySym, title, data }) => buildVisionDatasetTorch(pySym, title, "cifar10_dataset", data),
  gaussian_blob_dataset: ({ pySym, title, data }) => buildVisionDatasetTorch(pySym, title, "gaussian_blob_dataset", data),
  shape_world_dataset: ({ pySym, title, data }) => buildVisionDatasetTorch(pySym, title, "shape_world_dataset", data),
  hole_counting_dataset: ({ pySym, title, data }) => buildVisionDatasetTorch(pySym, title, "hole_counting_dataset", data),
  token_prediction_dataset: ({ pySym, title, data }) => buildTokenPredictionDatasetLoaders(pySym, title, data),
  bigram_low_rank_dataset: ({ pySym, title, data }) => buildBigramLowRankDatasetLoaders(pySym, title, data),
  circle_random_walk_dataset: ({ pySym, title, data }) => buildCircleRandomWalkDatasetLoaders(pySym, title, data),
  circular_motion_dataset: ({ pySym, title, data }) => buildCircularMotionDatasetLoaders(pySym, title, data),
  in_context_associative_recall_dataset: ({ pySym, title, data }) => buildInContextAssociativeRecallDatasetLoaders(pySym, title, data),
  modular_addition_dataset: ({ pySym, title, data }) => buildModularAdditionDatasetTorch(pySym, title, data),
  unigram_dataset: ({ pySym, title, data }) => buildUnigramDatasetTorch(pySym, title, data),
  kepler_2d_dataset: ({ pySym, title, data }) => buildKepler2dDatasetLoaders(pySym, title, data),
  uniform_linear_motion_dataset: ({ pySym, title, data }) => buildUniformLinearMotionDatasetLoaders(pySym, title, data),
  symbolic_func_dataset: ({ pySym, title, data }) => buildSymbolicFuncDatasetLoaders(pySym, title, data),
  information_bottleneck_dataset: ({ pySym, title, data }) => buildInformationBottleneckDatasetLoaders(pySym, title, data),
  crl_env_config: ({ pySym, title, nodeType }) => crlServerSideStub(pySym, title, nodeType),
  teacher_dataset: ({ pySym, title, data }) => buildTeacherDatasetLoaders(pySym, title, data),
  input_sampler: ({ pySym, title, data, ctx, nodeId }) => buildInputSamplerLoaders(pySym, title, data, ctx as CodegenContext | undefined, nodeId),
  random_input_distribution: ({ pySym, title, data }) => buildRandomInputDistributionLoaders(pySym, title, data),
  dataset_mixer: ({ pySym, title, data, ctx, nodeId }) => buildDatasetMixerLoaders(pySym, title, data, ctx as CodegenContext | undefined, nodeId),
  dataset_mixer_b: ({ pySym, title }) => buildDatasetMixerBPlaceholder(pySym, title),
  mlp_model: ({ pySym, title, data }) => buildMlpTorch(pySym, title, data),
  gated_mlp_model: ({ pySym, title, data }) => buildGatedMlpModelTorch(pySym, title, data),
  moe_mlp_model: ({ pySym, title, data }) => buildMoeMlpModelTorch(pySym, title, data),
  mlp_token_model: ({ pySym, title, data }) => buildMlpTokenTorch(pySym, title, "mlp_token_model", data),
  gated_mlp_token_model: ({ pySym, title, data }) => buildMlpTokenTorch(pySym, title, "gated_mlp_token_model", data),
  moe_mlp_token_model: ({ pySym, title, data }) => buildMlpTokenTorch(pySym, title, "moe_mlp_token_model", data),
  attention_only_model: ({ pySym, title, data }) => buildAttentionOnlyTorch(pySym, title, data),
  linear_attention_model: ({ pySym, title, data }) => buildAlternativeArchTokenLmNotebookPython(pySym, title, "linear_attention_model", data),
  diagonal_ssm_token_model: ({ pySym, title, data }) => buildAlternativeArchTokenLmNotebookPython(pySym, title, "diagonal_ssm_token_model", data),
  rwkv_time_mix_token_model: ({ pySym, title, data }) => buildAlternativeArchTokenLmNotebookPython(pySym, title, "rwkv_time_mix_token_model", data),
  hyena_like_conv_model: ({ pySym, title, data }) => buildAlternativeArchTokenLmNotebookPython(pySym, title, "hyena_like_conv_model", data),
  slot_attention_token_model: ({ pySym, title, data }) => buildAlternativeArchTokenLmNotebookPython(pySym, title, "slot_attention_token_model", data),
  transformer_token_model: ({ pySym, title, data }) => buildTransformerTokenTorch(pySym, title, data),
  transformer_multi_token_model: ({ pySym, title, data }) => buildTransformerMultiTokenModelTorch(pySym, title, data),
  numeric_transformer_model: ({ pySym, title, data }) => buildNumericTransformerModelTorch(pySym, title, data),
  numeric_hyena_model: ({ pySym, title, data }) => buildNumericHyenaModelTorch(pySym, title, data),
  mpp_spatiotemporal_model: ({ pySym, title, data }) => buildMppSpatiotemporalModelTorch(pySym, title, data),
  afno_lite_spatiotemporal_model: ({ pySym, title, data }) => buildAfnoLiteSpatiotemporalModelTorch(pySym, title, data),
  kan_model: ({ pySym, title, data }) => buildKanModelTorch(pySym, title, data),
  resnet_model: ({ pySym, title, data }) => buildResnetModelNotebookPython(pySym, title, data),
  vit_model: ({ pySym, title, data }) => buildVitModelNotebookPython(pySym, title, data),
  diffusion_score_model: ({ pySym, title, data }) => buildDiffusionScoreNotebookPython(pySym, title, data),
  residual_ln_model: ({ pySym, title, data }) => buildResidualLnModelTorch(pySym, title, data),
  crl_residual_mlp: ({ pySym, title, nodeType }) => crlServerSideStub(pySym, title, nodeType),
  combined_model: ({ pySym, title, data }) => buildCombinedModelTorch(pySym, title, data),
  linear_layer: ({ pySym, title, data }) => buildLinearLayerModule(pySym, title, data),
  activation_layer: ({ pySym, title, data }) => buildActivationLayerModule(pySym, title, data),
  layer_norm_layer: ({ pySym, title, data }) => buildLayerNormLayerModule(pySym, title, data),
  rms_norm_layer: ({ pySym, title, data }) => buildRmsNormLayerModule(pySym, title, data),
  embedding_layer: ({ pySym, title, data }) => buildEmbeddingLayerModule(pySym, title, data),
  unembedding_layer: ({ pySym, title, data }) => buildUnembeddingLayerModule(pySym, title, data),
  absolute_pos_embed_layer: ({ pySym, title, data }) => buildAbsolutePosEmbedLayerModule(pySym, title, data),
  rotary_embed_layer: ({ pySym, title, data }) => buildRotaryEmbedLayerModule(pySym, title, data),
  local_mixing_layer: ({ pySym, title, data }) => buildLocalMixingLayerModule(pySym, title, data),
  afno_patch_embed_layer: ({ pySym, title, data }) => buildAfnoPatchEmbedLayerModule(pySym, title, data),
  afno_spectral_mixer_layer: ({ pySym, title, data }) => buildAfnoSpectralMixerLayerModule(pySym, title, data),
  afno_encoder_block_layer: ({ pySym, title, data }) => buildAfnoEncoderBlockLayerModule(pySym, title, data),
  afno_patch_decode_layer: ({ pySym, title, data }) => buildAfnoPatchDecodeLayerModule(pySym, title, data),
  causal_mask: ({ pySym, title, data }) => buildCausalMaskLayerModule(pySym, title, data),
  einsum: ({ pySym, title, data }) => buildEinsumLayerModule(pySym, title, data),
  flatten: ({ pySym, title, data }) => buildFlattenLayerModule(pySym, title, data),
  reshape: ({ pySym, title, data }) => buildReshapeLayerModule(pySym, title, data),
  softmax: ({ pySym, title, data }) => buildSoftmaxLayerModule(pySym, title, data),
  adam_optimizer: ({ pySym, title, data }) => buildAdamTorch(pySym, title, data),
  adamw_optimizer: ({ pySym, title, data }) => buildAdamWTorch(pySym, title, data),
  sgd_optimizer: ({ pySym, title, data }) => buildSgdTorch(pySym, title, data),
  signsgd_optimizer: ({ pySym, title, data }) => buildSignSgdTorch(pySym, title, data),
  muon_optimizer: ({ pySym, title, data }) => buildMuonTorch(pySym, title, data),
  shampoo_optimizer: ({ pySym, title, data }) => buildShampooTorch(pySym, title, data),
  soap_optimizer: ({ pySym, title, data }) => buildSoapTorch(pySym, title, data),
  lr_schedule: ({ pySym, title, data }) => buildLrScheduleCell(pySym, title, data),
  mup_lr_schedule: ({ pySym, title, data }) => buildMupLrScheduleCell(pySym, title, data),
  mse_loss: ({ pySym, title, data }) => buildMseTorch(pySym, title, data),
  cross_entropy_loss: ({ pySym, title, data }) => buildCrossEntropyTorch(pySym, title, data),
  diffusion_mse_loss: ({ pySym, title }) => buildDiffusionMseLossNotebookPython(pySym, title),
  kan_reg: ({ pySym, title }) => buildKanRegTorch(pySym, title),
  mup_initialization: ({ pySym, title }) => buildMupInitializationCell(pySym, title),
  saxe_initialization: ({ pySym, title }) => buildSaxeInitializationCell(pySym, title),
  symmetrized_mlp_init: ({ pySym, title }) => buildSymmetrizedMlpInitCell(pySym, title),
  rank_aligned_initialization: ({ pySym, title }) =>
    buildGraphAuxCell(
      pySym,
      title,
      "rank_aligned_initialization",
      'Server-side: POST /api/train builds a balanced F^T F = W W^T factorization on the wired model '
        + 'when this node is connected to a model initialization socket (Rank Collapse paper, Appendix B).',
    ),
  paper_classification_dataset: ({ pySym, title, data }) =>
    buildLinearLikeDatasetTorch(pySym, title, "paper_classification_dataset", data),
  tinyshakespeare_lm_dataset: ({ pySym, title, data }) =>
    buildToyLanguageDatasetLoaders("tinyshakespeare_lm_dataset", pySym, title, data),
  observable_representation_rankme: ({ pySym, title }) =>
    buildGraphAuxCell(
      pySym,
      title,
      "observable_representation_rankme",
      'Server-side observable: exponential entropy effective rank of a selected representation centered '
        + 'covariance (RankMe, Roy & Vetterli 2007), plus the two leading covariance eigenvalues.',
    ),
  observable_representation_alpha_req: ({ pySym, title }) =>
    buildGraphAuxCell(
      pySym,
      title,
      "observable_representation_alpha_req",
      'Server-side observable: power-law decay exponent of a selected representation covariance spectrum '
        + '(alphaReQ). Larger alpha means more concentrated variance.',
    ),
  trainer: ({ pySym, title, nodeId, data, ctx }) => buildTrainerTorch(pySym, title, nodeId, data, ctx as never),
  crl_trainer: ({ pySym, title, nodeType }) => crlServerSideStub(pySym, title, nodeType),
  // aux 文档 cell(note 逐字保留自旧 NODE_SPEC_OVERRIDES;codegen golden 兜恒等)。
  activation: ({ pySym, title, nodeType }) =>
    buildGraphAuxCell(
      pySym,
      title,
      nodeType,
      "Collects forward tensors from a wired model handle during /api/train (see comfy_research/engine/activation_collect.py).",
    ),
};

/** specCode fn 表(CODEGEN_ADAPTERS 同构;fn 本体留 TS,不持 schema 真相)。 */
const SPEC_CODE_ADAPTERS: Record<string, (data: Record<string, unknown>) => string> = {
  adam_optimizer: (d) => generateAdamOptimizerSpecCode(d as never),
  adamw_optimizer: (d) => generateAdamWOptimizerSpecCode(d as never),
  sgd_optimizer: (d) => generateSgdOptimizerSpecCode(d as never),
  signsgd_optimizer: (d) => generateSignSgdOptimizerSpecCode(d as never),
  muon_optimizer: (d) => generateMuonOptimizerSpecCode(d as never),
  shampoo_optimizer: (d) => generateShampooOptimizerSpecCode(d as never),
  soap_optimizer: (d) => generateSoapOptimizerSpecCode(d as never),
};

function resolveGeneratedSpecCode(type: string, specCodeKey: string | undefined) {
  if (specCodeKey == null || specCodeKey === "") return {};
  const fn = SPEC_CODE_ADAPTERS[specCodeKey];
  if (!fn) {
    throw new Error(
      `generated node "${type}" declares specCodeKey "${specCodeKey}" but SPEC_CODE_ADAPTERS has no such entry — add the adapter line in generatedNodeSpecTypes.ts`,
    );
  }
  return { specCode: fn };
}

function resolveGeneratedCodegen(type: string, codegenKey: string | undefined) {
  if (codegenKey == null || codegenKey === "") return {};
  const fn = CODEGEN_ADAPTERS[codegenKey];
  if (!fn) {
    throw new Error(
      `generated node "${type}" declares codegenKey "${codegenKey}" but CODEGEN_ADAPTERS has no such entry — add the adapter line in generatedNodeSpecTypes.ts`,
    );
  }
  return { codegen: fn };
}

/** spawn defaults 深拷贝——legacy defaults 含数组/嵌套对象
 * (regions: []、activationTensorCaches: {}),浅拷贝会跨 spawn 共享引用。 */
function cloneDefaults(defaults: Record<string, unknown>): Record<string, unknown> {
  return structuredClone(defaults) as Record<string, unknown>;
}

export function specFromGenerated(type: string, g: GeneratedNodeSpec): NodeRegistrySpec {
  return {
    type,
    label: g.label,
    category: g.category,
    hint: g.hint,
    // defaults 条件发射:GeneratedNodeSpec 无 defaults 键 = legacy
    // hasDefaults:false 语义(nodeRegistryDefaults 返回 undefined,不是 {})。
    ...(g.defaults !== undefined ? { defaults: () => cloneDefaults(g.defaults!) } : {}),
    ...(g.fields ? { fields: g.fields } : {}),
    ...(g.observable ? { observable: g.observable as NodeRegistrySpec["observable"] } : {}),
    ...(g.ui ? { ui: g.ui } : {}),
    // resize 透传:否则 buildNodeTypesFromRegistry 会给
    // graph_assist_failure_overlay 包 withNodeResize。
    ...(g.resizable === false ? { resize: false } : {}),
    ...resolveGeneratedCodegen(type, g.frontend?.codegenKey),
    ...resolveGeneratedSpecCode(type, g.frontend?.specCodeKey),
    family: g.family ?? (g.category === "observables" ? ["observable"] : undefined),
  } as NodeRegistrySpec;
}
