import { researchNodeTypes } from "../src/components/nodeTypes";
import { NODE_REGISTRY, registeredResearchNodeTypes } from "../src/graph/nodeRegistry";
import {
  NODE_SPEC_REGISTRY,
  nodeRegistryDefaults,
  nodeRegistryHint,
  nodeRegistryTitle,
} from "../src/graph/nodeRegistrySpec";

const problems: string[] = [];

const registryKeys = Object.keys(NODE_REGISTRY).sort((a, b) => a.localeCompare(b));
const specRegistryKeys = Object.keys(NODE_SPEC_REGISTRY).sort((a, b) => a.localeCompare(b));
const nodeTypeKeys = Object.keys(researchNodeTypes).sort((a, b) => a.localeCompare(b));
const expectedDefaults: Record<string, Record<string, unknown>> = {
  observable_weight_l2: { normAggregation: "global" },
  observable_capacity: {},
  observable_hessian_eigenvalues: { topK: 5, order: "descending" },
  observable_relu_nonlinear_count: { hiddenLayerIndex: 1 },
  observable_accuracy: {},
  observable_gradient_norm: { normAggregation: "global", gradientNormNormalized: true },
  observable_train_test_gap: {},
  observable_activation_stats: {},
};
const expectedObservableVariants: Record<string, string> = {
  observable_weight_l2: "weight_l2",
  observable_capacity: "capacity",
  observable_hessian_eigenvalues: "hessian_eigenvalues",
  observable_relu_nonlinear_count: "relu_nonlinear",
  observable_accuracy: "accuracy",
  observable_gradient_norm: "gradient_norm",
  observable_train_test_gap: "user",
  observable_activation_stats: "activation_stats",
};

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return JSON.stringify(value);
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value as Record<string, unknown>).sort((a, b) => a.localeCompare(b))) {
    out[key] = (value as Record<string, unknown>)[key];
  }
  return JSON.stringify(out);
}

for (const key of registryKeys) {
  if (!(key in registeredResearchNodeTypes)) {
    problems.push(`${key}: missing from registeredResearchNodeTypes`);
  }
  if (!(key in researchNodeTypes)) {
    problems.push(`${key}: missing from researchNodeTypes`);
  }
  const expected = expectedDefaults[key];
  if (expected) {
    const title = nodeRegistryTitle(key);
    if (!title || title !== NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY].label) {
      problems.push(`${key}: nodeRegistryTitle does not match registry label`);
    }
    if (!nodeRegistryHint(key)) {
      problems.push(`${key}: registry hint is required`);
    }
    const defaults = nodeRegistryDefaults(key);
    if (!defaults || typeof defaults !== "object" || Array.isArray(defaults)) {
      problems.push(`${key}: defaults must return a plain object`);
    } else if (stableJson(defaults) !== stableJson(expected)) {
      problems.push(`${key}: defaults differ from the declared node defaults`);
    }
    const observable = NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY].observable;
    if (!observable?.infoMarkdown || !observable.vizTitle || !observable.vizVariant || observable.spawnsVizNode !== true) {
      problems.push(`${key}: observable nodes must declare observable metadata`);
    } else if (observable.vizVariant !== expectedObservableVariants[key]) {
      problems.push(`${key}: observable vizVariant changed from expected registry metadata`);
    }
  }
}

if (stableJson(registryKeys) !== stableJson(nodeTypeKeys)) {
  problems.push("researchNodeTypes keys must match NODE_REGISTRY keys exactly");
}
if (stableJson(registryKeys) !== stableJson(specRegistryKeys)) {
  problems.push("NODE_REGISTRY keys must match NODE_SPEC_REGISTRY keys exactly");
}
if (registryKeys.length < 180) {
  problems.push(`registry unexpectedly small: ${registryKeys.length} entries`);
}

const catalogMetadataCount = registryKeys.filter((key) => {
  const spec = NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY];
  return Boolean(spec.label && spec.category);
}).length;
if (catalogMetadataCount < 175) {
  problems.push(`registry catalog metadata coverage is too low: ${catalogMetadataCount}/${registryKeys.length}`);
}

for (const key of registryKeys) {
  const spec = NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY];
  if (spec.category === "observables" && !spec.family?.includes("observable")) {
    problems.push(`${key}: observables category nodes must declare observable capability`);
  }
}
if (!NODE_SPEC_REGISTRY.kan_reg.family?.includes("observable")) {
  problems.push("kan_reg must declare observable capability for loss-side observable behavior");
}

const registryCodegenKeys = registryKeys.filter((key) => {
  const spec = NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY];
  return typeof spec.codegen === "function";
});
const defaultsCoverageCount = registryKeys.filter((key) => Boolean(nodeRegistryDefaults(key))).length;
function inferFieldKind(key: string, value: unknown): string | null {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "string") return "enum";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (/^lamb/i.test(key) || /(rate|scale|alpha|beta|epsilon|decay|momentum|fraction|ratio|noise|noiseLevel|smoothing|lambda|norm|prob|temperature|dropout|factor|cap|leakyP|mult|multiplier)$/i.test(key)) {
      return "float";
    }
    return Number.isInteger(value) ? "int" : "float";
  }
  return null;
}
const fieldsCoverageCount = registryKeys.filter((key) => {
  const spec = NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY];
  if (spec.fields?.length) return true;
  const defaults = spec.defaults?.();
  if (!defaults || typeof defaults !== "object" || Array.isArray(defaults)) return false;
  return Object.entries(defaults as Record<string, unknown>).some(([fieldKey, value]) => Boolean(inferFieldKind(fieldKey, value)));
}).length;
if (defaultsCoverageCount < 183) {
  problems.push(`registry defaults coverage is too low: ${defaultsCoverageCount}/${registryKeys.length}`);
}
if (fieldsCoverageCount < 155) {
  problems.push(`registry fields coverage is too low: ${fieldsCoverageCount}/${registryKeys.length}`);
}
for (const key of [
  "observable_weight_l2",
  "mup_initialization",
  "kan_reg",
  "diffusion_mse_loss",
  "activation",
  "crl_trainer",
  "crl_env_config",
  "crl_residual_mlp",
  "advection_dataset",
  "absolute_pos_embed_layer",
  "activation_layer",
  "attention_only_model",
  "afno_lite_spatiotemporal_model",
  "afno_encoder_block_layer",
  "afno_patch_decode_layer",
  "afno_patch_embed_layer",
  "afno_spectral_mixer_layer",
  "bigram_low_rank_dataset",
  "biography_lm_dataset",
  "circle_random_walk_dataset",
  "circular_motion_dataset",
  "cogs_dataset",
  "combined_model",
  "causal_mask",
  "dataset_mixer",
  "dataset_mixer_b",
  "diagonal_ssm_token_model",
  "diffusion_pde_dataset",
  "diffusion_score_model",
  "dyck_dataset",
  "einsum",
  "embedding_layer",
  "flatten",
  "formal_language_suite_dataset",
  "gaussian_blob_dataset",
  "gated_mlp_model",
  "gated_mlp_token_model",
  "hole_counting_dataset",
  "hyena_like_conv_model",
  "in_context_associative_recall_dataset",
  "input_sampler",
  "kan_model",
  "kepler_2d_dataset",
  "layer_norm_layer",
  "linear_attention_model",
  "linear_dataset",
  "linear_layer",
  "listops_dataset",
  "local_mixing_layer",
  "memorization_a_dataset",
  "memorization_b_dataset",
  "mlp_model",
  "mlp_token_model",
  "moe_mlp_model",
  "moe_mlp_token_model",
  "modular_addition_dataset",
  "mnist_dataset",
  "multi_hop_fact_chain_dataset",
  "mpp_spatiotemporal_model",
  "ngram_language_dataset",
  "numeric_hyena_model",
  "numeric_transformer_model",
  "pcfg_dataset",
  "phi1_style_dataset",
  "random_input_distribution",
  "random_noise_dataset",
  "reaction_diffusion_dataset",
  "relation_tuple_dataset",
  "reshape",
  "residual_ln_model",
  "resnet_model",
  "rms_norm_layer",
  "rotary_embed_layer",
  "rwkv_time_mix_token_model",
  "scan_dataset",
  "shape_world_dataset",
  "slot_attention_token_model",
  "softmax",
  "symbolic_func_dataset",
  "synthetic_playground_dataset",
  "teacher_dataset",
  "tinystories_dataset",
  "token_prediction_dataset",
  "trainer",
  "transformer_token_model",
  "transformer_multi_token_model",
  "uniform_linear_motion_dataset",
  "unigram_dataset",
  "unembedding_layer",
  "vit_model",
  "adam_optimizer",
  "sgd_optimizer",
  "muon_optimizer",
  "lr_schedule",
  "mup_lr_schedule",
  "mse_loss",
  "cross_entropy_loss",
]) {
  if (!registryCodegenKeys.includes(key)) {
    problems.push(`${key} codegen must be owned by NODE_SPEC_REGISTRY`);
  }
}
const weightL2Codegen = NODE_SPEC_REGISTRY.observable_weight_l2.codegen?.({
  pySym: "verify_weight_l2",
  title: "Weight L2 0",
  nodeId: "observable_weight_l2-test",
  nodeType: "observable_weight_l2",
  data: {},
});
if (!weightL2Codegen?.includes("def fn_verify_weight_l2_weight_l2_norm")) {
  problems.push("observable_weight_l2 registry codegen output is missing expected function");
}

if (problems.length) {
  console.error("Node registry verification FAILED:\n" + problems.map((p) => `  - ${p}`).join("\n"));
  process.exit(1);
}

console.log(
  `OK: ${registryKeys.length} node registry entries derive researchNodeTypes; ${catalogMetadataCount} include catalog metadata; ${registryCodegenKeys.length} registry codegen entries; ${defaultsCoverageCount} defaults entries; ${fieldsCoverageCount} field-schema entries; ${Object.keys(expectedDefaults).length} observable entries derive titles, hints, and defaults.`,
);
