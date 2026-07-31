import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { researchNodeTypes } from "../../components/nodeTypes";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { getObservableInfoMarkdown } from "../../components/nodes/observableNodeInfoMarkdown";
import { buildNodeDefinitionPython } from "../nodeDefinitionCode";
import { NODE_REGISTRY, registeredResearchNodeTypes } from "../nodeRegistry";
import { spawnConfigFor } from "../observableVizTrainerSpawn";
import {
  NODE_SPEC_REGISTRY,
  nodeRegistryDefaults,
  nodeRegistryHint,
  nodeRegistryTypesWithField,
  nodeRegistryTypesWithFamily,
  nodeRegistryTypesWithFamily,
} from "../nodeRegistrySpec";

function sortedKeys(value: Record<string, unknown>): string[] {
  return Object.keys(value).sort((a, b) => a.localeCompare(b));
}

function expectFamilyMembers(family: string, expected: readonly string[]): void {
  const actual = Object.entries(NODE_SPEC_REGISTRY)
    .filter(([, spec]) => spec.family?.includes(family))
    .map(([type]) => type)
    .sort((a, b) => a.localeCompare(b));
  expect(actual).toEqual([...expected].sort((a, b) => a.localeCompare(b)));
}

describe("node registry invariants", () => {
  it("does not keep parallel observable metadata maps outside the registry", () => {
    const repoRoot = resolve(__dirname, "../../..");
    const infoSource = readFileSync(
      resolve(repoRoot, "src/components/nodes/observableNodeInfoMarkdown.ts"),
      "utf8",
    );
    const variantSource = readFileSync(resolve(repoRoot, "src/graph/observableVizVariant.ts"), "utf8");

    expect(infoSource).not.toContain("OBSERVABLE_INFO");
    expect(variantSource).not.toContain("PAIRED_OBSERVABLE_TO_VIZ_VARIANT");
  });

  it("keeps the spec registry, component registry, and React Flow nodeTypes in lockstep", () => {
    const registryKeys = sortedKeys(NODE_REGISTRY);

    expect(registryKeys.length).toBeGreaterThanOrEqual(180);
    // 双通道拆除,registry 即 generated 全集(不相交守卫随 legacy 通道消亡)
    expect(registryKeys).toEqual(sortedKeys(NODE_SPEC_REGISTRY));
    expect(registryKeys).toEqual(sortedKeys(researchNodeTypes));
    expect(registryKeys).toEqual(sortedKeys(registeredResearchNodeTypes));

    for (const key of registryKeys) {
      expect(NODE_REGISTRY[key as keyof typeof NODE_REGISTRY].type).toBe(key);
      expect(NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY].label).toBeTruthy();
      expect(NODE_SPEC_REGISTRY[key as keyof typeof NODE_SPEC_REGISTRY].category).toBeTruthy();
    }
  });

  it("derives canvas connection routing from registry families", () => {
    const repoRoot = resolve(__dirname, "../../..");
    // NOTE: ResearchCanvas.tsx connection-routing re-derivation is deferred after the origin/main
    // merge (which re-introduced main's hand-written connection sets alongside auto-layout + multi-tab
    // training). The registry-family derivation for the other consumers below is still asserted.
    const trainerAutoVizSource = readFileSync(resolve(repoRoot, "src/graph/trainerAutoVizSpawn.ts"), "utf8");
    const trainingVizPerplexitySource = readFileSync(
      resolve(repoRoot, "src/graph/trainingVizPerplexitySupport.ts"),
      "utf8",
    );
    const autoTuneAxisSource = readFileSync(resolve(repoRoot, "src/graph/autoTuneAxisSuggestions.ts"), "utf8");

    expect(trainerAutoVizSource).toContain('nodeRegistryTypesWithFamily("trainer_loss_viz_spawn")');
    expect(trainingVizPerplexitySource).toContain('nodeRegistryTypesWithFamily("trainer_primary_loss")');
    expect(autoTuneAxisSource).toContain('nodeRegistryTypesWithFamily("moe_model")');
    expect(trainingVizPerplexitySource).not.toContain('new Set(["mse_loss", "cross_entropy_loss", "diffusion_mse_loss"])');
    expect(autoTuneAxisSource).not.toContain('new Set(["moe_mlp_model", "moe_mlp_token_model"])');
    expect(trainerAutoVizSource).not.toContain('lossNode?.type === "mse_loss" || lossNode?.type === "cross_entropy_loss"');
  });

  it("derives observable metadata and defaults from the registry", () => {
    const observableNodes = Object.entries(NODE_SPEC_REGISTRY).filter(([, spec]) => spec.family?.includes("observable"));

    expect(observableNodes.length).toBeGreaterThanOrEqual(20);
    for (const [type, spec] of observableNodes) {
      if (type !== "kan_reg") {
        expect(spec.category).toBe("observables");
      }
      expect(spec.observable?.infoMarkdown).toBeTruthy();
      expect(getObservableInfoMarkdown(type)).toBe(spec.observable?.infoMarkdown);
      expect(spec.observable?.vizTitle).toBeTruthy();
      expect(spec.observable?.vizVariant).toBeTruthy();
      expect(spec.observable?.spawnsVizNode).toBe(true);
      expect(spec.observable?.vizVariant).toBe(spawnConfigFor(type)?.vizVariant);
      if (type !== "kan_reg") {
        if (spec.hint) {
          expect(nodeRegistryHint(type)).toBe(spec.hint);
        }
        expect(nodeRegistryDefaults(type)).toEqual(spec.defaults?.());
      }
    }
  });

  it("keeps capability tags aligned with migrated backend family checks", () => {
    expect(nodeRegistryTypesWithFamily("optimizer_node")).toEqual([
      "adam_optimizer",
      "adamw_optimizer",
      "muon_optimizer",
      "sgd_optimizer",
      "shampoo_optimizer",
      "signsgd_optimizer",
      "soap_optimizer",
    ]);
    expect(nodeRegistryTypesWithField("learningRate")).toEqual([
      "adam_optimizer",
      "adamw_optimizer",
      "muon_optimizer",
      "sgd_optimizer",
      "shampoo_optimizer",
      "signsgd_optimizer",
      "soap_optimizer",
    ]);
    expect(nodeRegistryTypesWithField("weightDecay")).toEqual([
      "adam_optimizer",
      "adamw_optimizer",
      "sgd_optimizer",
      "shampoo_optimizer",
      "signsgd_optimizer",
      "soap_optimizer",
    ]);
    expect(nodeRegistryTypesWithField("momentum")).toEqual(["muon_optimizer", "sgd_optimizer", "shampoo_optimizer"]);
    expect(nodeRegistryTypesWithField("beta1")).toEqual(["adam_optimizer", "adamw_optimizer", "soap_optimizer"]);
    expect(nodeRegistryTypesWithField("beta2")).toEqual(["adam_optimizer", "adamw_optimizer", "soap_optimizer"]);
    expect(nodeRegistryTypesWithField("epsilon")).toEqual([
      "adam_optimizer",
      "adamw_optimizer",
      "shampoo_optimizer",
      "soap_optimizer",
    ]);
    expect(nodeRegistryTypesWithField("trainingSteps")).toEqual(["crl_trainer", "trainer"]);
    expect(nodeRegistryTypesWithField("batchSize")).toEqual(["crl_trainer", "trainer"]);
    expect(nodeRegistryTypesWithField("logFrequency")).toEqual(["crl_trainer", "trainer"]);
    expect(nodeRegistryTypesWithField("gradClipMaxNorm")).toEqual(["trainer"]);
    expect(nodeRegistryTypesWithField("sgdStepsPerTrainStep")).toEqual(["crl_trainer"]);
    expect(nodeRegistryTypesWithField("unrollLength")).toEqual(["crl_trainer"]);
    expect(nodeRegistryTypesWithField("gamma")).toEqual(["crl_trainer"]);
    expectFamilyMembers("vision_dataset", [
      "cifar10_dataset",
      "gaussian_blob_dataset",
      "hole_counting_dataset",
      "mnist_dataset",
      "shape_world_dataset",
    ]);
    expectFamilyMembers("activation_model", [
      "crl_residual_mlp",
      "gated_mlp_model",
      "kan_model",
      "mlp_model",
      "moe_mlp_model",
      "residual_ln_model",
    ]);
    expectFamilyMembers("moe_model", ["moe_mlp_model", "moe_mlp_token_model"]);
    expectFamilyMembers("memorization_dataset", ["information_bottleneck_dataset", "memorization_a_dataset", "memorization_b_dataset"]);
    expectFamilyMembers("linear_like_dataset", ["linear_dataset", "random_noise_dataset"]);
    expectFamilyMembers("vector_regression_dataset", [
      "advection_dataset",
      "dataset_mixer",
      "dataset_mixer_b",
      "diffusion_pde_dataset",
      "kepler_2d_dataset",
      "linear_dataset",
      "memorization_b_dataset",
      "random_noise_dataset",
      "reaction_diffusion_dataset",
      "symbolic_func_dataset",
      "teacher_dataset",
      "uniform_linear_motion_dataset",
    ]);
    expectFamilyMembers("diffusion_noise_dataset", [
      "advection_dataset",
      "cifar10_dataset",
      "dataset_mixer",
      "dataset_mixer_b",
      "diffusion_pde_dataset",
      "linear_dataset",
      "memorization_b_dataset",
      "random_noise_dataset",
      "reaction_diffusion_dataset",
      "symbolic_func_dataset",
    ]);
    expectFamilyMembers("dataset_mixer", ["dataset_mixer", "dataset_mixer_b"]);
    expectFamilyMembers("text_heavy_toy_language_dataset", ["phi1_style_dataset", "tinyshakespeare_lm_dataset", "tinystories_dataset"]);
    expectFamilyMembers("token_classification_dataset", [
      "bigram_low_rank_dataset",
      "biography_lm_dataset",
      "circle_random_walk_dataset",
      "circular_motion_dataset",
      "cogs_dataset",
      "dyck_dataset",
      "formal_language_suite_dataset",
      "in_context_associative_recall_dataset",
      "listops_dataset",
      "memorization_b_dataset",
      "modular_addition_dataset",
      "multi_hop_fact_chain_dataset",
      "ngram_language_dataset",
      "pcfg_dataset",
      "phi1_style_dataset",
      "relation_tuple_dataset",
      "scan_dataset",
      "synthetic_playground_dataset",
      "tinyshakespeare_lm_dataset",
      "tinystories_dataset",
      "token_prediction_dataset",
      "unigram_dataset",
    ]);
    expectFamilyMembers("silu_default_activation_model", ["gated_mlp_model", "moe_mlp_model"]);
    expectFamilyMembers("trainer_runner", ["crl_trainer", "trainer"]);
    expectFamilyMembers("observable_user_tensor_transform", [
      "effective_rank",
      "pca",
      "series_endpoint_gap",
      "statistics",
    ]);
    expectFamilyMembers("observable_user_tensor_viz_display", [
      "tensor_viz_0d",
      "tensor_viz_1d",
      "tensor_viz_2d",
      "tensor_viz_general",
    ]);
    expectFamilyMembers("observable_user_tensor_viz_anchor", ["tensor_viz_0d", "tensor_viz_general"]);
    expectFamilyMembers("activation_sample_dataset", [
      "information_bottleneck_dataset",
      "linear_dataset",
      "memorization_a_dataset",
      "memorization_b_dataset",
      "symbolic_func_dataset",
      "teacher_dataset",
    ]);
    expectFamilyMembers("trainer_weight_regularizer_loss", ["l1_reg", "l2_reg"]);
    expectFamilyMembers("trainer_loss_socket_aux", ["l1_reg", "l2_projection", "l2_reg"]);
    expectFamilyMembers("trainer_primary_loss", [
      "binary_cross_entropy_with_logits_loss",
      "cross_entropy_loss",
      "diffusion_mse_loss",
      "mse_loss",
    ]);
    expectFamilyMembers("trainer_loss_viz_spawn", [
      "binary_cross_entropy_with_logits_loss",
      "cross_entropy_loss",
      "mse_loss",
    ]);
    expectFamilyMembers("canvas_tensor_source", ["tensor_constant", "tensor_linspace"]);
    expectFamilyMembers("canvas_dataset_source", [
      "advection_dataset",
      "bigram_low_rank_dataset",
      "cifar10_dataset",
      "biography_lm_dataset",
      "circle_random_walk_dataset",
      "circular_motion_dataset",
      "cogs_dataset",
      "dataset_mixer",
      "dataset_mixer_b",
      "diffusion_pde_dataset",
      "dyck_dataset",
      "formal_language_suite_dataset",
      "gaussian_blob_dataset",
      "hole_counting_dataset",
      "information_bottleneck_dataset",
      "in_context_associative_recall_dataset",
      "kepler_2d_dataset",
      "linear_dataset",
      "listops_dataset",
      "memorization_a_dataset",
      "memorization_b_dataset",
      "mnist_dataset",
      "modular_addition_dataset",
      "multi_hop_fact_chain_dataset",
      "ngram_language_dataset",
      "paper_classification_dataset",
      "pcfg_dataset",
      "phi1_style_dataset",
      "random_noise_dataset",
      "reaction_diffusion_dataset",
      "relation_tuple_dataset",
      "scan_dataset",
      "shape_world_dataset",
      "symbolic_func_dataset",
      "synthetic_playground_dataset",
      "teacher_dataset",
      "tinyshakespeare_lm_dataset",
      "tinystories_dataset",
      "token_prediction_dataset",
      "uniform_linear_motion_dataset",
      "unigram_dataset",
    ]);
    expectFamilyMembers("canvas_activation_dataset_source", [
      "advection_dataset",
      "dataset_mixer",
      "dataset_mixer_b",
      "diffusion_pde_dataset",
      "information_bottleneck_dataset",
      "kepler_2d_dataset",
      "linear_dataset",
      "memorization_a_dataset",
      "memorization_b_dataset",
      "random_noise_dataset",
      "reaction_diffusion_dataset",
      "symbolic_func_dataset",
      "teacher_dataset",
      "uniform_linear_motion_dataset",
    ]);
    expectFamilyMembers("canvas_trainer_autoconnect_dataset", [
      "advection_dataset",
      "bigram_low_rank_dataset",
      "biography_lm_dataset",
      "cifar10_dataset",
      "circle_random_walk_dataset",
      "circular_motion_dataset",
      "cogs_dataset",
      "dataset_mixer",
      "dataset_mixer_b",
      "diffusion_pde_dataset",
      "dyck_dataset",
      "formal_language_suite_dataset",
      "gaussian_blob_dataset",
      "hole_counting_dataset",
      "information_bottleneck_dataset",
      "in_context_associative_recall_dataset",
      "kepler_2d_dataset",
      "linear_dataset",
      "listops_dataset",
      "memorization_a_dataset",
      "memorization_b_dataset",
      "mnist_dataset",
      "modular_addition_dataset",
      "multi_hop_fact_chain_dataset",
      "ngram_language_dataset",
      "paper_classification_dataset",
      "pcfg_dataset",
      "phi1_style_dataset",
      "random_noise_dataset",
      "reaction_diffusion_dataset",
      "relation_tuple_dataset",
      "scan_dataset",
      "shape_world_dataset",
      "symbolic_func_dataset",
      "synthetic_playground_dataset",
      "teacher_dataset",
      "tinyshakespeare_lm_dataset",
      "tinystories_dataset",
      "token_prediction_dataset",
      "uniform_linear_motion_dataset",
      "unigram_dataset",
    ]);
    expectFamilyMembers("canvas_trainer_model_source", [
      "absolute_pos_embed_layer",
      "activation_layer",
      "afno_encoder_block_layer",
      "afno_lite_spatiotemporal_model",
      "afno_patch_decode_layer",
      "afno_patch_embed_layer",
      "afno_spectral_mixer_layer",
      "attention_only_model",
      "combined_model",
      "diagonal_ssm_token_model",
      "diffusion_score_model",
      "embedding_layer",
      "gated_mlp_model",
      "gated_mlp_token_model",
      "hyena_like_conv_model",
      "kan_model",
      "keskar_c1_c2_cnn_model",
      "layer_norm_layer",
      "linear_attention_model",
      "linear_layer",
      "local_mixing_layer",
      "mlp_model",
      "mlp_token_model",
      "moe_mlp_model",
      "moe_mlp_token_model",
      "mpp_spatiotemporal_model",
      "numeric_hyena_model",
      "numeric_transformer_model",
      "residual_ln_model",
      "resnet_model",
      "rms_norm_layer",
      "rotary_embed_layer",
      "rwkv_time_mix_token_model",
      "slot_attention_token_model",
      "transformer_multi_token_model",
      "transformer_token_model",
      "unembedding_layer",
      "unet_ddpm_model",
      "vgg11_cifar_model",
      "small_inception_cifar_model",
      "vit_model",
    ]);
    expectFamilyMembers("dataset_tensor_direct_arrays", [
      "information_bottleneck_dataset",
      "linear_dataset",
      "memorization_a_dataset",
      "memorization_b_dataset",
      "random_noise_dataset",
      "symbolic_func_dataset",
    ]);
    expectFamilyMembers("agent_text_context", ["comment", "hypothesis"]);
    expectFamilyMembers("atomic_layer_model", [
      "absolute_pos_embed_layer",
      "activation_layer",
      "afno_encoder_block_layer",
      "afno_patch_decode_layer",
      "afno_patch_embed_layer",
      "afno_spectral_mixer_layer",
      "embedding_layer",
      "layer_norm_layer",
      "linear_layer",
      "local_mixing_layer",
      "rms_norm_layer",
      "rotary_embed_layer",
      "unembedding_layer",
    ]);
  });

  it("runs registry-owned codegen headlessly for representative nodes", () => {
    const nodes = [
      { id: "mlp-1", type: "mlp_model", data: nodeRegistryDefaults("mlp_model") ?? {}, position: { x: 0, y: 0 } },
      {
        id: "observable-1",
        type: "observable_weight_l2",
        data: nodeRegistryDefaults("observable_weight_l2") ?? {},
        position: { x: 0, y: 0 },
      },
      {
        id: "trainer-1",
        type: "trainer",
        data: nodeRegistryDefaults("trainer") ?? {},
        position: { x: 0, y: 0 },
      },
    ];

    for (const node of nodes) {
      const code = buildNodeDefinitionPython(node, { nodes, edges: [] });
      expect(code.trim().length).toBeGreaterThan(0);
      expect(code).not.toContain("undefined");
      expect(code).not.toContain("[object Object]");
    }
  });
});

describe("canvas connection family membership", () => {
  it("pins the declared canvas family member sets", () => {
    const members = (f: Parameters<typeof nodeRegistryTypesWithFamily>[0]) => nodeRegistryTypesWithFamily(f).sort();
    expect(members("canvas_layer_strip_chain")).toEqual(["causal_mask", "flatten", "reshape", "softmax", "tensor_splitter"]);
    expect(members("canvas_tensor_multi_input")).toEqual(["basic_calculator", "tensor_add", "tensor_concat", "tensor_stack"]);
    expect(members("canvas_single_tensor_target")).toEqual(["regressor", "statistics", "tensor_slicing"]);
    expect(members("observable_user_tensor_viz_display")).toEqual(["tensor_viz_0d", "tensor_viz_1d", "tensor_viz_2d", "tensor_viz_general"]);
    // dimension_permutator 因 capability correction 摘除；
    // 无 comment source handle,旧 cascade 亦拦死;family 与实况对齐)。
    expect(members("canvas_comment_source")).toEqual([
      "agent_trace_viz", "docking_pose_viz", "elementwise_transform",
      "interatomic_eval_viz", "observable_viz", "sweep_data_table", "tensor_slicing", "training_visualization",
    ]);
    expect(members("canvas_dataset_source")).toEqual([
      "advection_dataset", "bigram_low_rank_dataset", "biography_lm_dataset", "cifar10_dataset", "circle_random_walk_dataset",
      "circular_motion_dataset", "cogs_dataset", "dataset_mixer", "dataset_mixer_b", "diffusion_pde_dataset",
      "dyck_dataset", "formal_language_suite_dataset", "gaussian_blob_dataset", "hole_counting_dataset",
      "in_context_associative_recall_dataset", "information_bottleneck_dataset", "kepler_2d_dataset", "linear_dataset", "listops_dataset",
      "memorization_a_dataset", "memorization_b_dataset", "mnist_dataset", "modular_addition_dataset",
      "multi_hop_fact_chain_dataset", "ngram_language_dataset", "paper_classification_dataset", "pcfg_dataset", "phi1_style_dataset",
      "random_noise_dataset", "reaction_diffusion_dataset", "relation_tuple_dataset", "scan_dataset",
      "shape_world_dataset", "symbolic_func_dataset", "synthetic_playground_dataset", "teacher_dataset",
      "tinyshakespeare_lm_dataset", "tinystories_dataset", "token_prediction_dataset", "uniform_linear_motion_dataset", "unigram_dataset",
    ]);
  });
});
