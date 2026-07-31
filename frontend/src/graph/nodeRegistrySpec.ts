import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { specFromGenerated } from "./generatedNodeSpecTypes";

export type NodeFamily =
  | "observable"
  | "token_model"
  | "mlp_token_family"
  | "mlp_family"
  | "moe_model"
  | "vector_model"
  | "vision_model"
  | "token_classification_dataset"
  | "vision_dataset"
  | "vector_regression_dataset"
  | "diffusion_noise_dataset"
  | "dataset_mixer"
  | "pde_field_dataset"
  | "toy_language_token_dataset"
  | "text_heavy_toy_language_dataset"
  | "linear_like_dataset"
  | "activation_model"
  | "silu_default_activation_model"
  | "memorization_dataset"
  | "diffusion_loss_model"
  | "trainer_runner"
  | "atomic_layer_model"
  | "canvas_layer_strip_chain"
  | "canvas_tensor_multi_input"
  | "canvas_single_tensor_target"
  /** comment-connect capability(可作 comment 节点连线源;跨 viz/analysis/tensor 类,非领域类别)。 */
  | "canvas_comment_source"
  | "observable_user_tensor_transform"
  | "observable_user_tensor_viz_display"
  | "observable_user_tensor_viz_anchor"
  | "activation_sample_dataset"
  | "trainer_weight_regularizer_loss"
  | "trainer_loss_socket_aux"
  | "trainer_primary_loss"
  | "trainer_loss_viz_spawn"
  | "canvas_tensor_source"
  | "canvas_dataset_source"
  | "canvas_activation_dataset_source"
  | "canvas_trainer_autoconnect_dataset"
  | "canvas_trainer_model_source"
  /** full-model 画布连接面(canvas_trainer_model_source ∖ atomic ∖ combined,
   * 28 员恰等 invariant 钉住);ports 声明的 source_family 引用。 */
  | "canvas_full_model"
  | "dataset_tensor_direct_arrays"
  | "agent_text_context"
  | "optimizer_node"
  /** lr-schedule 源资格(lr_schedule/mup_lr_schedule/cyclic_lr_schedule 恰三员)。 */
  | "trainer_lr_schedule"
  /** trainer batch_schedule socket 源(cyclic_batch_schedule 单员)。 */
  | "trainer_batch_schedule"
  | "analysis_runner"
  | "curve_series_sink";

export type ObservableVizVariant =
  | "weight_l2"
  | "weight_l1"
  | "capacity"
  | "accuracy"
  | "relu_nonlinear"
  | "kan_reg"
  | "hessian_eigenvalues"
  | "gradient_norm"
  | "activation_stats"
  | "user"
  | "embedding_trajectory"
  | "weight_product_sv"
  | "neuron_trajectory_2d"
  | "information_plane"
  | "layer_spectral_norm"
  | "attention_map";

// sweepable:false → axesFromGeneratedSpec 跳过该字段；options 对 enum 可缺省
// (无 options 的 enum 不出 sweep 轴、Generic 不渲染)。二者只存在于
// generatedNodeSpecs.ts(UI/sweep truth),绝不进 manifest 裸四键。
export type FieldDef =
  | { kind: "int"; key: string; label: string; defaultValue: number; min?: number; max?: number; step?: number; tooltip?: string; ariaLabel?: string; sweepable?: boolean }
  | { kind: "float"; key: string; label: string; defaultValue: number; min?: number; max?: number; step?: number; tooltip?: string; positiveOnly?: boolean; ariaLabel?: string; sweepable?: boolean; sweepKind?: "int" }
  | { kind: "enum"; key: string; label: string; defaultValue: string; options?: readonly string[]; tooltip?: string; sweepable?: boolean }
  | { kind: "boolean"; key: string; label: string; defaultValue: boolean; tooltip?: string; sweepable?: boolean }
  | { kind: "string"; key: string; label: string; defaultValue: string; tooltip?: string; sweepable?: boolean }
  | { kind: "floatList"; key: string; label: string; defaultValue: number; min?: number; max?: number; positiveOnly?: boolean; tooltip?: string; ariaLabel?: string; sweepable?: boolean }
  | { kind: "intList"; key: string; label: string; defaultValue: number; min?: number; positiveOnly?: boolean; tooltip?: string; ariaLabel?: string; sweepable?: boolean };

export type NodeCodegenArgs = {
  pySym: string;
  title: string;
  nodeId: string;
  nodeType: string;
  data: Record<string, unknown>;
  ctx?: unknown;
};

export type NodeAccent = "optimizer";
export type NodeSocketRows = "optimizerLrSchedule";
export type NodeCodeKind = "model" | "optimizer" | "observable" | "dataset";
export interface NodeUiSpec {
  accent?: NodeAccent;
  socketRows?: NodeSocketRows;
  codeKind?: NodeCodeKind;
  info?: { title: string; text: string };
}

export interface NodeRegistrySpec<D = Record<string, unknown>> {
  type: string;
  label?: string;
  category?: string;
  hint?: string;
  version?: number;
  defaults?: () => D;
  codegen?: (args: NodeCodegenArgs) => string;
  specCode?: (data: D) => string;
  ui?: NodeUiSpec;
  family?: readonly NodeFamily[];
  fields?: readonly FieldDef[];
  observable?: {
    infoMarkdown?: string;
    vizTitle?: string;
    vizVariant?: ObservableVizVariant;
    spawnsVizNode?: boolean;
  };
  resize?: boolean;
}

// 双通道拆除——LEGACY_NODE_SPEC_REGISTRY/CATALOG_NODE_METADATA/
// NODE_SPEC_REGISTRY is assembled from generated Python definitions through
// specFromGenerated.
export const NODE_SPEC_REGISTRY = Object.fromEntries(
  Object.entries(GENERATED_NODE_SPECS).map(([type, g]) => [type, specFromGenerated(type, g)]),
) as Record<string, NodeRegistrySpec>;

export type RegisteredNodeKind = keyof typeof NODE_SPEC_REGISTRY;

export function nodeRegistryTitle(nodeType: string): string | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.label;
}

export function nodeRegistryLabel(nodeType: string): string | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.label;
}

export function nodeRegistryHint(nodeType: string): string | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.hint;
}

export function nodeRegistryDefaults(nodeType: string): Record<string, unknown> | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.defaults?.();
}

export function nodeRegistryTypesWithFamily(family: NodeFamily): string[] {
  return Object.entries(NODE_SPEC_REGISTRY)
    .filter(([, spec]) => spec.family?.includes(family))
    .map(([type]) => type)
    .sort((a, b) => a.localeCompare(b));
}

export function nodeRegistryTypesWithField(fieldKey: string): string[] {
  return Object.entries(NODE_SPEC_REGISTRY)
    .filter(([, spec]) => spec.fields?.some((field) => field.key === fieldKey))
    .map(([type]) => type)
    .sort((a, b) => a.localeCompare(b));
}

export function nodeRegistryCodegen(args: NodeCodegenArgs): string | undefined {
  return NODE_SPEC_REGISTRY[args.nodeType as RegisteredNodeKind]?.codegen?.(args);
}

export function nodeRegistryObservableInfoMarkdown(nodeType: string): string | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.observable?.infoMarkdown;
}

export function nodeRegistryObservableVizTitle(nodeType: string): string | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.observable?.vizTitle;
}

export function nodeRegistryObservableVizVariant(nodeType: string): ObservableVizVariant | undefined {
  return NODE_SPEC_REGISTRY[nodeType as RegisteredNodeKind]?.observable?.vizVariant;
}
