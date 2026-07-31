import type { Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { defaultObservableVizEmbeddingTrajectoryData } from "../../components/nodes/observableVizEmbeddingTrajectoryDefaults";
import { defaultObservableVizHessianEigenvaluesData } from "../../components/nodes/observableVizHessianEigenvaluesDefaults";
import { defaultObservableVizKanRegData } from "../../components/nodes/observableVizKanRegDefaults";
import { defaultObservableVizNeuronTrajectory2dData } from "../../components/nodes/observableVizNeuronTrajectory2dDefaults";
import { defaultObservableVizReluNonlinearData } from "../../components/nodes/observableVizReluNonlinearDefaults";
import { defaultObservableVizWeightL1Data } from "../../components/nodes/observableVizWeightL1Defaults";
import { defaultObservableVizUserData } from "../../components/nodes/observableVizUserDefaults";
import { defaultAttentionMapVizData } from "../../components/nodes/attentionMapVizDefaults";
import { spawnConfigFor } from "../observableVizTrainerSpawn";

/** spawn 等价契约:4 个迁移样本的 spawn defaultData 在迁移前(hand)与迁移后
 * (generated 路径)必须逐字段等价——default builders 是共同地面真值。 */
describe("spawn config equivalence", () => {
  const src = (data: Record<string, unknown>) => ({ id: "s", data }) as unknown as Node;

  it("observable_weight_l2 → hessian defaults with fixed topK=1", () => {
    expect(spawnConfigFor("observable_weight_l2")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 1, "descending"),
      vizVariant: "weight_l2",
    });
  });

  it("observable_weight_product_sv → hessian defaults with source topK (fallback 3)", () => {
    expect(spawnConfigFor("observable_weight_product_sv")!.defaultData("p", "t", src({ topK: 4 }))).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 4, "descending"),
      vizVariant: "weight_product_sv",
    });
    expect(spawnConfigFor("observable_weight_product_sv")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 3, "descending"),
      vizVariant: "weight_product_sv",
    });
  });

  it("observable_weight_displacement → user defaults with ‖Δw‖/‖w₀‖ unit", () => {
    expect(spawnConfigFor("observable_weight_displacement")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Weight displacement", "‖Δw‖/‖w₀‖"),
      vizVariant: "user",
    });
  });

  it("observable_neuron_trajectory_2d → dedicated trajectory defaults", () => {
    expect(spawnConfigFor("observable_neuron_trajectory_2d")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizNeuronTrajectory2dData("p", "t"),
      vizVariant: "neuron_trajectory_2d",
    });
  });

  it("observable_attention_map → paired heatmap defaults that follow latest", () => {
    const got = spawnConfigFor("observable_attention_map")!.defaultData("p", "t", undefined);
    expect(got).toEqual({
      ...defaultAttentionMapVizData("p", "t", "Attention map"),
      vizVariant: "attention_map",
    });
    expect("selectedFrameIndex" in got).toBe(false);
  });
});

/** spawn 等价契约:unit 三态(显式 "value" / omitted)与 vizVariant 逐字段复刻 hand 条目。 */
describe("spawn config unit semantics", () => {
  it("observable_capacity → user defaults with explicit unit value", () => {
    expect(spawnConfigFor("observable_capacity")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Capacity", "value"),
      vizVariant: "capacity",
    });
  });

  it("observable_accuracy → user defaults with OMITTED unit (no vizYAxisLabel key)", () => {
    const got = spawnConfigFor("observable_accuracy")!.defaultData("p", "t", undefined);
    expect(got).toEqual({
      ...defaultObservableVizUserData("p", "t", "Accuracy"),
      vizVariant: "accuracy",
    });
    expect("vizYAxisLabel" in got).toBe(false);
  });

  it("observable_train_test_gap → user defaults with test − train loss unit", () => {
    expect(spawnConfigFor("observable_train_test_gap")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Train vs test gap", "test − train loss"),
      vizVariant: "user",
    });
  });
});

/** spawn.title 复刻 hand 标题 ≠ vizTitle 的场景。 */
describe("spawn config title overrides", () => {
  it("observable_sink_attention_mass → user defaults, title = vizTitle", () => {
    expect(spawnConfigFor("observable_sink_attention_mass")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Sink attention mass", "mean P(sink)"),
      vizVariant: "user",
    });
  });

  it("observable_attention_head_sink_max → spawn.title 'Head sink max' (≠ vizTitle)", () => {
    expect(spawnConfigFor("observable_attention_head_sink_max")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Head sink max", "mean P(sink)"),
      vizVariant: "user",
    });
  });
});

/** scalar_series kind 对各自专用 builder 逐字段等价。 */
describe("spawn config scalar-series builders", () => {
  it("observable_weight_l1 → weight_l1 builder shape", () => {
    expect(spawnConfigFor("observable_weight_l1")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizWeightL1Data("p", "t"),
      vizVariant: "weight_l1",
    });
  });

  it("observable_relu_nonlinear_count → relu builder shape", () => {
    expect(spawnConfigFor("observable_relu_nonlinear_count")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizReluNonlinearData("p", "t"),
      vizVariant: "relu_nonlinear",
    });
  });
});

/** order_from_field 透传两态、gradient_norm fixedTopK、activation_stats seriesLabels。 */
describe("spawn config ordering and labels", () => {
  const src = (data: Record<string, unknown>) => ({ id: "s", data }) as unknown as Node;

  it("observable_hessian_eigenvalues → source topK+order threaded (ascending)", () => {
    expect(
      spawnConfigFor("observable_hessian_eigenvalues")!.defaultData("p", "t", src({ topK: 4, order: "ascending" })),
    ).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 4, "ascending"),
      vizVariant: "hessian_eigenvalues",
    });
  });

  it("observable_hessian_eigenvalues → defaults 5/descending without source", () => {
    expect(spawnConfigFor("observable_hessian_eigenvalues")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 5, "descending"),
      vizVariant: "hessian_eigenvalues",
    });
  });

  it("observable_gradient_norm → hessian defaults fixedTopK=1", () => {
    expect(spawnConfigFor("observable_gradient_norm")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 1, "descending"),
      vizVariant: "gradient_norm",
    });
  });

  it("observable_activation_stats → fixedTopK=2 + seriesLabels mean/std", () => {
    expect(spawnConfigFor("observable_activation_stats")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizHessianEigenvaluesData("p", "t", 2, "descending"),
      vizVariant: "activation_stats",
      seriesLabels: ["mean", "std"],
    });
  });
});

/** title_from_field 动态标题(trim + 回退 vizTitle)+ embedding_trajectory kind。 */
describe("spawn config dynamic titles", () => {
  const src = (data: Record<string, unknown>) => ({ id: "s", data }) as unknown as Node;

  it("observable_embedding_evolution → title from source label (trimmed)", () => {
    expect(spawnConfigFor("observable_embedding_evolution")!.defaultData("p", "t", src({ label: " My drift " }))).toEqual({
      ...defaultObservableVizUserData("p", "t", "My drift", "value"),
      vizVariant: "user",
    });
  });

  it("observable_embedding_evolution → fallback title without source", () => {
    expect(spawnConfigFor("observable_embedding_evolution")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "Embedding evolution", "value"),
      vizVariant: "user",
    });
  });

  it("observable_embedding_trajectory → trajectory builder with dynamic name", () => {
    expect(spawnConfigFor("observable_embedding_trajectory")!.defaultData("p", "t", src({ label: "Tokens" }))).toEqual({
      ...defaultObservableVizEmbeddingTrajectoryData("p", "t", "Tokens"),
      vizVariant: "embedding_trajectory",
    });
    expect(spawnConfigFor("observable_embedding_trajectory")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizEmbeddingTrajectoryData("p", "t", "Embedding trajectory"),
      vizVariant: "embedding_trajectory",
    });
  });
});

/** observable_user: title_from_field + unit "Value"(大写,hand 原文)。 */
describe("spawn config user observable", () => {
  const src = (data: Record<string, unknown>) => ({ id: "s", data }) as unknown as Node;

  it("observable_user → title from source label, unit Value", () => {
    expect(spawnConfigFor("observable_user")!.defaultData("p", "t", src({ label: "My metric" }))).toEqual({
      ...defaultObservableVizUserData("p", "t", "My metric", "Value"),
      vizVariant: "user",
    });
  });

  it("observable_user → fallback title without source", () => {
    expect(spawnConfigFor("observable_user")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizUserData("p", "t", "User observable", "Value"),
      vizVariant: "user",
    });
  });
});

/** kan_reg 收编(最后一条手写表条目):scalar_series 生成路径对 hand builder 逐字段等价。 */
describe("spawn config equivalence (kan_reg fold-in)", () => {
  it("kan_reg → kan_reg builder shape via generated scalar_series", () => {
    expect(spawnConfigFor("kan_reg")!.defaultData("p", "t", undefined)).toEqual({
      ...defaultObservableVizKanRegData("p", "t"),
      vizVariant: "kan_reg",
    });
    expect(spawnConfigFor("kan_reg")!.vizVariant).toBe("kan_reg");
  });
});

/** encoder-toggle 家族：5 条 spawn.title 复刻 hand 标题字面量，2 条 title=vizTitle。 */
describe("spawn config encoder observables", () => {
  const CASES: Array<[string, string, string]> = [
    ["observable_attention_entropy_mean", "Attention entropy", "nat"],
    ["observable_attention_max_weight_mean", "Attention max weight", "mean max P"],
    ["observable_attention_position_bias_ratio", "Position bias ratio", "ratio"],
    ["observable_activation_norm_mean", "Activation norm mean", "‖h‖ mean"],
    ["observable_activation_outlier_ratio", "Activation outlier ratio", "ratio"],
    ["observable_embedding_effective_rank", "Embedding effective rank", "rank"],
    ["observable_embedding_feature_drift", "Embedding feature drift", "1 − cos"],
  ];
  for (const [type, title, unit] of CASES) {
    it(`${type} → user defaults titled "${title}" (${unit})`, () => {
      expect(spawnConfigFor(type)!.defaultData("p", "t", undefined)).toEqual({
        ...defaultObservableVizUserData("p", "t", title, unit),
        vizVariant: "user",
      });
    });
  }
});
