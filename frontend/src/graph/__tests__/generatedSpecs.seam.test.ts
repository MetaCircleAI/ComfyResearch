import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { defaultObservableVizUserData } from "../../components/nodes/observableVizUserDefaults";
import { defaultParametricPathSamplerData } from "../../components/nodes/parametricPathSamplerDefaults";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { specFromGenerated } from "../generatedNodeSpecTypes";
import { isValidCanvasConnection } from "../connectionRules";
import { NODE_REGISTRY, resolveGeneratedComponent } from "../nodeRegistry";
import { NODE_SPEC_REGISTRY, nodeRegistryDefaults, nodeRegistryTypesWithFamily } from "../nodeRegistrySpec";

describe("generated spec seam", () => {
  it("parametric path sampler has one fixed interpolation behavior", () => {
    expect(defaultParametricPathSamplerData()).not.toHaveProperty("interpolationMode");
    expect(GENERATED_NODE_SPECS.parametric_path_sampler.fields).not.toContainEqual(
      expect.objectContaining({ key: "interpolationMode" }),
    );
  });

  it("registry keys = generated(单通道,键序即生成物键序)", () => {
    const generated = Object.keys(GENERATED_NODE_SPECS);
    expect(Object.keys(NODE_SPEC_REGISTRY)).toEqual(generated);
    expect(Object.keys(NODE_REGISTRY)).toEqual(generated);
  });

  it("specFromGenerated produces a registry-shaped spec with fresh-copy defaults", () => {
    const spec = specFromGenerated("observable__seam_test", {
      label: "Seam",
      category: "observables",
      hint: "h",
      defaults: { topK: 3 },
      fields: [{ kind: "int", key: "topK", label: "Top K", defaultValue: 3 }],
      observable: { vizVariant: "user", vizTitle: "Seam", spawnsVizNode: true },
      frontend: { componentKey: "GenericObservableNode" },
    });
    expect(spec.type).toBe("observable__seam_test");
    expect(spec.family).toEqual(["observable"]);
    const a = spec.defaults!();
    const b = spec.defaults!();
    expect(a).toEqual({ topK: 3 });
    expect(a).not.toBe(b);
  });

  it("no-defaults generated spec: undefined defaults + resize:false passthrough", () => {
    // graph_assist_failure_overlay 迁移后的 hasDefaults:false 语义:
    // nodeRegistryDefaults 必须回 undefined(不是 {}),resize 必须透传到 NODE_REGISTRY。
    expect(GENERATED_NODE_SPECS["graph_assist_failure_overlay"]).toBeTruthy();
    expect(GENERATED_NODE_SPECS["graph_assist_failure_overlay"].defaults).toBeUndefined();
    expect(NODE_SPEC_REGISTRY["graph_assist_failure_overlay"].defaults).toBeUndefined();
    expect(nodeRegistryDefaults("graph_assist_failure_overlay")).toBeUndefined();
    expect(NODE_REGISTRY["graph_assist_failure_overlay"].resize).toBe(false);
    // 合成条目:GeneratedNodeSpec 无 defaults → NodeRegistrySpec 不含 defaults 键。
    const spec = specFromGenerated("frontend__no_defaults_probe", {
      label: "P", category: "internal", resizable: false,
      frontend: { componentKey: "GenericObservableNode" },
    });
    expect("defaults" in spec).toBe(false);
    expect(spec.resize).toBe(false);
  });

  it("defaults deep-clone: nested arrays/objects are fresh per spawn", () => {
    const spec = specFromGenerated("frontend__deep_clone_probe", {
      label: "P", category: "analysis",
      defaults: { rows: [], nested: { caches: {} }, n: 1 },
      frontend: { componentKey: "GenericObservableNode" },
    });
    const a = spec.defaults!() as { rows: unknown[]; nested: { caches: object } };
    const b = spec.defaults!() as { rows: unknown[]; nested: { caches: object } };
    expect(a).toEqual({ rows: [], nested: { caches: {} }, n: 1 });
    expect(a.rows).not.toBe(b.rows);
    expect(a.nested).not.toBe(b.nested);
    expect(a.nested.caches).not.toBe(b.nested.caches);
  });

  it("observable_viz undefined-key handling: JSON round-trip identical", () => {
    // hand defaults 含 pairedObservableId/pairedTrainerId 两个 undefined 值键,
    // python defaults 无法表达 undefined,def 侧省略。等价依据:全仓消费均为值
    // 访问(truthy/===/??),零 in/hasOwnProperty/Object.keys 存在性检查;
    // JSON 序列化本就丢 undefined 键——故 JSON 往返后两者必须逐键恒等。
    const hand = JSON.parse(
      JSON.stringify({ ...defaultObservableVizUserData(), vizVariant: "user" }),
    );
    expect(nodeRegistryDefaults("observable_viz")).toEqual(hand);
    const gen = GENERATED_NODE_SPECS["observable_viz"].defaults!;
    expect(Object.prototype.hasOwnProperty.call(gen, "pairedObservableId")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(gen, "pairedTrainerId")).toBe(false);
  });

  it("bad componentKey throws; absent key defaults to Generic (review contract)", () => {
    expect(() => resolveGeneratedComponent("observable__seam_test", "NoSuchAdapter")).toThrow(
      /GENERATED_COMPONENT_ADAPTERS has no such entry/,
    );
    expect(resolveGeneratedComponent("observable__seam_test", undefined)).toBeTypeOf("function");
    expect(resolveGeneratedComponent("observable__seam_test", "GenericObservableNode")).toBeTypeOf("function");
  });

  it("GenericObservableNode handles every schema field kind (source contract)", () => {
    // 无 testing-library,用源契约防"boolean 分支被改回去":四种 kind 必须都有处理分支
    const src = readFileSync(
      resolve(__dirname, "../../components/nodes/GenericObservableNode.tsx"),
      "utf8",
    );
    for (const kind of ["int", "float", "enum", "boolean"]) {
      expect(src).toContain(`f.kind === "${kind}"`);
    }
  });

  it("GenericObservableNode skips enum fields without options (source contract)", () => {
    // defaults 推断的 enum-无-options 字段(embedding_evolution/trajectory 的 label)
    // hand TSX 从不渲染;Generic 必须跳过而不是渲染空 multiselect。
    const src = readFileSync(
      resolve(__dirname, "../../components/nodes/GenericObservableNode.tsx"),
      "utf8",
    );
    expect(src).toContain('f.kind === "enum" && "options" in f && f.options.length');
  });

  it("canvas generic add-node branch excludes observable_user (source contract)", () => {
    // observable_user 迁入 GENERATED_NODE_SPECS 后,前置 generic 分支若不排除会
    // 先命中并 return,DnD options 线程静默丢失——排除条件必须存在。
    const src = readFileSync(resolve(__dirname, "../../components/ResearchCanvas.tsx"), "utf8");
    expect(src).toContain('GENERATED_NODE_SPECS[nodeType] && nodeType !== "observable_user"');
    expect(src).toContain('if (nodeType === "observable_user") {');
  });
});

describe("declared ports", () => {
  const declared = Object.entries(GENERATED_NODE_SPECS).filter(([, g]) => g.ports != null);

  it("declared takeover set is explicit (+ diffusion reproducibility nodes)", () => {
    expect(declared.map(([t]) => t).sort()).toEqual([
      "comment", "curve_annotator", "curve_series_table", "curve_series_viz",
      "dataset_mixer", "dataset_mixer_b",
      "deterministic_diffusion_sampler",
      "image_dataset_displayer", "input_sampler", "metric_compare",
      "model_checkpoint", "observable_bezier_mode_connectivity", "observable_linear_interpolation_barrier", "observable_nearest_train_gl",
      "observable_paired_generation_similarity", "observable_rp_score_sscd", "observable_user", "parametric_path_sampler", "sweep_data_table",
      "table_viz", "teacher_dataset",
      "training_visualization",
    ]);
    // Trainer ports accept the two remaining trainer variants.
    const trainerAccepts = (h: string) =>
      ["trainer", "crl_trainer"].map((type) => ({ type, handles: [h] }));
    expect(GENERATED_NODE_SPECS["model_checkpoint"].ports).toEqual({
      in: [{ id: "model_checkpoint", accepts: trainerAccepts("checkpoint") }],
    });
    expect(GENERATED_NODE_SPECS["training_visualization"].ports).toEqual({
      in: [{ id: "tensor_list", accepts: trainerAccepts("loss_results") }],
    });
    expect(GENERATED_NODE_SPECS["comment"].ports).toEqual({
      // 两 family 并集(VIZ_COMMENT_SOURCE_TYPES 逐字;金标抓过单 family 漏声明)。
      in: [{ id: "comment", accepts: [
        { family: "observable_user_tensor_viz_display", handles: ["comment"] },
        { family: "canvas_comment_source", handles: ["comment"] },
      ] }],
    });
    expect(GENERATED_NODE_SPECS["curve_annotator"].ports).toEqual({
      in: [{ id: "from_viz", accepts: ["training_visualization", "observable_viz", "observable_accuracy"].map(
        (type) => ({ type, handles: ["annotator"] })) }],
    });
    expect(GENERATED_NODE_SPECS["image_dataset_displayer"].ports).toEqual({
      in: [{ id: "dataset", accepts: [{ family: "vision_dataset", handles: ["dataset"] }] }],
    });
    // image_dataset_displayer 声明引用的 vision_dataset family 与
    // VISION_DATASET_KINDS 字面表恰等(4 型 +  cifar10)。
    expect(nodeRegistryTypesWithFamily("vision_dataset" as never).sort()).toEqual([
      "cifar10_dataset", "gaussian_blob_dataset", "hole_counting_dataset", "mnist_dataset", "shape_world_dataset",
    ]);
    // source_io_mode 首战结构 pin(teacher model 口 = cascade 三谓词逐字)。
    expect(GENERATED_NODE_SPECS["teacher_dataset"].ports).toEqual({
      in: [
        { id: "model", accepts: [
          { family: "canvas_full_model", ioMode: "model", handles: ["model"] },
          { family: "canvas_full_model", ioMode: "input-output", handles: ["tensor_out", "tensor"] },
          { type: "combined_model", ioMode: "model", handles: ["tensor", "model"] },
        ] },
        { id: "train_input", accepts: [{ type: "input_sampler", handles: ["sample_tensor"] }] },
        { id: "test_input", accepts: [{ type: "input_sampler", handles: ["sample_tensor"] }] },
      ],
    });
    expect(GENERATED_NODE_SPECS["sweep_data_table"].ports).toEqual({
      in: [{ id: "stream", accepts: [{ type: "tensor_viz_0d", handles: ["out_tensor"] }] }],
    });
    expect(GENERATED_NODE_SPECS["table_viz"].ports).toEqual({
      in: [{ id: "table", accepts: [{ type: "sweep_data_table", handles: ["table"] }] }],
    });
    expect(GENERATED_NODE_SPECS["metric_compare"].ports).toEqual({
      in: ["left", "right"].map((id) => ({ id, accepts: [
        { type: "curve_series_viz", handles: ["compare"] },
        { type: "observable_viz", handles: ["compare"] },
        { type: "tensor_viz_1d", handles: ["compare"] },
      ] })),
    });
    expect(GENERATED_NODE_SPECS["observable_user"].ports).toEqual({
      in: [{ id: "observable_in", accepts: [{ type: "tensor_viz_0d", handles: ["observable"] }] }],
    });
    expect(GENERATED_NODE_SPECS["input_sampler"].ports).toEqual({
      in: [{ id: "distribution", accepts: [{ type: "random_input_distribution", handles: ["input_distribution"] }] }],
    });
    const mixerPorts = {
      in: ["dataset_a", "dataset_b"].map((id) => ({
        id, accepts: [{ family: "canvas_dataset_source", handles: ["dataset", "train_dataset", "test_dataset"] }],
      })),
    };
    expect(GENERATED_NODE_SPECS["dataset_mixer"].ports).toEqual(mixerPorts);
    expect(GENERATED_NODE_SPECS["dataset_mixer_b"].ports).toEqual(mixerPorts);
  });

  it("PortAccept schema boundary: only type/family/ioMode/handles keys", () => {
    // ports 声明刻意止步于静态接受集 + source_io_mode 窄扩展;pattern/prefix/
    // anySource 之类 predicate-DSL 键不得出现；当前字段已覆盖所需表达能力。
    const allowed = new Set(["type", "family", "ioMode", "handles"]);
    for (const [t, g] of declared) {
      for (const p of g.ports!.in) {
        for (const a of p.accepts) {
          for (const k of Object.keys(a)) {
            expect(allowed.has(k), `${t}.${p.id} accept key "${k}" outside schema boundary`).toBe(true);
          }
          if (a.ioMode != null) expect(["model", "input-output"]).toContain(a.ioMode);
        }
      }
    }
  });

  it("ports references are well-formed", () => {
    for (const [t, g] of declared) {
      const seen = new Set<string>();
      for (const p of g.ports!.in) {
        expect(seen.has(p.id), `${t} duplicate in-port ${p.id}`).toBe(false);
        seen.add(p.id);
        expect(p.accepts.length, `${t}.${p.id} empty accepts`).toBeGreaterThan(0);
        for (const a of p.accepts) {
          expect(a.handles.length, `${t}.${p.id} empty handles`).toBeGreaterThan(0);
          expect((a.type == null) !== (a.family == null), `${t}.${p.id} must declare exactly one of type/family`).toBe(true);
          if (a.type != null) expect(NODE_SPEC_REGISTRY[a.type], `${t}.${p.id} unknown source type ${a.type}`).toBeTruthy();
          if (a.family != null) expect(nodeRegistryTypesWithFamily(a.family as never).length, `${t}.${p.id} empty family ${a.family}`).toBeGreaterThan(0);
        }
      }
    }
  });
});

describe("ports verdict end-to-end semantics", () => {
  const N = (id: string, type: string) => ({ id, type, position: { x: 0, y: 0 }, data: {} });
  const check = (srcType: string, sh: string, tgtType: string, th: string) =>
    isValidCanvasConnection(
      { source: "s", target: "t", sourceHandle: sh, targetHandle: th } as never,
      [N("s", srcType), N("t", tgtType)] as never,
      [] as never,
    );

  it("declared target: matching port true, unknown handle false, wrong source false", () => {
    expect(check("tensor_viz_0d", "observable", "observable_user", "observable_in")).toBe(true);
    expect(check("tensor_viz_0d", "observable", "observable_user", "made_up_handle")).toBe(false);
    expect(check("tensor_viz_1d", "observable", "observable_user", "observable_in")).toBe(false);
    expect(check("linear_dataset", "train_dataset", "dataset_mixer", "dataset_b")).toBe(true);
    expect(check("curve_series_viz", "compare", "metric_compare", "left")).toBe(true);
    expect(check("curve_series_viz", "compare", "metric_compare", "right")).toBe(true);
    expect(check("observable_viz", "compare", "metric_compare", "left")).toBe(true);
    expect(check("tensor_viz_1d", "compare", "metric_compare", "left")).toBe(true);
    expect(check("curve_series_viz", "wrong", "metric_compare", "left")).toBe(false);
    expect(check("table_viz", "compare", "metric_compare", "left")).toBe(false);
    expect(check("curve_series_viz", "compare", "metric_compare", "unknown")).toBe(false);
  });

  it("undeclared target falls through the cascade (permissive tail intact)", () => {
    expect(check("mlp_model", "weird_src", "totally_unknown_type", "weird_tgt")).toBe(true);
  });

  it("source_io_mode flips the verdict on data.ioMode alone", () => {
    // 防字段名接错(canvasIoMode 假探针教训):同 source/type/handle,仅
    // data.ioMode 不同 → teacher model 口判定翻转。不依赖 450 万矩阵。
    const N = (id: string, type: string, ioMode: string) =>
      ({ id, type, position: { x: 0, y: 0 }, data: { ioMode } });
    const checkIo = (srcIoMode: string, sh: string) =>
      isValidCanvasConnection(
        { source: "s", target: "t", sourceHandle: sh, targetHandle: "model" } as never,
        [N("s", "mlp_model", srcIoMode), N("t", "teacher_dataset", "model")] as never,
        [] as never,
      );
    expect(checkIo("model", "model")).toBe(true);
    expect(checkIo("input-output", "model")).toBe(false);
    expect(checkIo("input-output", "tensor_out")).toBe(true);
    expect(checkIo("model", "tensor_out")).toBe(false);
  });

  it("schedule families are pinned exactly", () => {
    expect(nodeRegistryTypesWithFamily("trainer_lr_schedule" as never).sort()).toEqual([
      "cyclic_lr_schedule", "lr_schedule", "mup_lr_schedule",
    ]);
    expect(nodeRegistryTypesWithFamily("trainer_batch_schedule" as never)).toEqual([
      "cyclic_batch_schedule",
    ]);
  });

  it("canvas_full_model family matches the derived composite (exact identity)", () => {
    // ports 声明引用的新 family 与 connectionRules 的派生集恒等,防两面漂移:
    // canvas_trainer_model_source ∖ atomic_layer_model ∖ {combined_model}。
    const derived = nodeRegistryTypesWithFamily("canvas_trainer_model_source").filter(
      (t) => !new Set(nodeRegistryTypesWithFamily("atomic_layer_model")).has(t) && t !== "combined_model",
    );
    expect(nodeRegistryTypesWithFamily("canvas_full_model" as never)).toEqual(derived);
    // vision additions, UNet-DDPM, and Small Inception.
    expect(derived.length).toBe(27);
  });
});
