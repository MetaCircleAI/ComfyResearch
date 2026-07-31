import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { NODE_SPEC_REGISTRY } from "../nodeRegistrySpec";
import { resolveGeneratedComponent } from "../nodeRegistry";
import { getSweptAxisIdSet, planTrainSeriesAssignments, trainSeriesAxisKey } from "../trainSeriesPlan";

function trainerGraph(dsType: string, dsData: Record<string, unknown>) {
  const nodes = [
    { id: "ds", type: dsType, data: dsData },
    { id: "m", type: "mlp_model", data: { inputDim: 8, outputDim: 1, depth: 1, width: 8, activation: "relu", seed: 0 } },
    { id: "t", type: "trainer", data: { trainingSteps: 4, logFrequency: 1, batchSize: -1 } },
  ] as unknown as Node[];
  const edges = [
    { id: "e1", source: "ds", target: "t", sourceHandle: "dataset", targetHandle: "dataset" },
    { id: "e2", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" },
  ] as unknown as Edge[];
  return { nodes, edges };
}

/** dataset 通道前端缝契约(空集恒等 + Generic 渲染规则源契约)。 */
describe("dataset channel seams", () => {
  it("GenericDatasetNode is a registered component adapter", () => {
    expect(resolveGeneratedComponent("dataset__seam_test", "GenericDatasetNode")).toBeTypeOf("function");
  });

  it("GenericDatasetNode rendering rules", () => {
    const src = readFileSync(
      resolve(__dirname, "../../components/nodes/GenericDatasetNode.tsx"),
      "utf8",
    );
    // int/float → 多值 sweep 输入(绝不 select);enum 有 options 才 select;bool checkbox
    expect(src).toContain("ComfyIntListField");
    expect(src).toContain("ComfyFloatListField");
    expect(src).toContain('f.kind === "enum" && "options" in f && f.options.length');
    expect(src).toContain('type="checkbox"');
    // 骨架三件:header info / source sockets / min-max 只作约束传参
    expect(src).toContain("DatasetNodeHeaderWithInfo");
    expect(src).toContain("DatasetSourceSockets");
    expect(src).not.toContain("DiscreteMultiSelect<number>");
  });

  it("PDE sweep axes: all 13 numeric fields sweepable, samplingMode never an axis", () => {
    // hand 的 axesForPdeFieldDataset 注册全部 13 个数值字段(含按 kind 隐藏的
    // velocity*/diffusionCoeff/reactionRate)、无 samplingMode 轴——generated 路径
    // 必须逐键复刻。全字段多值 + samplingMode 数组探针:
    const numericKeys = [
      "contextFrames", "channels", "gridSize", "trainSize", "testSize", "warmupSteps",
      "dt", "diffusionCoeff", "reactionRate", "velocityX", "velocityY", "icScale", "initSeed",
    ];
    const data: Record<string, unknown> = { samplingMode: ["fixed", "streaming"] };
    for (const k of numericKeys) data[k] = [1, 2];
    const { nodes, edges } = trainerGraph("diffusion_pde_dataset", data);
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    for (const k of numericKeys) expect(swept.has(trainSeriesAxisKey("ds", k)), k).toBe(true);
    expect(swept.has(trainSeriesAxisKey("ds", "samplingMode"))).toBe(false);
    expect(swept.size).toBe(numericKeys.length);
  });

  it("GenericDatasetNode commit wiring per kind", () => {
    const src = readFileSync(resolve(__dirname, "../../components/nodes/GenericDatasetNode.tsx"), "utf8");
    expect(src).toContain("patch(f.key, packIntList(vals))");
    expect(src).toContain("patch(f.key, packFloatList(vals))");
    expect(src).toContain("patch(f.key, e.target.checked)");
    expect(src).toContain("patch(f.key, v)"); // enum multiselect commit
  });

  it("linear sweep axes: exact ordered set incl enum axes (fidelity)", () => {
    // hand 轴序恰与字段序一致:8 轴含两个枚举轴;alpha(UI 隐藏)与 samplingMode 绝不出轴。
    const keys = ["inputDim", "outputDim", "inputDistribution", "outputDistribution", "trainSize", "testSize", "noiseLevel", "seed"];
    const data: Record<string, unknown> = {
      inputDim: [4, 8], outputDim: [1, 2], inputDistribution: ["standard_normal", "uniform_0_1"],
      outputDistribution: ["additive_gaussian", "deterministic"], trainSize: [64, 128], testSize: [0, 16],
      noiseLevel: [0, 0.1], seed: [0, 1], alpha: [1, 2], samplingMode: ["fixed", "streaming"],
    };
    for (const handle of ["dataset", "test_dataset"]) {
      const { nodes, edges } = trainerGraph("linear_dataset", data);
      if (handle === "test_dataset") {
        nodes.push({ id: "ds2", type: "linear_dataset", data } as unknown as Node);
        edges.push({ id: "e9", source: "ds2", target: "t", sourceHandle: "dataset", targetHandle: "test_dataset" } as unknown as Edge);
      }
      const nid = handle === "test_dataset" ? "ds2" : "ds";
      const swept = [...getSweptAxisIdSet(nodes, edges, "t")].filter((k) => k.startsWith(nid));
      expect(swept).toEqual(keys.map((k) => trainSeriesAxisKey(nid, k)));
    }
  });

  it("mem_b sweep axes: vocabSize mirror fields never explode combinations", () => {
    // vocabSize 的 commit 会把同一多值列表写进 inputDim/outputDim——sweepable=false
    // 保证只有 vocabSize 一条轴;noiseLevel 是 declared 修复轴;枚举轴只剩 outputDistribution。
    const data: Record<string, unknown> = {
      vocabSize: [16, 32], inputDim: [16, 32], outputDim: [16, 32],
      outputDistribution: ["uniform_class_probs", "power_law_class_probs"],
      inputDistribution: ["standard_normal", "uniform_0_1"],
      noiseLevel: [0, 0.1], alpha: [1, 2], trainSize: 160, testSize: 0, seed: 0,
    };
    const { nodes, edges } = trainerGraph("memorization_b_dataset", data);
    const swept = [...getSweptAxisIdSet(nodes, edges, "t")];
    expect(swept).toEqual([
      trainSeriesAxisKey("ds", "vocabSize"),
      trainSeriesAxisKey("ds", "outputDistribution"),
      trainSeriesAxisKey("ds", "noiseLevel"),
      trainSeriesAxisKey("ds", "alpha"),
    ]);
    // 组合数 = 2^4(镜像字段若出轴会变 2^6)
    expect(planTrainSeriesAssignments(nodes, edges, "t").length).toBe(16);
  });


  it("canvas generic branch excludes linear_dataset; options branch survives", () => {
    const src = readFileSync(resolve(__dirname, "../../components/ResearchCanvas.tsx"), "utf8");
    expect(src).toContain('nodeType !== "observable_user" && nodeType !== "linear_dataset"');
    expect(src).toContain("userLinearDatasetId");
  });


  it("collectAxes dataset handles are generated-first (source contract)", () => {
    const src = readFileSync(resolve(__dirname, "../trainSeriesPlan.ts"), "utf8");
    // 不止数出现次数——generated 查询必须位于各自 handle 块内、
    // 且在第一个 legacy linear_dataset 分支之前。
    const trainBlock = src.slice(src.indexOf("const trainDs = "), src.indexOf("const testDs = "));
    const testBlock = src.slice(src.indexOf("const testDs = "), src.indexOf("if (isCrlTrainer)"));
    for (const block of [trainBlock, testBlock]) {
      const gen = block.indexOf("GENERATED_NODE_SPECS[String(n.type)]");
      const legacy = block.indexOf('n?.type === "teacher_dataset"');
      expect(gen).toBeGreaterThan(-1);
      expect(legacy).toBeGreaterThan(-1);
      expect(gen).toBeLessThan(legacy);
    }
  });
});

/** toy-language sweep 保真 + A类修复 + 镜像轴防护。 */
describe("toy-language sweep", () => {
  it("pcfg axes: hand-equal keys incl enum axes; grammarId/samplingMode never axes", () => {
    const data: Record<string, unknown> = {
      vocabSize: [16, 32], contextLength: [8, 16], trainSize: [64, 128], testSize: [0, 16], seed: [0, 1],
      dataSource: ["synthetic", "download"], pcfgGenMode: ["binary_tree", "cfg_sentence"],
      pcfgMaxDepth: [4, 8], pcfgTermProb: [0.2, 0.35], pcfgGrammarId: ["world_model"],
      initSeed: [0, 1], inspectFormat: ["id", "word"],
    };
    const { nodes, edges } = trainerGraph("pcfg_dataset", data);
    const swept = [...getSweptAxisIdSet(nodes, edges, "t")];
    const want = ["vocabSize", "contextLength", "trainSize", "testSize", "seed", "dataSource", "pcfgGenMode", "pcfgMaxDepth", "pcfgTermProb"];
    expect(swept.sort()).toEqual(want.map((k) => trainSeriesAxisKey("ds", k)).sort());
  });

  it("A-fix axes appear: slotNoiseProb/chainHops/depoWindow/manoModulus/lanoNestingDepth (declared)", () => {
    const cases: Array<[string, Record<string, unknown>, string[]]> = [
      ["biography_lm_dataset", { slotNoiseProb: [0, 0.2] }, ["slotNoiseProb"]],
      ["multi_hop_fact_chain_dataset", { chainHops: [2, 3] }, ["chainHops"]],
      ["synthetic_playground_dataset", { depoWindow: [3, 4], manoModulus: [13, 17], lanoNestingDepth: [3, 4] }, ["depoWindow", "manoModulus", "lanoNestingDepth"]],
    ];
    for (const [type, data, keys] of cases) {
      const { nodes, edges } = trainerGraph(type, data);
      const swept = getSweptAxisIdSet(nodes, edges, "t");
      for (const k of keys) expect(swept.has(trainSeriesAxisKey("ds", k)), `${type}.${k}`).toBe(true);
    }
    // 展开数:synthetic_playground 三轴 2^3
    const { nodes, edges } = trainerGraph("synthetic_playground_dataset", { depoWindow: [3, 4], manoModulus: [13, 17], lanoNestingDepth: [3, 4] });
    expect(planTrainSeriesAssignments(nodes, edges, "t").length).toBe(8);
  });

  it("vocabCap mirror never sweeps", () => {
    // 非 text-heavy:vocabCap 不是字段 → 旧图残留列表不出轴(hand 双计 bug 消失);
    // text-heavy:字段存在但 sweepable=false。
    for (const type of ["ngram_language_dataset", "tinystories_dataset"]) {
      const { nodes, edges } = trainerGraph(type, { vocabSize: [32, 64], vocabCap: [32, 64] });
      const swept = getSweptAxisIdSet(nodes, edges, "t");
      expect(swept.has(trainSeriesAxisKey("ds", "vocabSize")), type).toBe(true);
      expect(swept.has(trainSeriesAxisKey("ds", "vocabCap")), type).toBe(false);
      expect(planTrainSeriesAssignments(nodes, edges, "t").length, type).toBe(2);
    }
  });

  it("dyck vocabSize derived field never sweeps; initSeed mirror never sweeps", () => {
    const { nodes, edges } = trainerGraph("dyck_dataset", { vocabSize: [2, 4], numBracketTypes: [1, 2], initSeed: [0, 1], seed: [0, 1] });
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    expect(swept.has(trainSeriesAxisKey("ds", "vocabSize"))).toBe(false);
    expect(swept.has(trainSeriesAxisKey("ds", "numBracketTypes"))).toBe(true);
    expect(swept.has(trainSeriesAxisKey("ds", "initSeed"))).toBe(false);
    expect(swept.has(trainSeriesAxisKey("ds", "seed"))).toBe(true);
  });

  it("canvas generic add: 14 spec defaults ≡ hand factory output", async () => {
    const { defaultToyLanguageDatasetData } = await import("../../components/nodes/toyLanguageDatasetDefaults");
    for (const t of Object.keys(GENERATED_NODE_SPECS).filter((k) => GENERATED_NODE_SPECS[k].family?.includes("toy_language_token_dataset"))) {
      const hand = defaultToyLanguageDatasetData(t as never) as Record<string, unknown>;
      const gen = (GENERATED_NODE_SPECS[t].defaults ?? {}) as Record<string, unknown>;
      for (const k of Object.keys(gen)) expect(hand[k], `${t}.${k}`).toEqual(gen[k]);
      const missing = Object.keys(hand).filter((k) => hand[k] !== undefined && !(k in gen));
      expect(missing, t).toEqual([]);
    }
  });
});

/** vision sweep 镜像方向 + defaults 完整性。 */
describe("vision channel", () => {
  it("CIFAR hand defaults and UI expose both reproducibility and paper-randomization controls", async () => {
    const { defaultVisionDatasetData, defaultParamOrderFor } = await import("../../components/nodes/visionDatasetDefaults");
    const defaults = defaultVisionDatasetData("cifar10_dataset");
    expect(defaults).toMatchObject({
      subsetSeed: 0,
      classBalanced: true,
      normalize: "zero_one",
      inputTransform: "none",
      preprocessing: "none",
      labelCorruption: 0,
    });
    expect(defaultParamOrderFor("cifar10_dataset")).toEqual([
      "downloadCacheDir", "subsetSeed", "classBalanced", "inputTransform", "preprocessing",
      "labelCorruption", "normalize", "flattenOutput", "samplingMode", "trainSize", "testSize", "initSeed",
    ]);
    const src = readFileSync(resolve(__dirname, "../../components/nodes/VisionDatasetNode.tsx"), "utf8");
    expect(src).toContain("CIFAR_INPUT_TRANSFORM_OPTIONS");
    expect(src).toContain("CIFAR_PREPROCESSING_OPTIONS");
    expect(src).toContain('disabled={cifarWhitening}');
  });

  it("initSeed sweeps, seed never sweeps (mirror direction opposite to token families)", () => {
    const { nodes, edges } = trainerGraph("gaussian_blob_dataset", {
      initSeed: [0, 1], seed: [0, 1], imageSize: [16, 28], numClasses: [4, 10],
    });
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    expect(swept.has(trainSeriesAxisKey("ds", "initSeed"))).toBe(true);
    expect(swept.has(trainSeriesAxisKey("ds", "seed"))).toBe(false);
    expect(swept.has(trainSeriesAxisKey("ds", "imageSize"))).toBe(true);
    expect(swept.has(trainSeriesAxisKey("ds", "numClasses"))).toBe(true);
  });

  it("generic add defaults: missing keys are exactly paramOrder; component fallback survives", async () => {
    const { defaultVisionDatasetData } = await import("../../components/nodes/visionDatasetDefaults");
    for (const kind of ["mnist_dataset", "gaussian_blob_dataset", "shape_world_dataset", "hole_counting_dataset"]) {
      const hand = defaultVisionDatasetData(kind as never) as Record<string, unknown>;
      const gen = (GENERATED_NODE_SPECS[kind]?.defaults ?? {}) as Record<string, unknown>;
      const missing = Object.keys(hand).filter((k) => hand[k] !== undefined && !(k in gen)).sort();
      expect(missing, kind).toEqual(["paramOrder"]);
      for (const k of Object.keys(gen)) expect(hand[k], `${kind}.${k}`).toEqual(gen[k]);
    }
    const src = readFileSync(resolve(__dirname, "../../components/nodes/VisionDatasetNode.tsx"), "utf8");
    expect(src).toContain("defaultParamOrderFor");
  });
});

/** CRL sweep 手路由源契约。 */
describe("crl_env_config", () => {
  it("axesForCrlEnvConfig stays hand-routed via the CRL trainer env branch", () => {
    const src = readFileSync(resolve(__dirname, "../trainSeriesPlan.ts"), "utf8");
    // 迁移进 GENERATED_NODE_SPECS 后 sweep 仍走 CRL env-handle 手分派——
    // generated-first 只覆盖 dataset handle;该 hand 路径不许被"顺手清理"。
    expect(src).toContain("axesForCrlEnvConfig");
    expect(src).toContain('en?.type === "crl_env_config"');
    expect(GENERATED_NODE_SPECS["crl_env_config"]).toBeDefined();
    expect(GENERATED_NODE_SPECS["crl_env_config"].family).toBeUndefined();
  });
});

/** /:teacher 卫星轴回归 + mixer 零轴 hostile 测试。 */
describe("final dataset families", () => {
  it("teacher satellite axes survive migration (hazard-1 regression)", () => {
    const nodes = [
      { id: "ds", type: "teacher_dataset", data: { samplingMode: ["fixed", "streaming"] } },
      { id: "sam", type: "input_sampler", data: { numSamples: [400, 800] } },
      { id: "rid", type: "random_input_distribution", data: { inputDim: [4, 8], inputDistribution: ["standard_normal", "uniform_0_1"], noiseDistribution: ["deterministic", "additive_gaussian"], noiseLevel: [0, 0.1], seed: [0, 1] } },
      { id: "m", type: "mlp_model", data: { inputDim: 8, outputDim: 1, depth: 1, width: 8, activation: "relu", seed: 0 } },
      { id: "t", type: "trainer", data: { trainingSteps: 4, logFrequency: 1, batchSize: -1 } },
    ] as unknown as Node[];
    const edges = [
      { id: "e1", source: "ds", target: "t", sourceHandle: "dataset", targetHandle: "dataset" },
      { id: "e2", source: "sam", target: "ds", sourceHandle: "sample_tensor", targetHandle: "train_input" },
      { id: "e3", source: "rid", target: "sam", sourceHandle: "input_distribution", targetHandle: "distribution" },
      { id: "e4", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" },
    ] as unknown as Edge[];
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    // 六轴全面(卫星链的完整轴面)
    const satelliteKeys = [
      trainSeriesAxisKey("sam", "numSamples"),
      trainSeriesAxisKey("rid", "inputDim"),
      trainSeriesAxisKey("rid", "inputDistribution"),
      trainSeriesAxisKey("rid", "noiseDistribution"),
      trainSeriesAxisKey("rid", "noiseLevel"),
      trainSeriesAxisKey("rid", "seed"),
    ];
    for (const k of satelliteKeys) expect(swept.has(k), k).toBe(true);
    expect(swept.has(trainSeriesAxisKey("ds", "samplingMode"))).toBe(false);
  });

  it("mixers mint zero axes even under hostile all-list data (hazard-2)", () => {
    for (const [type, data] of [
      ["dataset_mixer", { trainTotalSamples: [400, 800], testTotalSamples: [0, 100], proportionA: [0.3, 0.5], initSeed: [0, 1], samplingMode: ["fixed", "streaming"] }],
      ["dataset_mixer_b", { interpolationLambda: [0.3, 0.5] }],
    ] as const) {
      const { nodes, edges } = trainerGraph(type, data as Record<string, unknown>);
      const swept = getSweptAxisIdSet(nodes, edges, "t");
      const dsAxes = [...swept].filter((k) => k.startsWith("ds"));
      expect(dsAxes, type).toEqual([]);
      expect(planTrainSeriesAssignments(nodes, edges, "t").length, type).toBe(1);
    }
  });

  it("teacher generated-first exclusion source contract (both handles)", () => {
    const src = readFileSync(resolve(__dirname, "../trainSeriesPlan.ts"), "utf8");
    const hits = src.match(/n\.type !== "teacher_dataset" && GENERATED_NODE_SPECS/g) ?? [];
    expect(hits.length).toBe(2);
  });
});

/** mlp_family sweep 保真(allowlist 路径首启)。 */
describe("mlp family sweep", () => {
  it("mlp_model: full axis surface incl activation enum and outputScale field", () => {
    const { nodes, edges } = trainerGraph("mlp_model", {});
    nodes.find((n) => n.id === "m")!.data = {} as never; // 用 ds 位放 model?—— model handle 需要 m 节点
    // 直接构 model-handle 图:
    const mNodes = [
      { id: "ds", type: "linear_dataset", data: {} },
      { id: "m", type: "mlp_model", data: { inputDim: [4, 8], depth: [1, 2], width: [8, 16], activation: ["relu", "gelu"], outputScale: [0.1, 1], seed: [0, 1], outputDim: 1 } },
      { id: "t", type: "trainer", data: { trainingSteps: 4, logFrequency: 1, batchSize: -1 } },
    ] as unknown as Node[];
    const mEdges = [
      { id: "e1", source: "ds", target: "t", sourceHandle: "dataset", targetHandle: "dataset" },
      { id: "e2", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" },
    ] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("m")).sort();
    // 精确键集(防后续 def 扩张误加轴)
    expect(swept).toEqual(
      ["inputDim", "depth", "width", "activation", "outputScale", "seed"].map((k) => trainSeriesAxisKey("m", k)).sort(),
    );
  });

  it("gated/moe: stale outputScale data still sweeps", () => {
    for (const type of ["gated_mlp_model", "moe_mlp_model"]) {
      const mNodes = [
        { id: "m", type, data: { width: [8, 16], outputScale: [0.1, 1] } },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e2", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      const swept = getSweptAxisIdSet(mNodes, mEdges, "t");
      expect(swept.has(trainSeriesAxisKey("m", "width")), type).toBe(true);
      expect(swept.has(trainSeriesAxisKey("m", "outputScale")), type).toBe(true);
    }
  });
});

/** token-mlp 无轴 pin(hostile 全列表数据仍零轴,44 型教义首验)。 */
describe("token mlp no-axis pins", () => {
  it("hostile all-list data mints zero axes for the token-mlp trio", () => {
    for (const type of ["mlp_token_model", "gated_mlp_token_model", "moe_mlp_token_model"]) {
      const mNodes = [
        { id: "m", type, data: { vocabSize: [50, 100], width: [32, 64], activation: ["relu", "gelu"], seed: [0, 1] } },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("m"));
      expect(swept, type).toEqual([]);
    }
  });
});

/** alt-arch sweep 保真(精确键集,含枚举轴)。 */
describe("alt-arch sweep", () => {
  it("attention_only: exact axis set incl causalAttention/qkNorm enum axes", () => {
    const data: Record<string, unknown> = {
      vocabSize: [50, 100], embedDim: [16, 32], numHeads: [2, 4], contextLength: [4, 8],
      causalAttention: ["yes", "no"], localMixingKernel: [0, 3], qkNorm: ["yes", "no"],
      attnTemperature: [0.5, 1], attnLogitCap: [0, 10], attnDropout: [0, 0.1], seed: [0, 1],
    };
    const mNodes = [
      { id: "m", type: "attention_only_model", data },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].sort();
    expect(swept).toEqual(Object.keys(data).map((k) => trainSeriesAxisKey("m", k)).sort());
  });

  it("hyena: ffMult sweeps with legacy INT semantics", () => {
    // runtime/_scalar_int/UI/notebook 均 int——sweep 值必须取整保真,防标签≠实际。
    const mk = (ff: unknown) => {
      const mNodes = [
        { id: "m", type: "hyena_like_conv_model", data: { depth: [1, 2], convKernel: [5, 7], ffMult: ff } },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      return { mNodes, mEdges };
    };
    // int 列表 → sweep(hand intChoices 语义)
    const a = mk([1, 2]);
    expect(getSweptAxisIdSet(a.mNodes, a.mEdges, "t").has(trainSeriesAxisKey("m", "ffMult"))).toBe(true);
    // 浮点列表 → intChoices 过滤为 fallback 单值,不 swept(hand 语义:非整数被滤,不是取整)
    const b = mk([1.2, 2.8]);
    expect(getSweptAxisIdSet(b.mNodes, b.mEdges, "t").has(trainSeriesAxisKey("m", "ffMult"))).toBe(false);
  });

  it("alt-arch axis order follows field order (declared, mem-family precedent)", () => {
    const mNodes = [
      { id: "m", type: "linear_attention_model", data: { numHeads: [2, 4], localMixingKernel: [0, 3], seed: [0, 1] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")];
    // manifest 字段序:seed → localMixingKernel → numHeads(hand 曾为 numHeads 先行)
    expect(swept).toEqual(["seed", "localMixingKernel", "numHeads"].map((k) => trainSeriesAxisKey("m", k)));
  });
});

/** vector/numeric sweep 保真(4 轴型精确键集)+ numeric_hyena 无轴 pin(allowlist 外)。 */
describe("vector/numeric sweep", () => {
  const sweepGraph = (type: string, data: Record<string, unknown>) => {
    const mNodes = [
      { id: "m", type, data },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
    return getSweptAxisIdSet(mNodes, mEdges, "t");
  };

  it("kan/numeric_transformer/mpp/afno: exact axis surface == field set (hand parity)", () => {
    const cases: Record<string, Record<string, unknown>> = {
      kan_model: {
        inputDim: [4, 8], outputDim: [1, 2], depth: [2, 3], width: [4, 8],
        grid: [3, 5], k: [2, 3], baseFun: ["silu", "identity"], seed: [0, 1],
      },
      numeric_transformer_model: {
        contextLength: [2, 4], inputDim: [1, 2], outputDim: [1, 2], modelDim: [8, 16],
        numHeads: [1, 2], numLayers: [1, 2], ffDim: [16, 32], activation: ["gelu", "relu"],
        encoderBackend: ["pytorch", "stable"], encoderDropout: [0, 0.1],
        spectralNormLinears: ["yes", "no"], stableQkNorm: ["yes", "no"],
        stableAttnTemperature: [0.5, 1], stableAttnLogitCap: [0, 10], stableAttnDropout: [0, 0.1],
        causalAttention: ["yes", "no"], seed: [0, 1],
      },
      mpp_spatiotemporal_model: {
        contextFrames: [2, 4], channels: [1, 2], gridSize: [8, 16], inputDim: [128, 256],
        outputDim: [128, 256], patchSize: [2, 4], embedDim: [16, 32], depth: [1, 2],
        numHeads: [2, 4], ffRatio: [2, 4], dropout: [0, 0.1], seed: [0, 1],
      },
      afno_lite_spatiotemporal_model: {
        contextFrames: [2, 4], channels: [1, 2], gridSize: [8, 16], inputDim: [128, 256],
        outputDim: [128, 256], patchSize: [2, 4], embedDim: [16, 32], depth: [1, 2],
        numHeads: [2, 4], ffRatio: [2, 4], dropout: [0, 0.1], numSpectralBlocks: [1, 2],
        maxFrequencyModes: [2, 4], spectralShrinkFactor: [0.5, 1], seed: [0, 1],
      },
    };
    for (const [type, data] of Object.entries(cases)) {
      const swept = [...sweepGraph(type, data)].sort();
      expect(swept, type).toEqual(Object.keys(data).map((k) => trainSeriesAxisKey("m", k)).sort());
    }
  });

  it("numeric_transformer axis order follows manifest field order (hand parity)", () => {
    const swept = [...sweepGraph("numeric_transformer_model", {
      contextLength: [2, 4], modelDim: [8, 16], activation: ["gelu", "relu"], seed: [0, 1],
    })];
    expect(swept).toEqual(["contextLength", "modelDim", "activation", "seed"].map((k) => trainSeriesAxisKey("m", k)));
  });

  it("numeric_hyena mints zero axes even under hostile all-list data (allowlist pin)", () => {
    const swept = [...sweepGraph("numeric_hyena_model", {
      contextLength: [4, 8], inputDim: [1, 2], outputDim: [1, 2], modelDim: [16, 32],
      depth: [1, 2], convKernel: [3, 5], ffMult: [1, 2], localMixingKernel: [0, 3], seed: [0, 1],
    })].filter((k) => k.startsWith("m"));
    expect(swept).toEqual([]);
  });
});

/** vision 二型无轴 pin(allowlist 外;hostile 全列表数据仍零轴)。 */
describe("vision model no-axis pins", () => {
  it("resnet/vit mint zero axes even under hostile all-list data", () => {
    const cases: Record<string, Record<string, unknown>> = {
      resnet_model: {
        variant: ["resnet18", "resnet34"], baseChannels: [16, 32], blocksStage1: [1, 2],
        blocksStage2: [1, 2], blocksStage3: [1, 2], blocksStage4: [1, 2], kernelSize: [3, 5], seed: [0, 1],
      },
      vit_model: {
        variant: ["tiny", "small"], patchSize: [2, 4], hiddenDim: [64, 128],
        depth: [1, 3], numHeads: [2, 4], seed: [0, 1],
      },
    };
    for (const [type, data] of Object.entries(cases)) {
      const mNodes = [
        { id: "m", type, data },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("m"));
      expect(swept, type).toEqual([]);
    }
  });
});

/** diffusion/crl sweep 保真 + crl activation 卫星 pin + residual_ln/alias 无轴。 */
describe("diffusion/alias/crl sweep", () => {
  const sweep = (type: string, data: Record<string, unknown>, trainerType = "trainer") => {
    const mNodes = [
      { id: "m", type, data },
      { id: "t", type: trainerType, data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
    return getSweptAxisIdSet(mNodes, mEdges, "t");
  };

  it("diffusion_score_model: exact axis surface == 6 int fields (hand parity)", () => {
    const data: Record<string, unknown> = {
      inputDim: [4, 8], hiddenDim: [64, 128], depth: [2, 3], timeEmbedDim: [32, 64],
      diffusionTimesteps: [50, 100], seed: [0, 1],
    };
    const swept = [...sweep("diffusion_score_model", data)].sort();
    expect(swept).toEqual(Object.keys(data).map((k) => trainSeriesAxisKey("m", k)).sort());
  });

  it("crl_residual_mlp on crl_trainer: 9 axes sweep, activation never does (hand parity)", () => {
    const data: Record<string, unknown> = {
      stateDim: [4, 8], actionDim: [2, 4], goalDim: [2, 4], actorWidth: [64, 128],
      criticWidth: [64, 128], actorDepth: [4, 8], criticDepth: [4, 8], embedDim: [32, 64],
      activation: ["silu", "relu"], seed: [0, 1],
    };
    const swept = [...sweep("crl_residual_mlp", data, "crl_trainer")].sort();
    const expected = Object.keys(data).filter((k) => k !== "activation");
    expect(swept).toEqual(expected.map((k) => trainSeriesAxisKey("m", k)).sort());
  });

  it("residual_ln mint zero axes under hostile data", () => {
    const cases: Record<string, Record<string, unknown>> = {
      residual_ln_model: { dim: [128, 256], depth: [50, 100], alpha: [0.5, 1], lnMode: ["pre_ln", "post_ln"], activation: ["relu", "gelu"], seed: [0, 1] },
    };
    for (const [type, data] of Object.entries(cases)) {
      const swept = [...sweep(type, data)].filter((k) => k.startsWith("m"));
      expect(swept, type).toEqual([]);
    }
  });
});

/** combined+atomic 无轴 pin(全员 allowlist 外;atomic 链 literal switch stay-hand)。 */
describe("combined/atomic no-axis pins", () => {
  it("atomic chain members and combined_model mint zero axes under hostile data", () => {
    const cases: Record<string, Record<string, unknown>> = {
      linear_layer: { inFeatures: [10, 20], outFeatures: [10, 20], bias: [0, 1], seed: [0, 1] },
      activation_layer: { activation: ["relu", "gelu"], leakyP: [0, 0.1] },
      embedding_layer: { numEmbeddings: [2048, 4096], embeddingDim: [32, 64], seed: [0, 1] },
      afno_encoder_block_layer: { embedDim: [32, 64], numHeads: [2, 4], ffRatio: [1, 2], seed: [0, 1] },
      rotary_embed_layer: { rotaryDim: [32, 64], thetaBase: [10000, 100000], seed: [0, 1] },
      combined_model: { displayName: ["a", "b"] },
    };
    for (const [type, data] of Object.entries(cases)) {
      const mNodes = [
        { id: "m", type, data },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("m"));
      expect(swept, type).toEqual([]);
    }
  });
});

/** tensor 族 spawn defaults 保真(null 运行态键/shape 列表)+ 无轴 pin。 */
describe("tensor-ops channel", () => {
  it("spawn defaults reproduce hand shapes incl null state keys (spawn_defaults)", () => {
    expect(GENERATED_NODE_SPECS["tensor_add"].defaults).toEqual({ outputTensor: null, lastError: null });
    expect(GENERATED_NODE_SPECS["tensor_constant"].defaults).toEqual({
      shape: [2, 3], init: "zero", initSeed: 0, outputTensor: null, lastError: null,
    });
    expect(GENERATED_NODE_SPECS["fake_tensor"].defaults).toEqual({ shape: [2, 3, 4], dtype: "float", lastError: null });
    expect(GENERATED_NODE_SPECS["flatten"].defaults).toEqual({ exceptDim: null, ioMode: "input-output", levelMode: "high" });
    expect(GENERATED_NODE_SPECS["tensor_linspace"].defaults).toEqual({
      start: 0, end: 1, numPoints: 8, space: "linear", outputTensor: null, lastError: null,
    });
    expect(GENERATED_NODE_SPECS["elementwise_transform"].defaults).toEqual({
      ruleLatex: "x^2", outputTensor: null, lastError: null,
    });
  });

  it("tensor ops and ai4science atomic layers mint zero axes under hostile data", () => {
    const cases: Record<string, Record<string, unknown>> = {
      tensor_concat: { inputCount: [2, 3], concatDimension: [0, 1] },
      tensor_constant: { init: ["zero", "gaussian"], initSeed: [0, 1] },
      tensor_linspace: { start: [0, 1], end: [1, 2], numPoints: [8, 16] },
      softmax: { dimension: [-1, 1] },
      pairwise_rbf_layer: { inFeatures: [9, 18], outFeatures: [24, 48], bias: [0, 1], seed: [0, 1] },
      causal_mask: { diagonalOffset: [0, 1] },
    };
    for (const [type, data] of Object.entries(cases)) {
      const mNodes = [
        { id: "m", type, data },
        { id: "t", type: "trainer", data: {} },
      ] as unknown as Node[];
      const mEdges = [{ id: "e", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" }] as unknown as Edge[];
      const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("m"));
      expect(swept, type).toEqual([]);
    }
  });
});

/** optimizer sweep 保真(7 型精确键集,轴=fields 教义)+ ui/specCode 透传。 */
describe("optimizer channel", () => {
  const sweepOpt = (type: string, data: Record<string, unknown>) => {
    const mNodes = [
      { id: "o", type, data },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "o", target: "t", sourceHandle: "optimizer", targetHandle: "optimizer" }] as unknown as Edge[];
    return getSweptAxisIdSet(mNodes, mEdges, "t");
  };

  it("all seven optimizers: exact axis surface == field set (hand parity)", () => {
    const cases: Record<string, Record<string, unknown>> = {
      adam_optimizer: { learningRate: [0.001, 0.01], beta1: [0.9, 0.95], beta2: [0.99, 0.999], epsilon: [1e-8, 1e-6], weightDecay: [0, 0.01] },
      adamw_optimizer: { learningRate: [0.001, 0.01], beta1: [0.9, 0.95], beta2: [0.99, 0.999], epsilon: [1e-8, 1e-6], weightDecay: [0.01, 0.1] },
      sgd_optimizer: { learningRate: [0.01, 0.1], momentum: [0, 0.9], weightDecay: [0, 0.01] },
      signsgd_optimizer: { learningRate: [0.001, 0.01], weightDecay: [0, 0.01] },
      muon_optimizer: { learningRate: [0.003, 0.01], momentum: [0.9, 0.95] },
      shampoo_optimizer: { learningRate: [0.01, 0.1], momentum: [0, 0.9], epsilon: [1e-8, 1e-6], weightDecay: [0, 0.01], preconditionFrequency: [5, 10], maxPreconditionerDim: [512, 1024] },
      soap_optimizer: { learningRate: [0.0003, 0.003], beta1: [0.9, 0.95], beta2: [0.99, 0.999], epsilon: [1e-8, 1e-6], weightDecay: [0, 0.01], preconditionFrequency: [5, 10], maxPreconditionerDim: [512, 1024] },
    };
    for (const [type, data] of Object.entries(cases)) {
      const swept = [...sweepOpt(type, data)].sort();
      expect(swept, type).toEqual(Object.keys(data).map((k) => trainSeriesAxisKey("o", k)).sort());
    }
  });

  it("axis order follows field order; intList keeps int semantics", () => {
    const swept = [...sweepOpt("shampoo_optimizer", {
      learningRate: [0.01, 0.1], preconditionFrequency: [5, 10],
    })];
    expect(swept).toEqual(["learningRate", "preconditionFrequency"].map((k) => trainSeriesAxisKey("o", k)));
    // intList 语义:非整数列表被 intChoices 过滤为 fallback 单值 → 不 sweep
    const filtered = sweepOpt("shampoo_optimizer", { preconditionFrequency: [1.5, 2.5] });
    expect(filtered.has(trainSeriesAxisKey("o", "preconditionFrequency"))).toBe(false);
  });

  it("ui/specCode/fields survive specFromGenerated into NODE_SPEC_REGISTRY (SchemaNode contract)", () => {
    const spec = NODE_SPEC_REGISTRY["sgd_optimizer"];
    expect(spec.ui?.accent).toBe("optimizer");
    expect(spec.ui?.socketRows).toBe("optimizerLrSchedule");
    expect(spec.ui?.codeKind).toBe("optimizer");
    expect(spec.ui?.info?.title).toBe("SGD optimizer");
    expect(spec.ui?.info?.text).toContain("Classic stochastic gradient descent");
    expect(typeof spec.specCode).toBe("function");
    expect(String(spec.specCode?.({ learningRate: 0.05, momentum: 0, weightDecay: 0 }))).toContain("0.05");
    expect(spec.fields?.map((f) => `${f.key}:${f.kind}`)).toEqual([
      "learningRate:floatList", "momentum:floatList", "weightDecay:floatList",
    ]);
    expect((spec.fields?.[0] as { positiveOnly?: boolean }).positiveOnly).toBe(true);
  });
});

/** lr/mup 卫星双路径保真 + 去重结构 pin(optimizer 通道收官)。 */
describe("lr schedule satellites", () => {
  it("lr_schedule satellite axes survive via the optimizer branch (exact set incl enum)", () => {
    const mNodes = [
      { id: "o", type: "sgd_optimizer", data: {} },
      { id: "ls", type: "lr_schedule", data: { lrWarmupSteps: [0, 100], lrSchedule: ["constant", "cosine"], cosineLrMinFraction: [0, 0.1] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [
      { id: "e0", source: "o", target: "t", sourceHandle: "optimizer", targetHandle: "optimizer" },
      { id: "e1", source: "ls", target: "o", sourceHandle: "schedule", targetHandle: "lr_schedule" },
    ] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("ls")).sort();
    expect(swept).toEqual(["cosineLrMinFraction", "lrSchedule", "lrWarmupSteps"].map((k) => trainSeriesAxisKey("ls", k)).sort());
  });

  it("mup dual-handle dedup: one node on both handles mints axes exactly once", () => {
    const mNodes = [
      { id: "o", type: "adam_optimizer", data: {} },
      { id: "ms", type: "mup_lr_schedule", data: { mupEmbedLrMult: [1, 2], mupHiddenLrMult: [1, 2], mupOutputLrMult: [1, 2] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [
      { id: "e0", source: "o", target: "t", sourceHandle: "optimizer", targetHandle: "optimizer" },
      { id: "e1", source: "ms", target: "o", sourceHandle: "mup", targetHandle: "lr_schedule" },
      { id: "e2", source: "ms", target: "o", sourceHandle: "mup", targetHandle: "mup_lr_schedule" },
    ] as unknown as Edge[];
    const assigns = planTrainSeriesAssignments(mNodes, mEdges, "t");
    // 3 轴 ×2 值 → 8 组合;若去重失效轴翻倍 → 64
    expect(assigns.length).toBe(8);
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((k) => k.startsWith("ms"));
    expect(swept.length).toBe(3);
  });

  it("mup on the dedicated handle alone still mints its axes", () => {
    const mNodes = [
      { id: "o", type: "adam_optimizer", data: {} },
      { id: "ms", type: "mup_lr_schedule", data: { mupEmbedLrMult: [1, 2] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [
      { id: "e0", source: "o", target: "t", sourceHandle: "optimizer", targetHandle: "optimizer" },
      { id: "e1", source: "ms", target: "o", sourceHandle: "mup", targetHandle: "mup_lr_schedule" },
    ] as unknown as Edge[];
    expect(getSweptAxisIdSet(mNodes, mEdges, "t").has(trainSeriesAxisKey("ms", "mupEmbedLrMult"))).toBe(true);
  });
});

/** loss sweep 保真(6 型;lossMaskCustom/targetNorm 无轴 pins;多 loss 循环)。 */
describe("loss channel", () => {
  const sweepLoss = (nodes: Array<[string, string, Record<string, unknown>]>) => {
    const mNodes = [...nodes.map(([id, type, data]) => ({ id, type, data })), { id: "t", type: "trainer", data: {} }] as unknown as Node[];
    const mEdges = nodes.map(([id], i) => ({ id: `e${i}`, source: id, target: "t", sourceHandle: "loss", targetHandle: "loss" })) as unknown as Edge[];
    return getSweptAxisIdSet(mNodes, mEdges, "t");
  };

  it("primary + weight-reg losses: exact axis surface (hand parity, multi-loss loop)", () => {
    const swept = sweepLoss([
      ["m1", "mse_loss", { lossScale: [1, 2], lossMaskContextLength: [1, 4], lossMaskMode: ["all", "last_context"] }],
      ["l1", "l1_reg", { lossScale: [0.1, 0.5] }],
    ]);
    expect([...swept].sort()).toEqual([
      trainSeriesAxisKey("l1", "lossScale"),
      trainSeriesAxisKey("m1", "lossMaskContextLength"),
      trainSeriesAxisKey("m1", "lossMaskMode"),
      trainSeriesAxisKey("m1", "lossScale"),
    ].sort());
  });

  it("cross_entropy: 4 axes sweep, lossMaskCustom never does (hand parity)", () => {
    const swept = sweepLoss([["c", "cross_entropy_loss", {
      lossScale: [1, 2], labelSmoothing: [0, 0.1], lossMaskContextLength: [1, 4],
      lossMaskMode: ["all", "custom"], lossMaskCustom: ["a", "b"],
    }]]);
    const keys = [...swept].sort();
    expect(keys).toEqual(["labelSmoothing", "lossMaskContextLength", "lossMaskMode", "lossScale"].map((k) => trainSeriesAxisKey("c", k)).sort());
  });

  it("l2_projection mints zero axes under hostile data (ghost-axis pin)", () => {
    const swept = [...sweepLoss([["p", "l2_projection", { targetNorm: [1, 2] }]])];
    expect(swept).toEqual([]);
  });
});

/** kan_reg 收编双 handle 语义 pin。 */
describe("kan_reg dual-handle semantics", () => {
  it("kan_reg on observables handle mints its six axes (generated-first takeover)", () => {
    const mNodes = [
      { id: "k", type: "kan_reg", data: { regMetric: ["edge_forward_spline_n", "edge_backward"], lamb: [0.01, 0.1], lambL1: [1, 2], lambEntropy: [2, 4], lambCoef: [0, 1], lambCoefDiff: [0, 1] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "k", target: "t", sourceHandle: "observable", targetHandle: "observables" }] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((x) => x.startsWith("k"));
    expect(swept.length).toBe(6);
  });

  it("hostile kan_reg on the loss handle mints zero axes (LOSS_SWEEP_ALLOWLIST exclusion)", () => {
    const mNodes = [
      { id: "k", type: "kan_reg", data: { lamb: [0.01, 0.1], lambL1: [1, 2] } },
      { id: "t", type: "trainer", data: {} },
    ] as unknown as Node[];
    const mEdges = [{ id: "e", source: "k", target: "t", sourceHandle: "loss", targetHandle: "loss" }] as unknown as Edge[];
    const swept = [...getSweptAxisIdSet(mNodes, mEdges, "t")].filter((x) => x.startsWith("k"));
    expect(swept).toEqual([]);
  });
});

/** trainer/crl_trainer sweep 保真(computeDevice/boolean 无轴 pins)。 */
describe("trainer channel", () => {
  it("trainer: 4 axes sweep; computeDevice and disableExtraObservables never do", () => {
    const mNodes = [
      { id: "t", type: "trainer", data: { trainingSteps: [500, 1000], logFrequency: [5, 10], batchSize: [-1, 32], gradClipMaxNorm: [0, 1], computeDevice: ["cpu", "auto"], disableExtraObservables: [false, true] } },
    ] as unknown as Node[];
    const swept = [...getSweptAxisIdSet(mNodes, [] as unknown as Edge[], "t")].sort();
    expect(swept).toEqual(["batchSize", "gradClipMaxNorm", "logFrequency", "trainingSteps"].map((k) => trainSeriesAxisKey("t", k)).sort());
  });

  it("crl_trainer: 10 axes sweep; computeDevice/disableEntropy never do", () => {
    const mNodes = [
      { id: "t", type: "crl_trainer", data: {
        trainingSteps: [40, 80], logFrequency: [5, 10], batchSize: [32, 64], unrollLength: [24, 48],
        sgdStepsPerTrainStep: [4, 8], gamma: [0.9, 0.99], logsumexpPenaltyCoeff: [0.1, 0.2],
        entropyParam: [0.5, 1], maxReplayChunks: [100, 200], seed: [0, 1],
        computeDevice: ["cpu", "auto"], disableEntropy: [false, true],
      } },
    ] as unknown as Node[];
    const swept = [...getSweptAxisIdSet(mNodes, [] as unknown as Edge[], "t")];
    expect(swept.length).toBe(10);
    expect(swept.includes(trainSeriesAxisKey("t", "computeDevice"))).toBe(false);
    expect(swept.includes(trainSeriesAxisKey("t", "disableEntropy"))).toBe(false);
  });
});
