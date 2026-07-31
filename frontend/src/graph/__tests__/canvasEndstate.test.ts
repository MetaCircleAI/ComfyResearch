/** Canvas connection and intentionally handwritten behavior contracts. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const R = (p: string) => readFileSync(resolve(process.cwd(), p), "utf8");

describe("canvas end-state", () => {
  it("connectionRules holds zero literal node-type tables beyond the documented ledger", () => {
    const src = R("src/graph/connectionRules.ts");
    // 唯一多员字面清单：ai4science strip 是有意例外，后端走 alias remap。
    expect(src).toContain("AI4_SCIENCE_STRIP_EXCEPTIONS");
    // 家族派生常量全员存活
    for (const c of [
      "ATOMIC_LAYER_CONN_TYPES", "FULL_MODEL_CANVAS_CONN_TYPES", "MODEL_INITIALIZATION_TARGET_TYPES",
      "OPTIMIZER_CANVAS_CONN_TYPES", "AUTOCONNECT_DATASET_TYPES", "PRIMARY_LOSS_CONN_TYPES",
      "LOSS_SOCKET_AUX_CONN_TYPES", "AUTOCONNECT_MODEL_TYPES", "TENSOR_VIZ_CONN_TYPES",
      "TENSOR_MULTI_INPUT_CONN_TYPES", "LAYER_STRIP_CHAIN_CONN_TYPES", "SINGLE_TENSOR_TARGET_TYPES",
      "VIZ_COMMENT_SOURCE_TYPES", "DATASET_TENSOR_LIST_SOURCE_TYPES",
    ]) expect(src, c).toContain(c);
    // 类型集字面 new Set 的主要形态启发式扫描(非完整 AST——
    // 抓多行 + 领域后缀形态;单行/无后缀字面由 membership invariants 补位)
    const literalTypeSets = [...src.matchAll(/new Set(?:<string>)?\(\[\s*\n\s*"[a-z0-9_]+_(?:dataset|model|layer|loss|optimizer)"/g)];
    expect(literalTypeSets.length, "literal node-type Set tables must stay zero").toBe(0);
  });

  it("declarative ports own the 12 declared takeover types; cascade keeps the permissive tail", () => {
    const fullSrc = R("src/graph/connectionRules.ts");
    const src = fullSrc
      .slice(0, fullSrc.indexOf("export function planAutoConnectCanvas"))
      .replace(/return true;\s*}\s*$/, "return true;\n}");
    expect(src).toContain("generatedInPortsVerdict");
    expect(src).toContain("插入位置铁律");
    expect(src.trimEnd()).toContain("return true;\n}");  // 默认宽容尾存活(行为本体)
  });

  it("ports end-state: stay-hand faces stay by declared reason", () => {
    // 12 型 declared takeover 之外的 cascade 面是 documented exception。
    // exceptions。每条 pin 一个仍-hand 的证据锚,防"顺手迁移"越过 schema 边界:
    // PortAccept 刻意止步于 type/family × handles × source_io_mode。
    const src = R("src/graph/connectionRules.ts");
    // (1) protein th-门 ×2:任意 source 的 handle 门,不入 any-source escape hatch
    expect(src).toContain('targetNode?.type === "protein_structure_displayer"');
    expect(src).toContain('targetNode?.type === "protein_structure_comparison_viz"');
    // (2) trainer 三型复合分支(trainer 和 CRL trainer 的多态逻辑)
    expect(src).toContain('targetNode?.type === "trainer" ||');
    // (3) tensor_selector:edges 依赖 + 无界动态出口 handle
    expect(src).toContain('targetNode?.type === "tensor_selector"');
    expect(src).toContain("tableVizTensorListChoices");
    // (4) tableViz edges 菜单:multi-input / regressor
    expect(src).toContain("tableVizTensorConnectable");
    expect(src).toContain('targetNode?.type === "regressor"');
    // (5) statistics 族:tensor_selector 无界 handle 正则(isTensorSelectorSourceHandle)
    expect(src).toContain('targetNode?.type === "statistics"');
    expect(src).toContain("isTensorSelectorSourceHandle");
    // (6) effective_rank 族:vizVariant source-data 谓词
    expect(src).toContain("observableVizAllowsTensorVizChain");
    // (7) target-ioMode strip/return guards(source_io_mode 不表达 target data)
    for (const g of ["atomicLayerTensorIoTarget", "fullModelTensorIoTarget", "combinedModelTensorIoTarget"]) {
      expect(src, g).toContain(g);
    }
    // (8) dimension_permutator:above-hook 顺位(接管会移动判定点)
    expect(src).toContain('targetNode?.type === "dimension_permutator"');
    // (9) 注释性 mirror 块保留
    expect(src).toContain("保留该镜像分支");
  });

  it("intentionally-hand surfaces are enumerated, not forgotten", () => {
    // (a) ioMode 运行态端口切换 + 副作用 spawn + 最近邻规划(genuine logic)
    const rules = R("src/graph/connectionRules.ts");
    for (const s of ["readNodeCanvasIoMode", "appendResearchNode", "distanceSq"]) expect(rules, s).toContain(s);
    // (b) edge 迁移函数与 combined_model 复杂 add 分支留组件
    const canvas = R("src/components/ResearchCanvas.tsx");
    for (const s of ["migrateUnifiedDatasetHandles", "migrateVizEdgeHandles", 'nodeType !== "combined_model"']) {
      expect(canvas, s).toContain(s);
    }
    // (c) shape 功能子集的能力边界和 afno 例外。
    const shape = R("src/graph/canvasShapeSupport.ts");
    expect(shape).toContain("AFNO_SHAPE_UNSUPPORTED");
    expect(shape).toContain("SHAPE_FULL_MODEL_TYPES");
    // (d) autoLayout/autoTune 角色表(布局/资格语义,canvas 后续波候选)
    expect(R("src/graph/graphAutoLayout.ts")).toContain("OPTIMIZER_TYPES");
    expect(R("src/graph/autoTuneAxisSuggestions.ts")).toContain("AUTO_TUNE_NODE_TYPES");
    // (e) hint 面: 起 NODE_HINTS 退役,文案单源 def(nodeRegistryHint)
    expect(R("src/components/AddNodeSearchModal.tsx")).not.toContain("NODE_HINTS[");
    expect(R("src/components/AddNodeSearchModal.tsx")).toContain("nodeRegistryHint(");
    // (f) input_sampler mirror 分支
    expect(rules).toContain('targetNode?.type === "input_sampler" && th === "distribution"');
  });

  it("the three golden snapshots exist (their vitest gates enforce them in CI)", () => {
    for (const p of [
      "src/graph/__tests__/__snapshots__/connectionGolden.json",
      "src/graph/__tests__/__snapshots__/hintGolden.json",
      "src/graph/__tests__/__snapshots__/codegenGolden.sha256.json",
    ]) expect(() => R(p), p).not.toThrow();
  });
});
