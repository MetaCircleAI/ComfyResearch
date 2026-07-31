/**
 * 连接矩阵金标生成器。
 *
 * 三层:matrix(静态五元组全枚举 sha + 分桶计数)/ graphCases(图态 fixtures)/
 * effectsCases(applyCanvasConnection / planAutoConnectCanvas 结构化快照,
 * 新节点 id 归一为 $newN)。--check 模式重算并与 committed golden 比对。
 */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { Connection, Edge, Node } from "@xyflow/react";

import {
  applyCanvasConnection,
  isValidCanvasConnection,
  planAutoConnectCanvas,
} from "../src/graph/connectionRules";
import { NODE_SPEC_REGISTRY } from "../src/graph/nodeRegistrySpec";

// bundle 后 import.meta.url 指向 node_modules/.cache——生成器恒从 frontend/ 运行,用 cwd。
export const GOLDEN_PATH = resolve(process.cwd(), "src/graph/__tests__/__snapshots__/connectionGolden.json");

/** (sh, th) 组合集:cascade 分支中实际出现的搭配 + hostile 未知口(recon 词汇表)。 */
const HANDLE_PAIRS: ReadonlyArray<readonly [string, string]> = [
  ["tensor", "tensor"], ["tensor", "tensor_in"], ["tensor", "tensor_1"], ["tensor", "tensor_2"],
  ["tensor", "tensor_return"], ["tensor", "tensor_list"], ["tensor", "stream"],
  ["tensor_out", "tensor"], ["tensor_out", "tensor_in"], ["tensor_out", "tensor_1"],
  ["out_tensor", "tensor"], ["out_tensor", "stream"], ["out_tensor", "observable_in"],
  ["observable", "observable_in"], ["tensor", "distribution"], ["sample_tensor", "distribution"],
  ["out_tensor_list", "tensor_list"], ["tensor_boundary", "tensor_in"], ["tensor_boundary", "tensor"],
  ["tensor_list", "tensor_list"], ["selected_tensor", "tensor"],
  ["dataset", "dataset"], ["train_dataset", "dataset"], ["test_dataset", "dataset"],
  ["dataset", "dataset_a"], ["dataset", "dataset_b"], ["dataset", "train_dataset"],
  ["train_dataset", "dataset_a"], ["test_dataset", "dataset_a"],
  ["train_dataset", "dataset_b"], ["test_dataset", "dataset_b"],
  ["model", "model"], ["model", "tensor"], ["tensor", "model"],
  ["loss", "loss"], ["optimizer", "optimizer"], ["observables", "observables"],
  ["observable", "observables"], ["observable_results", "tensor"], ["loss_results", "tensor_list"],
  ["checkpoint", "model_checkpoint"], ["model_checkpoint", "model"],
  ["initialization", "initialization"], ["init", "initialization"],
  ["lr_schedule", "lr_schedule"], ["mup_lr_schedule", "mup_lr_schedule"],
  ["lr_schedule", "optimizer_lr_schedule"], ["mup_lr_schedule", "optimizer_lr_schedule"],
  ["mup_lr_schedule", "lr_schedule"],
  ["sample_tensor", "train_input"], ["sample_tensor", "test_input"],
  ["input_distribution", "distribution"], ["distribution", "input_distribution"],
  ["transformed_tensor", "tensor"], ["principal_components", "tensor"], ["u", "tensor"], ["s", "tensor"], ["v", "tensor"],
  ["table", "table"], ["table", "tensor_list"], ["env", "env"],
  ["coords", "coords"], ["coords", "pred_coords"], ["coords", "true_coords"],
  ["comment", "comment"], ["", ""], ["tensor", ""],
  ["__unknown_src__", "tensor"], ["tensor", "__unknown_tgt__"], ["__unknown_src__", "__unknown_tgt__"],
  // 扩针:strip 源 handle 接 model 口
  //(fullModel/combined tensor-io 源)+ curve_annotator 入口。
  ["tensor_out", "model"], ["annotator", "from_viz"],
  // 扩针(旧规则基线):paper-repro 新口——trainer batch_schedule、
  // 参数路径采样器双 checkpoint 口、curve-series 链(stream/series/curves)。
  ["batch_schedule", "batch_schedule"], ["model", "checkpoint_sb"], ["model", "checkpoint_lb"],
  ["stream", "stream"], ["out_tensor_list", "stream"], ["series", "curves"],
];

/** ioMode 敏感型(model/layer/combined 系)才展开 ioMode 变体。 */
const IO_MODE_VALUES = ["model", "input-output"] as const;
const IO_SENSITIVE_PREFIX = /(_model|_layer|combined_model|reshape|flatten|softmax|causal_mask|tensor_splitter|einsum|elementwise_transform|tensor_slicing)$/;

export function computeMatrix(): { sha256: string; total: number; trueCount: number; buckets: Record<string, number> } {
  const types = Object.keys(NODE_SPEC_REGISTRY).sort();
  const h = createHash("sha256");
  const buckets: Record<string, number> = {};
  let total = 0;
  let trueCount = 0;
  for (const srcType of types) {
    const srcModes = IO_SENSITIVE_PREFIX.test(srcType) ? IO_MODE_VALUES : ["model"] as const;
    for (const tgtType of types) {
      const tgtModes = IO_SENSITIVE_PREFIX.test(tgtType) ? IO_MODE_VALUES : ["model"] as const;
      for (const srcMode of srcModes) {
        for (const tgtMode of tgtModes) {
          // 字段名为 data.ioMode（由 readNodeCanvasIoMode 读取）；曾误写成
          // canvasIoMode，导致 394 万探针的 ioMode 变体全部返回 false。
          const nodes = [
            { id: "s", type: srcType, position: { x: 0, y: 0 }, data: { ioMode: srcMode } },
            { id: "t", type: tgtType, position: { x: 100, y: 0 }, data: { ioMode: tgtMode } },
          ] as unknown as Node[];
          let row = 0;
          for (const [sh, th] of HANDLE_PAIRS) {
            const ok = isValidCanvasConnection(
              { source: "s", target: "t", sourceHandle: sh, targetHandle: th } as Connection,
              nodes,
              [] as Edge[],
            );
            total += 1;
            row = row * 2 + (ok ? 1 : 0);
            if (ok) trueCount += 1;
          }
          h.update(`${srcType}|${srcMode}>${tgtType}|${tgtMode}=${row.toString(36)};`);
          buckets[tgtType] = (buckets[tgtType] ?? 0) + row.toString(2).split("1").length - 1;
        }
      }
    }
  }
  return { sha256: h.digest("hex"), total, trueCount, buckets };
}

function normalizeIds(nodes: Node[], edges: Edge[], known: Set<string>) {
  const map = new Map<string, string>();
  let i = 0;
  const norm = (id: string | null | undefined) => {
    if (id == null) return id;
    if (known.has(id)) return id;
    if (!map.has(id)) map.set(id, `$new${i++}`);
    return map.get(id);
  };
  // 字段分治:id 引用字段归一,literal 字段原样保留——
  // vizVariant 若被误归一成 $new0,值漂移就抓不到。
  const ID_REF_FIELDS = ["pairedObservableId", "pairedTrainerId", "sourceNodeId"];
  const LITERAL_FIELDS = ["vizVariant", "selectedTensorKey", "tensorKey"];
  return {
    nodes: nodes.map((n) => {
      const data = (n.data ?? {}) as Record<string, unknown>;
      const kept: Record<string, unknown> = {};
      for (const k of ID_REF_FIELDS) {
        if (k in data) kept[k] = typeof data[k] === "string" ? norm(data[k] as string) : data[k];
      }
      for (const k of LITERAL_FIELDS) {
        if (k in data) kept[k] = data[k];
      }
      return { id: norm(n.id), type: n.type, dataKeys: Object.keys(data).sort(), data: kept };
    }),
    edges: edges.map((e) => ({
      source: norm(e.source), target: norm(e.target),
      sourceHandle: e.sourceHandle ?? null, targetHandle: e.targetHandle ?? null,
    })),
  };
}

const N = (id: string, type: string, x: number, y: number, data: Record<string, unknown> = {}) =>
  ({ id, type, position: { x, y }, data }) as unknown as Node;
const E = (id: string, source: string, target: string, sourceHandle: string, targetHandle: string) =>
  ({ id, source, target, sourceHandle, targetHandle }) as unknown as Edge;

/** 图态/副作用 fixtures。 */
export function computeCases(): Record<string, unknown> {
  const out: Record<string, unknown> = {};

  // graphCase: observable_viz tensor 单扇入——已有边被替换
  {
    const nodes = [N("v", "observable_viz", 0, 0), N("a", "tensor_viz_0d", 0, 0), N("b", "tensor_viz_0d", 0, 0), N("t", "trainer", 0, 0)];
    const edges = [E("e0", "a", "v", "out_tensor", "tensor")];
    const res = applyCanvasConnection({ source: "b", target: "v", sourceHandle: "out_tensor", targetHandle: "tensor" } as Connection, nodes, edges);
    out.obsVizSingleFanIn = normalizeIds(res.nodes, res.edges, new Set(["v", "a", "b", "t"]));
  }
  // effectsCase: trainer loss ← mse_loss 触发 training_visualization spawn(以及 already-exists 去重)
  {
    const nodes = [N("l", "mse_loss", 0, 0), N("t", "trainer", 200, 0)];
    const first = applyCanvasConnection({ source: "l", target: "t", sourceHandle: "loss", targetHandle: "loss" } as Connection, nodes, []);
    out.lossSpawn = normalizeIds(first.nodes, first.edges, new Set(["l", "t"]));
    const again = applyCanvasConnection({ source: "l", target: "t", sourceHandle: "loss", targetHandle: "loss" } as Connection, first.nodes, first.edges);
    out.lossSpawnDedup = { nodeCount: again.nodes.length, edgeCount: again.edges.length };
  }
  // effectsCase: trainer observables ← kan_reg 触发 observable_viz spawn
  // kan_reg 曾是 OBSERVABLE_VIZ_SPAWN 手写兜底表中的唯一条目；转由 generated
  // 通道管理后，此 case 仍应保持逐字节一致。
  {
    const nodes = [N("o", "kan_reg", 0, 0), N("t", "trainer", 200, 0)];
    const res = applyCanvasConnection({ source: "o", target: "t", sourceHandle: "observables", targetHandle: "observables" } as Connection, nodes, []);
    out.observableSpawn = normalizeIds(res.nodes, res.edges, new Set(["o", "t"]));
    const again = applyCanvasConnection({ source: "o", target: "t", sourceHandle: "observables", targetHandle: "observables" } as Connection, res.nodes, res.edges);
    out.observableSpawnDedup = { nodeCount: again.nodes.length, edgeCount: again.edges.length };
  }
  // effectsCase: trainer observables ← observable_accuracy(generated 通道)触发 spawn
  // (issue #137 expected-changed:onConnect 必须走 spawnConfigFor,不得直查手写表)
  {
    const nodes = [N("o", "observable_accuracy", 0, 0), N("t", "trainer", 200, 0)];
    const res = applyCanvasConnection({ source: "o", target: "t", sourceHandle: "observables", targetHandle: "observables" } as Connection, nodes, []);
    out.generatedObservableSpawn = normalizeIds(res.nodes, res.edges, new Set(["o", "t"]));
    const again = applyCanvasConnection({ source: "o", target: "t", sourceHandle: "observables", targetHandle: "observables" } as Connection, res.nodes, res.edges);
    out.generatedObservableSpawnDedup = { nodeCount: again.nodes.length, edgeCount: again.edges.length };
  }
  // graphCase: ioMode 翻转代表节点
  {
    const flip = (srcType: string, mode: string, sh: string, tgtType: string, th: string, tgtMode = "model") =>
      isValidCanvasConnection(
        { source: "s", target: "t", sourceHandle: sh, targetHandle: th } as Connection,
        [N("s", srcType, 0, 0, { ioMode: mode }), N("t", tgtType, 100, 0, { ioMode: tgtMode })],
        [],
      );
    out.ioModeFlips = {
      mlpModelSocketModelMode: flip("mlp_model", "model", "model", "trainer", "model"),
      mlpStripOutIoMode: flip("mlp_model", "input-output", "tensor_out", "linear_layer", "tensor_in", "input-output"),
      mlpStripOutModelMode: flip("mlp_model", "model", "tensor_out", "linear_layer", "tensor_in", "input-output"),
      linearLayerIoTarget: flip("tensor_constant", "model", "tensor", "linear_layer", "tensor_in", "input-output"),
      linearLayerModelModeTarget: flip("tensor_constant", "model", "tensor", "linear_layer", "tensor_in", "model"),
      combinedBoundaryIo: flip("combined_model", "input-output", "tensor_boundary", "linear_layer", "tensor_in", "input-output"),
      combinedBoundaryModelMode: flip("combined_model", "model", "tensor_boundary", "linear_layer", "tensor_in", "input-output"),
    };
  }
  // effectsCase: optimizer_lr_schedule handle 重写
  {
    const nodes = [N("ls", "lr_schedule", 0, 0), N("op", "adam_optimizer", 200, 0)];
    const res = applyCanvasConnection({ source: "ls", target: "op", sourceHandle: "lr_schedule", targetHandle: "optimizer_lr_schedule" } as Connection, nodes, []);
    out.lrHandleRemap = normalizeIds(res.nodes, res.edges, new Set(["ls", "op"]));
  }
  // graphCase: tensor 多选触发 tensor_selector auto-spawn(training_visualization out_tensor_list → statistics)
  {
    const nodes = [
      N("tv", "training_visualization", 0, 0, { trainCurve: [1, 2], testCurve: [3, 4] }),
      N("st", "statistics", 200, 0),
      N("tr", "trainer", 0, 200),
    ];
    const edges = [E("e0", "tr", "tv", "loss_results", "tensor_list")];
    const res = applyCanvasConnection({ source: "tv", target: "st", sourceHandle: "out_tensor_list", targetHandle: "tensor" } as Connection, nodes, edges);
    out.tensorSelectorSpawn = normalizeIds(res.nodes, res.edges, new Set(["tv", "st", "tr"]));
  }
  // graphCase: table_viz 的 graph-dependent 分支(实打
  // sourceHandle 'tensor' 的 tableVizTensorConnectable 分支,空上游拒/有效上游放)
  {
    const rows = [1, 2, 3].map((i) => ({
      id: `r${i}`, rawSweep: `lr=${i}`, params: { lr: String(i) }, value: i * 0.1, valueLabel: "loss",
    }));
    const mk = (withUpstream: boolean) => {
      const nodes = [
        N("tb", "table_viz", 0, 0, { plotXParamKey: "lr" }),
        N("st", "regressor", 200, 0),
        ...(withUpstream ? [N("sw", "sweep_data_table", -200, 0, { rows })] : []),
      ];
      const edges = withUpstream ? [E("e0", "sw", "tb", "table", "table")] : [];
      return isValidCanvasConnection({ source: "tb", target: "st", sourceHandle: "tensor", targetHandle: "tensor" } as Connection, nodes, edges);
    };
    out.tableVizEmptyUpstream = mk(false);
    out.tableVizWithUpstream = mk(true);
  }
  // planAutoConnect: dataset/model/opt/loss 最近邻规划(确定性布局)
  {
    const nodes = [
      N("t", "trainer", 0, 0), N("d", "linear_dataset", -100, 0), N("m", "mlp_model", -100, 100),
      N("op", "adam_optimizer", -100, 200), N("l", "mse_loss", -100, 300), N("l2", "l1_reg", -100, 400),
      N("ob", "observable_gradient_norm", -100, 500),
    ];
    out.autoConnectPlan = planAutoConnectCanvas(nodes, []).map((c) => ({
      source: c.source, target: c.target, sourceHandle: c.sourceHandle, targetHandle: c.targetHandle,
    }));
    // 去重:已有 model 边不得重复规划
    const existing = [E("e0", "m", "t", "model", "model")];
    out.autoConnectPlanWithExisting = planAutoConnectCanvas(nodes, existing).map((c) => ({
      source: c.source, target: c.target, sourceHandle: c.sourceHandle, targetHandle: c.targetHandle,
    }));
  }
  // hostile: missing source/target、unknown type
  {
    const nodes = [N("a", "mlp_model", 0, 0)];
    out.missingTarget = isValidCanvasConnection({ source: "a", target: "ghost", sourceHandle: "model", targetHandle: "model" } as Connection, nodes, []);
    out.unknownType = isValidCanvasConnection(
      { source: "a", target: "b", sourceHandle: "weird", targetHandle: "weird" } as Connection,
      [...nodes, N("b", "totally_unknown_node", 1, 1)], [],
    );
  }
  return out;
}

export function computeGolden() {
  return { matrix: computeMatrix(), cases: computeCases() };
}

export function writeGolden(): void {
  const g = computeGolden();
  writeFileSync(GOLDEN_PATH, JSON.stringify(g, null, 1) + "\n");
  console.log("wrote golden:", g.matrix.total, "matrix probes,", g.matrix.trueCount, "true,", "sha", g.matrix.sha256.slice(0, 12));
}

export function checkGolden(): void {
  const g = computeGolden();
  const committed = JSON.parse(readFileSync(GOLDEN_PATH, "utf8"));
  if (JSON.stringify(g, null, 1) !== JSON.stringify(committed, null, 1)) {
    throw new Error("connection golden MISMATCH");
  }
  console.log("connection golden OK:", g.matrix.total, "matrix probes,", g.matrix.trueCount, "true");
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  if (process.argv.includes("--check")) checkGolden(); else writeGolden();
}
