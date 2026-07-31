#!/usr/bin/env npx tsx
/**
 * Seed grokking_physics_demo template + user observables (100 random algebra drafts).
 * Run from repo root: npx tsx scripts/seed_grokking_physics_demo.ts
 */
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  buildRandomObservableDrafts,
  DEFAULT_RANDOM_GENERATION_PREFERENCES,
  familyPatternFromRepresentationId,
  familyPatternFromTensorName,
  formatReductionPreview,
  globalFlattenLabelBase,
  globalFlattenRepresentationKind,
  type AxisReductionDraft,
  type ObservableFlattenMode,
  type ObservableSource,
  type ObservableTensorScope,
  type RandomObservableDraft,
  type RepresentationEntry,
} from "../frontend/src/observables/observableAlgebra.ts";

const REPO = path.resolve(import.meta.dirname, "..");
const OBS_ID_PREFIX = "grokking-demo-obs-";
const TOTAL_OBS_COUNT = 100;
const OBS_IDS = Array.from({ length: TOTAL_OBS_COUNT }, (_, i) => {
  const n = String(i + 1).padStart(3, "0");
  return `${OBS_ID_PREFIX}${n}`;
});
const TEMPLATE_ID = "f3a8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c";
const MODEL_ID = "grok-model-0";
const TRAINER_ID = "grok-trainer-0";
const SEED = 8674;
const GEN_OPTS = {
  preferences: {
    ...DEFAULT_RANDOM_GENERATION_PREFERENCES,
    svEntropy: "none" as const,
    allMatchingLayers: "all" as const,
  },
} as const;

type ModelProbeSpecs = {
  weights: Record<string, { shape: number[] }>;
  representations: RepresentationEntry[];
};

function fetchModelProbeSpecs(): ModelProbeSpecs {
  const out = execSync("python scripts/grokking_model_probe_specs.py", { cwd: REPO, encoding: "utf-8" });
  return JSON.parse(out.trim()) as ModelProbeSpecs;
}

function flatOp(reductions: AxisReductionDraft[]): string {
  return reductions[0]?.op ?? "l2_norm";
}

function arrayLoaderExpr(observableSource: ObservableSource, subjectId: string): string {
  if (observableSource === "representation") {
    return `activation_representation_as_numpy(${JSON.stringify(subjectId)})`;
  }
  return `named_parameter_as_numpy(${JSON.stringify(subjectId)})`;
}

function formatAlgebraDefinitionCode(draft: RandomObservableDraft): string {
  const { tensorName, reductions, flattenMode, observableSource, representationId } = draft;
  const subjectId =
    observableSource === "representation" ? (representationId ?? tensorName) : tensorName;
  const loader = arrayLoaderExpr(observableSource, subjectId);
  const op = flatOp(reductions);
  if (flattenMode === "sv_entropy") {
    return [`arr = ${loader}`, "metric = float(singular_value_entropy(np.asarray(arr)))"].join("\n");
  }
  if (flattenMode === "global") {
    return [
      "parts = []",
      `# global flatten over matching tensors for ${subjectId}`,
      `parts.append(np.asarray(${loader}).reshape(-1))`,
      "arr = np.concatenate(parts)",
      `metric = float(flat_stat_reduce(arr, ${JSON.stringify(op)}))`,
    ].join("\n");
  }
  if (flattenMode === "local") {
    return [
      `arr = ${loader}`,
      "arr = np.asarray(arr).reshape(-1)",
      `metric = float(flat_stat_reduce(arr, ${JSON.stringify(op)}))`,
    ].join("\n");
  }
  const ordered = [...reductions].sort((a, b) => a.axisIndex - b.axisIndex);
  const lines = [`arr = ${loader}`];
  for (const spec of ordered) {
    lines.push(`arr = reduce_along_axis(arr, 0, ${JSON.stringify(spec.op)})`);
  }
  lines.push("metric = float(np.asarray(arr).reshape(-1)[0])");
  return lines.join("\n");
}

function humanChain(draft: RandomObservableDraft): string {
  if (draft.observableSource === "representation") {
    const repLabel =
      draft.tensorScope === "all_matching"
        ? familyPatternFromRepresentationId(draft.representationId ?? draft.tensorName)
        : (draft.representationId ?? draft.tensorName);
    return formatReductionPreview(`rep ${repLabel}`, draft.tensorShape, draft.reductions, draft.flattenMode);
  }
  const pat =
    draft.tensorScope === "all_matching"
      ? familyPatternFromTensorName(draft.tensorName)
      : draft.tensorName;
  return formatReductionPreview(pat, draft.tensorShape, draft.reductions, draft.flattenMode);
}

type UserObsItem = {
  id: string;
  label: string;
  definition_kind: string;
  source_model_node_id: string;
  tensor_name: string;
  tensor_shape: number[];
  tensor_scope: string;
  tensor_pattern: string;
  flatten_mode: string;
  observable_source: string;
  representation_id: string;
  layer_index: number;
  layer_io: string;
  reductions: { axis_index: number; axis_label: string; op: string }[];
  definition_code: string;
  human_chain: string;
  created_at: string;
  tensor_viz_node_id: string;
  tensor_selector_node_id: string;
};

function draftToUserObsItem(draft: RandomObservableDraft, id: string, now: string): UserObsItem {
  const isRep = draft.observableSource === "representation";
  const subjectId = isRep ? (draft.representationId ?? draft.tensorName) : draft.tensorName;
  return {
    id,
    label: draft.label,
    definition_kind: "algebra",
    source_model_node_id: MODEL_ID,
    tensor_name: subjectId,
    tensor_shape: draft.tensorShape,
    tensor_scope: draft.tensorScope,
    tensor_pattern: isRep
      ? draft.flattenMode === "global"
        ? globalFlattenRepresentationKind(subjectId)
        : draft.tensorScope === "all_matching"
          ? familyPatternFromRepresentationId(subjectId)
          : subjectId
      : draft.flattenMode === "global"
        ? globalFlattenLabelBase(draft.tensorName)
        : draft.tensorScope === "all_matching"
          ? familyPatternFromTensorName(draft.tensorName)
          : draft.tensorName,
    flatten_mode: draft.flattenMode,
    observable_source: draft.observableSource,
    representation_id: isRep ? (draft.representationId ?? draft.tensorName) : "",
    layer_index: draft.layerIndex ?? 0,
    layer_io: draft.layerIo ?? "",
    reductions: draft.reductions.map((r) => ({
      axis_index: r.axisIndex,
      axis_label: r.axisLabel,
      op: r.op,
    })),
    definition_code: formatAlgebraDefinitionCode(draft),
    human_chain: humanChain(draft),
    created_at: now,
    tensor_viz_node_id: "",
    tensor_selector_node_id: "",
  };
}

function buildUserObservables(): UserObsItem[] {
  const { weights, representations } = fetchModelProbeSpecs();
  const tensorNames = Object.keys(weights).sort();
  const drafts = buildRandomObservableDrafts(
    TOTAL_OBS_COUNT,
    SEED,
    tensorNames,
    weights,
    representations,
    GEN_OPTS,
  );
  const now = new Date().toISOString();
  return drafts.map((d, i) => draftToUserObsItem(d, OBS_IDS[i]!, now));
}

function node(id: string, type: string, x: number, y: number, data: Record<string, unknown>) {
  return {
    id,
    type,
    position: { x, y },
    data,
    parentId: null,
    extent: null,
    hidden: null,
    style: null,
  };
}

function edge(id: string, source: string, target: string, sh: string, th: string) {
  return { id, source, target, sourceHandle: sh, targetHandle: th };
}

function buildTemplate(obsItems: UserObsItem[]) {
  const nodes: ReturnType<typeof node>[] = [];
  const edges: ReturnType<typeof edge>[] = [];

  nodes.push(
    node("grok-dataset-0", "modular_addition_dataset", 80, 120, {
      modulus: 59,
      trainFraction: 0.8,
      seed: 8674,
      samplingMode: "fixed",
      instanceTitle: "Modular addition dataset 0",
    }),
    node(MODEL_ID, "mlp_token_model", 420, 120, {
      vocabSize: 59,
      embedDim: 32,
      tokensPerInput: 2,
      depth: 2,
      width: 64,
      numExperts: 4,
      activation: "relu",
      tieWeights: "yes",
      seed: 9196,
      instanceTitle: "MLP_token model 0",
    }),
    node("grok-adam-0", "adam_optimizer", 80, 400, {
      learningRate: 0.003,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      weightDecay: 0.001,
      instanceTitle: "Adam 0",
    }),
    node("grok-ce-0", "cross_entropy_loss", 420, 400, {
      lossScale: 1,
      instanceTitle: "Cross-entropy loss 0",
    }),
    node(TRAINER_ID, "trainer", 760, 260, {
      trainingSteps: 2000,
      logFrequency: 10,
      disableExtraObservables: false,
      instanceTitle: "Trainer 0",
    }),
    node("grok-train-viz-0", "training_visualization", 1080, 260, {
      instanceTitle: "Training viz 0",
    }),
    node("grok-acc-0", "observable_accuracy", 420, 620, {
      instanceTitle: "Observable Accuracy 0",
    }),
    node("grok-acc-viz-0", "observable_viz", 1080, 620, {
      pairedObservableId: "grok-acc-0",
      pairedTrainerId: TRAINER_ID,
      vizVariant: "accuracy",
      instanceTitle: "Accuracy viz 0",
    }),
    node("grok-wl2-0", "observable_weight_l2", 420, 720, {
      normAggregation: "global",
      instanceTitle: "Weight L2 0",
    }),
    node("grok-wl2-viz-0", "observable_viz", 1080, 720, {
      pairedObservableId: "grok-wl2-0",
      pairedTrainerId: TRAINER_ID,
      vizVariant: "weight_l2",
      instanceTitle: "Weight L2 viz 0",
    }),
    node("grok-comment-0", "comment", 80, 560, {
      instanceTitle: "Grokking physics demo",
      text:
        "Train → CurveStarer → pick **Target** (e.g. test accuracy), **Objective**, **Threshold** → **Propose speed up tricks**.\n\n" +
        `Includes Weight L2 + ${TOTAL_OBS_COUNT} random algebra observables (weight + representation, seed ${SEED}, no SVD entropy). ` +
        "Use **Disable extra observables** on Trainer for fast debug runs (loss + accuracy only).",
    }),
  );

  edges.push(
    edge("e-grok-ds", "grok-dataset-0", TRAINER_ID, "dataset", "dataset"),
    edge("e-grok-model", MODEL_ID, TRAINER_ID, "model", "model"),
    edge("e-grok-opt", "grok-adam-0", TRAINER_ID, "optimizer", "optimizer"),
    edge("e-grok-loss", "grok-ce-0", TRAINER_ID, "loss", "loss"),
    edge("e-grok-tv", TRAINER_ID, "grok-train-viz-0", "loss_results", "tensor_list"),
    edge("e-grok-acc", "grok-acc-0", TRAINER_ID, "observables", "observables"),
    edge("e-grok-acc-viz", TRAINER_ID, "grok-acc-viz-0", "observable_results", "tensor"),
    edge("e-grok-wl2", "grok-wl2-0", TRAINER_ID, "observables", "observables"),
    edge("e-grok-wl2-viz", TRAINER_ID, "grok-wl2-viz-0", "observable_results", "tensor"),
  );

  const obsCols = 10;
  const obsColW = 280;
  const obsRowH = 152;
  const obsOriginX = -240;
  const obsOriginY = 820;

  const vizCols = 10;
  const vizColW = 280;
  const vizRowH = 280;
  const vizOriginX = 1180;
  const vizOriginY = 820;

  obsItems.forEach((item, i) => {
    const obsId = `grok-user-obs-${i}`;
    const vizId = `grok-user-viz-${i}`;
    const obsCol = i % obsCols;
    const obsRow = Math.floor(i / obsCols);
    const vizCol = i % vizCols;
    const vizRow = Math.floor(i / vizCols);
    nodes.push(
      node(obsId, "observable_user", obsOriginX + obsCol * obsColW, obsOriginY + obsRow * obsRowH, {
        userObservableId: item.id,
        label: item.label,
        tensorVizNodeId: "",
        tensorSelectorNodeId: "",
        instanceTitle: item.label,
      }),
      node(vizId, "observable_viz", vizOriginX + vizCol * vizColW, vizOriginY + vizRow * vizRowH, {
        pairedObservableId: obsId,
        pairedTrainerId: TRAINER_ID,
        observableName: item.label,
        instanceTitle: `${item.label} viz`,
      }),
    );
    edges.push(
      edge(`e-grok-obs-${i}`, obsId, TRAINER_ID, "observables", "observables"),
      edge(`e-grok-obs-viz-${i}`, TRAINER_ID, vizId, "observable_results", "tensor"),
    );
  });

  return {
    id: TEMPLATE_ID,
    name: "grokking_physics_demo",
    tier: "small",
    document: { version: 1, nodes, edges, viewport: { x: 120, y: -40, zoom: 0.62 } },
    savedAt: Date.now(),
    libraryOrigin: null,
  };
}

function mergeUserObservables(newItems: UserObsItem[]) {
  const p = path.join(REPO, "data/user_observables.json");
  const raw = JSON.parse(fs.readFileSync(p, "utf-8")) as { version: number; items: UserObsItem[] };
  raw.items = raw.items.filter((it) => !String(it.id).startsWith(OBS_ID_PREFIX));
  raw.items.push(...newItems);
  fs.writeFileSync(p, `${JSON.stringify(raw, null, 2)}\n`);
}

function writeBundledObservables(newItems: UserObsItem[]) {
  const bundledDir = path.join(REPO, "data/bundled");
  fs.mkdirSync(bundledDir, { recursive: true });
  const bundledPath = path.join(bundledDir, "grokking_demo_user_observables.json");
  fs.writeFileSync(
    bundledPath,
    `${JSON.stringify({ version: 1, items: newItems }, null, 2)}\n`,
  );
}

function main() {
  const obsItems = buildUserObservables();
  const repCount = obsItems.filter((o) => o.observable_source === "representation").length;
  mergeUserObservables(obsItems);
  writeBundledObservables(obsItems);
  const template = buildTemplate(obsItems);
  const outPath = path.join(REPO, "data/graph_library/templates", `${TEMPLATE_ID}.json`);
  fs.writeFileSync(outPath, `${JSON.stringify(template, null, 2)}\n`);
  console.log(
    `Wrote ${obsItems.length} user observables (${repCount} representation) and template ${outPath}`,
  );
}

main();
