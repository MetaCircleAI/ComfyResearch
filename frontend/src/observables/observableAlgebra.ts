/** Reduction operators for observable algebra (per-axis or 1D-after-flatten). */
export type ObservableReductionOp =
  | "mean"
  | "median"
  | "max"
  | "min"
  | "std"
  | "l2_norm"
  | "l1_norm"
  | "entropy";

/** How tensors are collapsed before applying reductions. */
export type ObservableFlattenMode = "none" | "local" | "global" | "sv_entropy";

export type ObservableSource = "weight" | "representation";

export type ObservableTensorScope = "single" | "all_matching";

export type RepresentationEntry = {
  layer_index: number;
  io: string;
  representation_id: string;
  module_name?: string;
  /** ``-1`` on axis 0 means batch size (resolved at training time). */
  shape: number[];
  label: string;
};

/** Axis 0 placeholder for representation batch (actual size fixed during training). */
export const BATCH_AXIS_PLACEHOLDER = -1;

export function formatTensorShapeDim(dim: number): string {
  return dim < 0 ? "batch" : String(dim);
}

export function formatTensorShapeBracket(shape: number[]): string {
  if (!shape.length) return "[?]";
  return `[${shape.map(formatTensorShapeDim).join(", ")}]`;
}

export function formatTensorShapeTimes(shape: number[]): string {
  if (!shape.length) return "";
  return `[${shape.map(formatTensorShapeDim).join(" × ")}]`;
}

export type AxisReductionDraft = {
  axisIndex: number;
  axisLabel: string;
  op: ObservableReductionOp;
};

export type AlgebraObservableItem = {
  id: string;
  label: string;
  definition_kind?: string;
  source_model_node_id?: string;
  tensor_name?: string;
  tensor_shape?: number[];
  tensor_scope?: ObservableTensorScope;
  tensor_pattern?: string;
  flatten_mode?: ObservableFlattenMode;
  observable_source?: ObservableSource;
  representation_id?: string;
  layer_index?: number;
  layer_io?: string;
  human_chain?: string;
  tensor_viz_node_id?: string;
  tensor_selector_node_id?: string;
};

export const OBSERVABLE_REDUCTION_OPTIONS: { id: ObservableReductionOp; label: string }[] = [
  { id: "mean", label: "Mean" },
  { id: "median", label: "Median" },
  { id: "max", label: "Max" },
  { id: "min", label: "Min" },
  { id: "std", label: "Std dev" },
  { id: "l2_norm", label: "L2 norm" },
  { id: "l1_norm", label: "L1 norm" },
  { id: "entropy", label: "Entropy" },
];

export const OBSERVABLE_FLATTEN_OPTIONS: { id: ObservableFlattenMode; label: string; hint: string }[] = [
  {
    id: "none",
    label: "No flatten",
    hint: "Treat as a high-rank tensor; pick one reduction per axis.",
  },
  {
    id: "local",
    label: "Local flatten",
    hint: "Flatten each tensor to a 1D vector, then apply one 1D reduction (per-module norm).",
  },
  {
    id: "global",
    label: "Global flatten",
    hint: "Concatenate matching tensors into one vector, then apply one 1D reduction (overall norm).",
  },
  {
    id: "sv_entropy",
    label: "Singular value entropy",
    hint: "2D matrix: SVD singular values → Shannon entropy (batch × features as a matrix).",
  },
];

export function suggestAxisLabels(shape: number[], tensorName = ""): string[] {
  const rank = shape.length;
  if (rank === 0) return [];
  const name = tensorName.toLowerCase();
  if (rank === 2 && shape[0]! >= 0 && name.includes("weight")) return ["out", "in"];
  if (rank === 3 && shape[0]! >= 0 && name.includes("weight")) return ["out", "in", "kernel"];
  if (rank === 4 && shape[0]! >= 0 && name.includes("weight")) return ["out", "in", "h", "w"];
  return shape.map((dim, i) => {
    if (i === 0 && dim < 0) return "batch";
    return `dim${i}`;
  });
}

export const OBSERVABLE_SCOPE_OPTIONS: { id: ObservableTensorScope; label: string }[] = [
  { id: "single", label: "This tensor only" },
  { id: "all_matching", label: "All matching layers" },
];

/** Scope picker labels; global flatten aggregates matching tensors into one scalar ("Everything"). */
export function observableScopeOptionsForFlattenMode(
  flattenMode: ObservableFlattenMode,
): { id: ObservableTensorScope; label: string }[] {
  if (flattenMode === "global") {
    return [
      { id: "single", label: "This tensor only" },
      { id: "all_matching", label: "Everything (one global scalar)" },
    ];
  }
  return OBSERVABLE_SCOPE_OPTIONS;
}

export function familyPatternFromTensorName(tensorName: string): string {
  const trimmed = tensorName.trim();
  const bodyM = /^(?:body\.)?\d+\.(.+)$/.exec(trimmed);
  if (bodyM) return `*.${bodyM[1]}`;
  const m = /^(\d+)\.(.+)$/.exec(trimmed);
  if (m) return `*.${m[2]}`;
  return trimmed || "tensor";
}

export function representationIoFromId(representationId: string): string | null {
  const idx = representationId.indexOf("::");
  if (idx < 0) return null;
  const io = representationId.slice(idx + 2).trim();
  return io || null;
}

/** IO family key without wildcard (for global flatten names). */
export function representationAllMatchingNameBase(
  representationId: string,
  entry?: RepresentationEntry | null,
): string {
  if (entry?.io?.trim()) return entry.io.trim().toLowerCase();
  const label = entry?.label?.trim();
  if (label) {
    const tail = label.split(/\s+/).pop();
    if (tail) return tail.toLowerCase();
  }
  const io = representationIoFromId(representationId);
  if (io) return io.toLowerCase();
  const hm = /^h\d+_(.+)$/.exec(representationId.trim());
  if (hm) return `h*_${hm[1]}`;
  return representationId.trim() || "rep";
}

/** Label base for all-matching scope — ``*.input`` / ``*.output`` (multiple tensors). */
export function representationAllMatchingLabelBase(
  representationId: string,
  entry?: RepresentationEntry | null,
): string {
  const kind = representationAllMatchingNameBase(representationId, entry);
  if (kind.startsWith("h*_")) return kind;
  return `*.${kind}`;
}

export function familyPatternFromRepresentationId(representationId: string): string {
  const trimmed = representationId.trim();
  if (!trimmed) return "representation";
  const io = representationIoFromId(trimmed);
  if (io) return `*.${io.toLowerCase()}`;
  const m = /^h\d+_(.+)$/.exec(trimmed);
  if (m) return `h*_${m[1]}`;
  return trimmed;
}

export function matchingRepresentationIds(
  allIds: string[],
  selected: string,
  scope: ObservableTensorScope,
): string[] {
  const sel = selected.trim();
  if (!sel) return [];
  if (scope === "single") return allIds.includes(sel) ? [sel] : [];
  const selIo = representationIoFromId(sel);
  if (selIo) {
    const matched = allIds.filter((n) => representationIoFromId(n) === selIo).sort();
    return matched.length > 0 ? matched : allIds.includes(sel) ? [sel] : [];
  }
  const hm = /^h\d+_(.+)$/.exec(sel);
  if (hm) {
    const suf = hm[1]!;
    const re = new RegExp(`^h\\d+_${suf.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
    const matched = allIds.filter((n) => re.test(n)).sort();
    return matched.length > 0 ? matched : allIds.includes(sel) ? [sel] : [];
  }
  return allIds.includes(sel) ? [sel] : [];
}

export function canUseAllMatchingScopeForRepresentation(representationId: string, allIds: string[]): boolean {
  return matchingRepresentationIds(allIds, representationId, "all_matching").length > 1;
}

export function matchingTensorNames(
  allNames: string[],
  selected: string,
  scope: ObservableTensorScope,
): string[] {
  const sel = selected.trim();
  if (!sel) return [];
  if (scope === "single") return allNames.includes(sel) ? [sel] : [];
  const bodyM = /^(?:body\.)?\d+\.(.+)$/.exec(sel);
  if (bodyM) {
    const suffix = bodyM[1]!;
    const reNum = new RegExp(`^\\d+\\.${suffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
    const reBody = new RegExp(`^body\\.\\d+\\.${suffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
    const matched = allNames
      .filter((n) => reNum.test(n) || reBody.test(n) || n.endsWith(`.${suffix}`))
      .sort();
    return matched.length > 0 ? matched : allNames.includes(sel) ? [sel] : [];
  }
  const m = /^(\d+)\.(.+)$/.exec(sel);
  if (!m) return allNames.includes(sel) ? [sel] : [];
  const suffix = m[2]!;
  const re = new RegExp(`^\\d+\\.${suffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
  const matched = allNames.filter((n) => re.test(n)).sort();
  return matched.length > 0 ? matched : allNames.includes(sel) ? [sel] : [];
}

export function defaultReductionDrafts(shape: number[], tensorName = ""): AxisReductionDraft[] {
  const labels = suggestAxisLabels(shape, tensorName);
  return labels.map((axisLabel, axisIndex) => ({
    axisIndex,
    axisLabel,
    op: "mean",
  }));
}

export function defaultFlatReductionDraft(op: ObservableReductionOp = "l2_norm"): AxisReductionDraft {
  return { axisIndex: 0, axisLabel: "flat", op };
}

export function defaultSvEntropyReductionDraft(): AxisReductionDraft {
  return { axisIndex: 0, axisLabel: "sv", op: "entropy" };
}

export function isMatrixRepresentation(shape: number[]): boolean {
  const rank = shape.length;
  return rank >= 2 || (rank === 1 && shape[0]! > 1);
}

export function representationLabelBase(entry: RepresentationEntry): string {
  return entry.label.trim().replace(/\s+/g, "_").toLowerCase() || entry.representation_id;
}

export function globalFlattenLabelBase(tensorName: string): string {
  const trimmed = tensorName.trim() || "tensor";
  const lastDot = trimmed.lastIndexOf(".");
  return lastDot >= 0 ? trimmed.slice(lastDot + 1) : trimmed;
}

/** Strip wildcard family prefix from representation patterns for global-flatten names. */
export function globalFlattenRepresentationKind(
  representationId: string,
  entry?: RepresentationEntry | null,
): string {
  return representationAllMatchingNameBase(representationId, entry);
}

export function autoObservableLabel(
  subjectId: string,
  reductions: AxisReductionDraft[],
  scope: ObservableTensorScope = "single",
  flattenMode: ObservableFlattenMode = "none",
  observableSource: ObservableSource = "weight",
  representationEntry?: RepresentationEntry | null,
): string {
  if (observableSource === "representation") {
    const base =
      scope === "all_matching"
        ? representationAllMatchingLabelBase(subjectId, representationEntry)
        : representationEntry
          ? representationLabelBase(representationEntry)
          : subjectId.trim() || "rep";
    if (flattenMode === "sv_entropy") return `${base}.sv_entropy`;
    if (flattenMode === "global") {
      const op = reductions[0]?.op ?? "l2_norm";
      const kind =
        scope === "all_matching"
          ? globalFlattenRepresentationKind(subjectId, representationEntry)
          : base;
      return `${kind}.${op}_global`;
    }
    if (flattenMode === "local") {
      const op = reductions[0]?.op ?? "l2_norm";
      return `${base}.${op}_flat`;
    }
    const ordered = [...reductions].sort((a, b) => a.axisIndex - b.axisIndex);
    const suffix = ordered.map((r) => `${r.op}_${r.axisLabel}`).join(".");
    return suffix ? `${base}.${suffix}` : base;
  }
  if (flattenMode === "global") {
    const base = globalFlattenLabelBase(subjectId);
    const op = reductions[0]?.op ?? "l2_norm";
    return `${base}.${op}_global`;
  }
  const base = scope === "all_matching" ? familyPatternFromTensorName(subjectId) : subjectId.trim() || "tensor";
  if (flattenMode === "local") {
    const op = reductions[0]?.op ?? "l2_norm";
    return `${base}.${op}_flat`;
  }
  const ordered = [...reductions].sort((a, b) => a.axisIndex - b.axisIndex);
  const suffix = ordered.map((r) => `${r.op}_${r.axisLabel}`).join(".");
  return suffix ? `${base}.${suffix}` : base;
}

export function formatReductionPreview(
  subjectLabel: string,
  shape: number[],
  reductions: AxisReductionDraft[],
  flattenMode: ObservableFlattenMode = "none",
): string {
  const shapeS = formatTensorShapeTimes(shape);
  if (flattenMode === "sv_entropy") {
    return `${subjectLabel}${shapeS ? ` ${shapeS}` : ""} → SVD → singular value entropy`;
  }
  if (flattenMode === "local") {
    const op = reductions[0]?.op ?? "l2_norm";
    return `${subjectLabel}${shapeS ? ` ${shapeS}` : ""} → flatten → ${op}(flat)`;
  }
  if (flattenMode === "global") {
    const op = reductions[0]?.op ?? "l2_norm";
    return `${globalFlattenLabelBase(subjectLabel)} → concat → flatten → ${op}(global)`;
  }
  const ordered = [...reductions].sort((a, b) => a.axisIndex - b.axisIndex);
  const chain = ordered.map((r) => `${r.op}(${r.axisLabel})`).join(" → ");
  return `${subjectLabel}${shapeS ? ` ${shapeS}` : ""} → ${chain}`;
}


/** True when ``all_matching`` spans more than one layer tensor. */
export function canUseAllMatchingScope(tensorName: string, allNames: string[]): boolean {
  return matchingTensorNames(allNames, tensorName, "all_matching").length > 1;
}

export type RandomGenerationPreference = "none" | "some" | "all";

export type RandomGenerationPreferences = {
  allMatchingLayers: RandomGenerationPreference;
  svEntropy: RandomGenerationPreference;
  mean: RandomGenerationPreference;
  median: RandomGenerationPreference;
  min: RandomGenerationPreference;
  max: RandomGenerationPreference;
  std: RandomGenerationPreference;
  l1Norm: RandomGenerationPreference;
  l2Norm: RandomGenerationPreference;
  entropy: RandomGenerationPreference;
};

export const DEFAULT_RANDOM_GENERATION_PREFERENCES: RandomGenerationPreferences = {
  allMatchingLayers: "some",
  svEntropy: "none",
  mean: "some",
  median: "none",
  min: "none",
  max: "none",
  std: "some",
  l1Norm: "some",
  l2Norm: "some",
  entropy: "some",
};

export const RANDOM_GENERATION_PREFERENCE_ROWS: {
  key: keyof RandomGenerationPreferences;
  label: string;
}[] = [
  { key: "allMatchingLayers", label: "All matching layers" },
  { key: "svEntropy", label: "SVD entropy" },
  { key: "mean", label: "Mean" },
  { key: "median", label: "Median" },
  { key: "min", label: "Min" },
  { key: "max", label: "Max" },
  { key: "std", label: "Std dev" },
  { key: "l1Norm", label: "L1 norm" },
  { key: "l2Norm", label: "L2 norm" },
  { key: "entropy", label: "Entropy" },
];

const REDUCTION_OP_TO_PREF_KEY: Record<ObservableReductionOp, keyof RandomGenerationPreferences> = {
  mean: "mean",
  median: "median",
  min: "min",
  max: "max",
  std: "std",
  l1_norm: "l1Norm",
  l2_norm: "l2Norm",
  entropy: "entropy",
};

export function resolveRandomGenerationPreferences(
  options: RandomObservableGenerateOptions = {},
): RandomGenerationPreferences {
  const prefs: RandomGenerationPreferences = { ...DEFAULT_RANDOM_GENERATION_PREFERENCES, ...options.preferences };
  if (options.disableSvdEntropy === true) prefs.svEntropy = "none";
  else if (options.disableSvdEntropy === false && options.preferences?.svEntropy === undefined) {
    prefs.svEntropy = "some";
  }
  if (options.preferAllMatching === true) prefs.allMatchingLayers = "all";
  else if (options.preferAllMatching === false && options.preferences?.allMatchingLayers === undefined) {
    prefs.allMatchingLayers = "none";
  }
  return prefs;
}

export function pickScopeWithPreference(
  canAllMatching: boolean,
  pref: RandomGenerationPreference,
  rng: SeededRng,
): ObservableTensorScope {
  if (!canAllMatching) return "single";
  if (pref === "all") return "all_matching";
  if (pref === "none") return "single";
  return rng.int(2) === 0 ? "single" : "all_matching";
}

export function pickRandomReductionOp(
  rng: SeededRng,
  prefs: RandomGenerationPreferences,
): ObservableReductionOp {
  const allPool: ObservableReductionOp[] = [];
  const somePool: ObservableReductionOp[] = [];
  for (const opt of OBSERVABLE_REDUCTION_OPTIONS) {
    const p = prefs[REDUCTION_OP_TO_PREF_KEY[opt.id]];
    if (p === "all") allPool.push(opt.id);
    else if (p === "some") somePool.push(opt.id);
  }
  if (allPool.length > 0) return allPool[rng.int(allPool.length)]!;
  if (somePool.length > 0) return somePool[rng.int(somePool.length)]!;
  return "mean";
}

export function pickRandomTensorScope(
  tensorName: string,
  allNames: string[],
  flattenMode: ObservableFlattenMode,
  allMatchingPref: RandomGenerationPreference,
  rng: SeededRng,
): ObservableTensorScope {
  const can = canUseAllMatchingScope(tensorName, allNames);
  if (flattenMode === "global") return can ? "all_matching" : "single";
  return pickScopeWithPreference(can, allMatchingPref, rng);
}

export function pickRandomRepresentationScope(
  representationId: string,
  allIds: string[],
  allMatchingPref: RandomGenerationPreference,
  rng: SeededRng,
): ObservableTensorScope {
  const can = canUseAllMatchingScopeForRepresentation(representationId, allIds);
  return pickScopeWithPreference(can, allMatchingPref, rng);
}

export type SeededRng = {
  /** Uniform float in [0, 1). */
  next: () => number;
  /** Uniform integer in [0, maxExclusive). */
  int: (maxExclusive: number) => number;
};

export function parseObservableRandomSeed(raw: string): { ok: true; seed: number } | { ok: false; error: string } {
  const t = raw.trim();
  if (t === "") return { ok: false, error: "Enter a seed." };
  const n = Number(t);
  if (!Number.isFinite(n)) return { ok: false, error: "Seed must be a finite number." };
  return { ok: true, seed: Math.trunc(n) >>> 0 };
}

export function createSeededRng(seed: number): SeededRng {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    int(maxExclusive: number) {
      if (maxExclusive <= 0) return 0;
      return Math.floor(next() * maxExclusive);
    },
  };
}

export function randomReductionDrafts(
  shape: number[],
  tensorName: string,
  rng: SeededRng,
  prefs: RandomGenerationPreferences,
): AxisReductionDraft[] {
  const labels = suggestAxisLabels(shape, tensorName);
  return labels.map((axisLabel, axisIndex) => ({
    axisIndex,
    axisLabel,
    op: pickRandomReductionOp(rng, prefs),
  }));
}

export function randomFlatReductionDraft(
  rng: SeededRng,
  prefs: RandomGenerationPreferences,
): AxisReductionDraft {
  return defaultFlatReductionDraft(pickRandomReductionOp(rng, prefs));
}

export type RandomObservableGenerateOptions = {
  preferences?: Partial<RandomGenerationPreferences>;
  /** @deprecated use ``preferences.allMatchingLayers === "all"`` */
  preferAllMatching?: boolean;
  /** @deprecated use ``preferences.svEntropy === "none"`` */
  disableSvdEntropy?: boolean;
};

export function randomFlattenMode(rng: SeededRng, svEntropyPref: RandomGenerationPreference): ObservableFlattenMode {
  if (svEntropyPref === "all") return "sv_entropy";
  const pool = OBSERVABLE_FLATTEN_OPTIONS.filter(
    (o) => svEntropyPref !== "none" || o.id !== "sv_entropy",
  );
  if (pool.length === 0) return "none";
  return pool[rng.int(pool.length)]!.id;
}

export type RandomObservableDraft = {
  observableSource: ObservableSource;
  tensorName: string;
  tensorShape: number[];
  tensorScope: ObservableTensorScope;
  flattenMode: ObservableFlattenMode;
  reductions: AxisReductionDraft[];
  label: string;
  representationId?: string;
  layerIndex?: number;
  layerIo?: string;
};

export function buildRandomObservableDraft(
  tensorNames: string[],
  specs: Record<string, { shape: number[] }>,
  rng: SeededRng,
  repEntries: RepresentationEntry[] = [],
  options: RandomObservableGenerateOptions = {},
): RandomObservableDraft | null {
  const prefs = resolveRandomGenerationPreferences(options);
  const useRep = repEntries.length > 0 && rng.int(3) === 0;
  const repIds = repEntries.map((e) => e.representation_id);
  if (useRep) {
    const entry = repEntries[rng.int(repEntries.length)]!;
    let flattenMode = randomFlattenMode(rng, prefs.svEntropy);
    if (flattenMode === "global") flattenMode = "local";
    if (flattenMode === "sv_entropy" && !isMatrixRepresentation(entry.shape)) {
      flattenMode = randomFlattenMode(rng, "none");
    }
    const reductions =
      flattenMode === "none"
        ? randomReductionDrafts(entry.shape, entry.representation_id, rng, prefs)
        : flattenMode === "sv_entropy"
          ? [defaultSvEntropyReductionDraft()]
          : [randomFlatReductionDraft(rng, prefs)];
    const tensorScope = pickRandomRepresentationScope(
      entry.representation_id,
      repIds,
      prefs.allMatchingLayers,
      rng,
    );
    const label = autoObservableLabel(
      entry.representation_id,
      reductions,
      tensorScope,
      flattenMode,
      "representation",
      entry,
    );
    return {
      observableSource: "representation",
      tensorName: entry.representation_id,
      tensorShape: entry.shape,
      tensorScope,
      flattenMode,
      reductions,
      label,
      representationId: entry.representation_id,
      layerIndex: entry.layer_index,
      layerIo: entry.io,
    };
  }
  if (tensorNames.length === 0) return null;
  const tensorName = tensorNames[rng.int(tensorNames.length)]!;
  const tensorShape = specs[tensorName]?.shape ?? [];
  let flattenMode = randomFlattenMode(rng, prefs.svEntropy);
  if (flattenMode === "sv_entropy") flattenMode = "local";
  const reductions =
    flattenMode === "none"
      ? randomReductionDrafts(tensorShape, tensorName, rng, prefs)
      : flattenMode === "sv_entropy"
        ? [defaultSvEntropyReductionDraft()]
        : [randomFlatReductionDraft(rng, prefs)];
  const tensorScope = pickRandomTensorScope(
    tensorName,
    tensorNames,
    flattenMode,
    prefs.allMatchingLayers,
    rng,
  );
  const label = autoObservableLabel(tensorName, reductions, tensorScope, flattenMode, "weight");
  return {
    observableSource: "weight",
    tensorName,
    tensorShape,
    tensorScope,
    flattenMode,
    reductions,
    label,
  };
}

export function buildRandomObservableDrafts(
  count: number,
  seed: number,
  tensorNames: string[],
  specs: Record<string, { shape: number[] }>,
  repEntries: RepresentationEntry[] = [],
  options: RandomObservableGenerateOptions = {},
): RandomObservableDraft[] {
  const rng = createSeededRng(seed);
  const out: RandomObservableDraft[] = [];
  for (let i = 0; i < count; i++) {
    const draft = buildRandomObservableDraft(tensorNames, specs, rng, repEntries, options);
    if (draft) out.push(draft);
  }
  return out;
}

export function reductionsForFlattenMode(
  flattenMode: ObservableFlattenMode,
  shape: number[],
  tensorName: string,
  current: AxisReductionDraft[],
): AxisReductionDraft[] {
  if (flattenMode === "sv_entropy") {
    return [defaultSvEntropyReductionDraft()];
  }
  if (flattenMode === "none") {
    if (current.length === shape.length && shape.length > 0) return current;
    return defaultReductionDrafts(shape, tensorName);
  }
  if (current.length === 1 && (flattenMode === "local" || flattenMode === "global")) return current;
  return [defaultFlatReductionDraft(current[0]?.op ?? "l2_norm")];
}

export function flattenOptionsForSource(source: ObservableSource): typeof OBSERVABLE_FLATTEN_OPTIONS {
  if (source === "representation") {
    return OBSERVABLE_FLATTEN_OPTIONS.filter((o) => o.id !== "global");
  }
  return OBSERVABLE_FLATTEN_OPTIONS;
}
