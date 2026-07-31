/**
 * Python spec snippet for toy / structured LM dataset nodes: full inlined engine logic (readable
 * in the Code notebook) plus the same `data` dict and `*_from_seed` call shape as training.
 */
import type { ToyLanguageDatasetKind, ToyLanguageDatasetNodeData } from "../../components/nodes/toyLanguageDatasetDefaults";
import { defaultToyLanguageDatasetData } from "../../components/nodes/toyLanguageDatasetDefaults";
import { intChoices, packIntList } from "../../components/nodes/multiValueUtils";
import { DYCK_LM_NOTEBOOK_IMPL_BLOCK } from "./toyLanguageDyckNotebookEmbed";
import {
  EXTERNAL_TOY_LM_NOTEBOOK_IMPL_BLOCK,
  FORMAL_SUITE_LM_NOTEBOOK_IMPL_BLOCK,
  NGRAM_LM_NOTEBOOK_IMPL_BLOCK,
  PCFG_LM_NOTEBOOK_IMPL_BLOCK,
  PHYSICS_LM_NOTEBOOK_IMPL_BLOCK,
} from "./toyLanguageRuntimeNotebookEmbeds.generated";

function firstScalar(v: unknown): unknown {
  if (Array.isArray(v) && v.length > 0) return v[0];
  return v;
}

/** Emit a Python literal suitable for a dict value. */
function pyLiteral(v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "string") return JSON.stringify(s);
  if (typeof s === "boolean") return s ? "True" : "False";
  if (typeof s === "number") {
    if (!Number.isFinite(s)) return "float('nan')";
    if (Number.isInteger(s)) return String(s);
    const t = s.toPrecision(12).replace(/\.?0+$/, "");
    return t.includes("e") || t.includes("E") ? String(s) : t;
  }
  if (s === null || s === undefined) return "None";
  return JSON.stringify(s);
}

/** Field order matches the trainer / JSON node payload (camelCase keys). */
export function toyLanguageDatasetSpecKeyOrder(kind: ToyLanguageDatasetKind): readonly string[] {
  switch (kind) {
    case "pcfg_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "pcfgGenMode",
        "pcfgGrammarId",
        "pcfgMaxDepth",
        "pcfgTermProb",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "dyck_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "contextLength",
        "numBracketTypes",
        "maxNestingDepth",
        "vocabSize",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "ngram_language_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "orderN",
        "dirichletAlpha",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "formal_language_suite_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "languageType",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "scan_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "dataSource",
        "cacheDir",
        "scanUrl",
        "vocabSize",
        "contextLength",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "cogs_dataset":
    case "listops_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "dataSource",
        "cacheDir",
        "vocabSize",
        "contextLength",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "tinystories_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "dataSource",
        "cacheDir",
        "tinyStoriesUrl",
        "vocabSize",
        "vocabCap",
        "contextLength",
        "tokenizerMode",
        "seqLen",
        "stride",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "phi1_style_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "dataSource",
        "cacheDir",
        "downloadUrl",
        "vocabSize",
        "vocabCap",
        "contextLength",
        "tokenizerMode",
        "seqLen",
        "stride",
        "domainMix",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "biography_lm_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "biographyAugmentation",
        "slotNoiseProb",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "relation_tuple_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "relationMode",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "synthetic_playground_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "playgroundFamily",
        "depoWindow",
        "manoModulus",
        "lanoNestingDepth",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "multi_hop_fact_chain_dataset":
      return [
        "samplingMode",
        "inspectFormat",
        "vocabSize",
        "contextLength",
        "chainHops",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
      ];
    case "tinyshakespeare_lm_dataset":
      return [
        "vocabSize",
        "contextLength",
        "trainSize",
        "testSize",
        "seed",
        "initSeed",
        "stride",
      ];
    default: {
      const _k: never = kind;
      return _k;
    }
  }
}

function safePythonFuncName(raw: string, fallback: string): string {
  const t = raw.trim() || fallback;
  const cleaned = t.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!cleaned) return fallback;
  if (/^[0-9]/.test(cleaned)) return `_${cleaned}`;
  return cleaned;
}

/** Inlined Python (indented for ``def <spec>()``) + entrypoint name used after ``data = {...}``. */
function notebookImplAndEntry(kind: ToyLanguageDatasetKind): { impl: string; fn: string } {
  const ext = EXTERNAL_TOY_LM_NOTEBOOK_IMPL_BLOCK.trimEnd();
  const phys = PHYSICS_LM_NOTEBOOK_IMPL_BLOCK.trimEnd();
  switch (kind) {
    case "dyck_dataset":
      return { impl: DYCK_LM_NOTEBOOK_IMPL_BLOCK.trimEnd(), fn: "build_dyck_lm_arrays_from_seed" };
    case "pcfg_dataset":
      return { impl: PCFG_LM_NOTEBOOK_IMPL_BLOCK.trimEnd(), fn: "build_pcfg_lm_arrays_from_seed" };
    case "ngram_language_dataset":
      return { impl: NGRAM_LM_NOTEBOOK_IMPL_BLOCK.trimEnd(), fn: "build_ngram_lm_arrays_from_seed" };
    case "formal_language_suite_dataset":
      return { impl: FORMAL_SUITE_LM_NOTEBOOK_IMPL_BLOCK.trimEnd(), fn: "build_formal_suite_lm_arrays_from_seed" };
    case "scan_dataset":
      return { impl: ext, fn: "build_scan_arrays_from_seed" };
    case "cogs_dataset":
      return { impl: ext, fn: "build_cogs_arrays_from_seed" };
    case "listops_dataset":
      return { impl: ext, fn: "build_listops_arrays_from_seed" };
    case "tinystories_dataset":
      return { impl: ext, fn: "tiny_stories_lm_from_seed" };
    case "phi1_style_dataset":
      return { impl: ext, fn: "phi_style_lm_from_seed" };
    case "biography_lm_dataset":
      return { impl: phys, fn: "build_biography_lm_arrays_from_seed" };
    case "relation_tuple_dataset":
      return { impl: phys, fn: "build_relation_tuple_lm_arrays_from_seed" };
    case "synthetic_playground_dataset":
      return { impl: phys, fn: "build_synthetic_playground_lm_arrays_from_seed" };
    case "multi_hop_fact_chain_dataset":
      return { impl: phys, fn: "build_multi_hop_fact_chain_lm_arrays_from_seed" };
    case "tinyshakespeare_lm_dataset":
      return { impl: ext, fn: "tiny_shakespeare_lm_from_seed" };
    default: {
      const _k: never = kind;
      return _k;
    }
  }
}

function pushDataDictLines(
  lines: string[],
  merged: Record<string, unknown>,
  keys: readonly string[],
): void {
  lines.push(`    data = {`);
  for (const key of keys) {
    if (!(key in merged)) continue;
    const v = merged[key];
    if (v === undefined) continue;
    lines.push(`        ${JSON.stringify(key)}: ${pyLiteral(v)},`);
  }
  if (merged.extras && typeof merged.extras === "object" && !Array.isArray(merged.extras)) {
    const ex = merged.extras as Record<string, unknown>;
    for (const [ek, ev] of Object.entries(ex)) {
      lines.push(`        ${JSON.stringify(ek)}: ${pyLiteral(ev)},`);
    }
  }
  lines.push(`    }`);
}

function pushSampleAndReturn(lines: string[], fn: string): void {
  lines.push(`    n_train = int(data["trainSize"])`);
  lines.push(`    n_test = int(data["testSize"])`);
  lines.push(`    x_train, y_train, x_test, y_test = ${fn}(data, n_train, n_test)`);
  lines.push(
    `    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "data": data}`,
  );
}

export function generateToyLanguageDatasetSpecCode(
  kind: ToyLanguageDatasetKind,
  d: ToyLanguageDatasetNodeData,
  specName: string,
): string {
  const defaultName = `${kind}Spec`;
  const name = safePythonFuncName(specName, defaultName);
  const merged: Record<string, unknown> = {
    ...(defaultToyLanguageDatasetData(kind) as Record<string, unknown>),
    ...(d as Record<string, unknown>),
  };
  if (kind === "dyck_dataset") {
    const ks = intChoices(merged.numBracketTypes, 1);
    const vDerived = ks.map((k) => 2 * Math.max(1, k));
    merged.vocabSize = packIntList(vDerived);
  }
  const keys = toyLanguageDatasetSpecKeyOrder(kind);
  const { impl, fn } = notebookImplAndEntry(kind);

  const lines: string[] = [];
  lines.push(`def ${name}():`);
  lines.push(`    """Toy language dataset (${kind}).`);
  lines.push(
    `    Generator implementation is inlined below (mirrors \`comfy_research/engine/toy_language_*.py\`; regenerate via \`python scripts/gen_toy_language_notebook_embeds.py\`, and keep \`toyLanguageDyckNotebookEmbed.ts\` aligned for Dyck).`,
  );
  lines.push(`    Run with repo root on \`PYTHONPATH\` so \`comfy_research\` package imports work.`);
  lines.push(`    """`);
  lines.push(impl);
  pushDataDictLines(lines, merged, keys);
  pushSampleAndReturn(lines, fn);
  return lines.join("\n");
}
