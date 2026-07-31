import type { BigramLowRankDatasetNodeData } from "../../components/nodes/bigramLowRankDatasetDefaults";
import {
  BIGRAM_SPECTRUM_DECAY_IDS,
  defaultBigramLowRankDatasetData,
} from "../../components/nodes/bigramLowRankDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "vocabSize",
  "rank",
  "logitScale",
  "corruptRatio",
  "corruptScale",
  "decayType",
  "alpha",
  "trainSize",
  "testSize",
  "seed",
  "initSeed",
]);

const DECAY_PARSE = new Set<string>(BIGRAM_SPECTRUM_DECAY_IDS);

export const DEFAULT_BIGRAM_LOW_RANK_DATASET_SPEC_NAME = "BigramLowRankDataset";

export const DEFAULT_BIGRAM_LOW_RANK_DATASET_PARAM_ORDER: (keyof BigramLowRankDatasetNodeData)[] = [
  "vocabSize",
  "rank",
  "logitScale",
  "corruptRatio",
  "corruptScale",
  "decayType",
  "alpha",
  "trainSize",
  "testSize",
  "seed",
  "initSeed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(v: unknown): string {
  const s = firstScalar(v);
  return typeof s === "number" ? String(s) : JSON.stringify(s);
}

export function generateBigramLowRankDatasetSpecCode(
  d: BigramLowRankDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_BIGRAM_LOW_RANK_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultBigramLowRankDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_BIGRAM_LOW_RANK_DATASET_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const key of keys) {
    const sn = camelToSnakeCase(key);
    lines.push(`    ${sn} = ${formatPyDefault(merged[key as keyof BigramLowRankDatasetNodeData])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(``);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    R = int(rank)`);
  lines.push(`    scale = float(logit_scale)`);
  lines.push(`    corrupt_ratio_f = float(corrupt_ratio)`);
  lines.push(`    corrupt_scale_f = float(corrupt_scale)`);
  lines.push(`    a = float(alpha)`);
  lines.push(`    dt = str(decay_type).strip().lower().replace("-", "_")`);
  lines.push(`    n_train = int(train_size)`);
  lines.push(`    n_test = int(test_size)`);
  lines.push(`    sample_rng = np.random.default_rng(int(seed))`);
  lines.push(`    init_rng = np.random.default_rng(int(init_seed))`);
  lines.push(`    if V < 2:`);
  lines.push(`        raise ValueError("vocab_size must be >= 2")`);
  lines.push(`    if R < 1 or R > V:`);
  lines.push(`        raise ValueError("rank must be in [1, vocab_size]")`);
  lines.push(`    if n_train < 1 or n_test < 0:`);
  lines.push(`        raise ValueError("train_size must be >= 1 and test_size >= 0")`);
  lines.push(`    if corrupt_ratio_f < 0.0 or corrupt_ratio_f > 1.0:`);
  lines.push(`        raise ValueError("corrupt_ratio must be in [0, 1]")`);
  lines.push(`    if corrupt_scale_f < 0.0:`);
  lines.push(`        raise ValueError("corrupt_scale must be >= 0")`);
  lines.push(`    if dt not in ("power_law", "exponential"):`);
  lines.push(`        raise ValueError("decay_type must be 'power_law' or 'exponential'")`);
  lines.push(``);
  lines.push(`    A = init_rng.standard_normal((V, R)).astype(np.float64)`);
  lines.push(`    B = init_rng.standard_normal((R, V)).astype(np.float64)`);
  lines.push(`    if a == 0.0:`);
  lines.push(`        lamb = np.ones((R,), dtype=np.float64)`);
  lines.push(`    elif dt == "exponential":`);
  lines.push(`        n = np.arange(1, R + 1, dtype=np.float64)`);
  lines.push(`        lamb = np.exp(-a * n)`);
  lines.push(`    else:`);
  lines.push(`        n = np.arange(1, R + 1, dtype=np.float64)`);
  lines.push(`        lamb = np.power(n, -a)`);
  lines.push(`    logits = (A * lamb[None, :]) @ B`);
  lines.push(`    logits = scale * logits / np.sqrt(float(max(R, 1)))`);
  lines.push(``);
  lines.push(`    row_max = logits.max(axis=1, keepdims=True)`);
  lines.push(`    probs = np.exp(logits - row_max)`);
  lines.push(`    probs = probs / probs.sum(axis=1, keepdims=True)`);
  lines.push(``);
  lines.push(`    pi = np.full((V,), 1.0 / float(V), dtype=np.float64)`);
  lines.push(`    for _ in range(256):`);
  lines.push(`        pi = pi @ probs`);
  lines.push(`        s = pi.sum()`);
  lines.push(`        if s <= 0 or not np.isfinite(s):`);
  lines.push(`            raise ValueError("invalid stationary distribution from transition matrix")`);
  lines.push(`        pi = pi / s`);
  lines.push(``);
  lines.push(`    def _sample(n):`);
  lines.push(`        if n <= 0:`);
  lines.push(`            return None, None`);
  lines.push(`        x = sample_rng.choice(V, size=(n,), p=pi).astype(np.int64)`);
  lines.push(`        y = np.empty((n,), dtype=np.int64)`);
  lines.push(`        for i in range(n):`);
  lines.push(`            p = probs[x[i]]`);
  lines.push(`            if corrupt_ratio_f > 0.0 and sample_rng.random() < corrupt_ratio_f:`);
  lines.push(`                noisy_logits = sample_rng.standard_normal((V,)).astype(np.float64) * corrupt_scale_f`);
  lines.push(`                noisy_logits = noisy_logits - np.max(noisy_logits)`);
  lines.push(`                noisy = np.exp(noisy_logits)`);
  lines.push(`                noisy = noisy / np.sum(noisy)`);
  lines.push(`                p = noisy`);
  lines.push(`            y[i] = sample_rng.choice(V, p=p)`);
  lines.push(`        x = x[:, None]`);
  lines.push(`        return x, y`);
  lines.push(``);
  lines.push(`    x_train, y_train = _sample(n_train)`);
  lines.push(`    x_test, y_test = _sample(n_test)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseBigramLowRankDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<BigramLowRankDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<BigramLowRankDatasetNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of p.params) {
    const camel = snakeToCamelCase(row.snakeName === "rank_decay" ? "alpha" : row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") extras[camel] = val;
      continue;
    }
    paramOrder.push(camel);
    if (camel === "logitScale" || camel === "corruptRatio" || camel === "corruptScale" || camel === "alpha") {
      const f = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(f)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      }
      patch[camel as keyof BigramLowRankDatasetNodeData] = f as never;
      continue;
    }
    if (camel === "decayType") {
      const s = typeof val === "string" ? val.trim().toLowerCase().replace(/-/g, "_") : String(val);
      const norm = s === "exp" ? "exponential" : s === "powerlaw" ? "power_law" : s;
      if (!DECAY_PARSE.has(norm)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid decayType: ${val}` };
      }
      patch.decayType = norm as never;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof BigramLowRankDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
