import type { OutputDistributionId } from "../../components/nodes/linearDatasetDefaults";
import type { UnigramDatasetNodeData } from "../../components/nodes/unigramDatasetDefaults";
import { defaultUnigramDatasetData } from "../../components/nodes/unigramDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "vocabSize",
  "outputDistribution",
  "alpha",
  "contextLength",
  "trainSize",
  "testSize",
  "seed",
]);

export const DEFAULT_UNIGRAM_DATASET_SPEC_NAME = "UnigramDataset";

export const DEFAULT_UNIGRAM_DATASET_PARAM_ORDER: (keyof UnigramDatasetNodeData)[] = [
  "vocabSize",
  "outputDistribution",
  "alpha",
  "contextLength",
  "trainSize",
  "testSize",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(v: unknown): string {
  const s = firstScalar(v);
  return typeof s === "number" ? String(s) : JSON.stringify(s);
}

export function generateUnigramDatasetSpecCode(
  d: UnigramDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_UNIGRAM_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultUnigramDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_UNIGRAM_DATASET_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const key of keys) {
    const sn = camelToSnakeCase(key);
    lines.push(`    ${sn} = ${formatPyDefault(merged[key as keyof UnigramDatasetNodeData])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    a = float(alpha)`);
  lines.push(`    L = int(context_length)`);
  lines.push(`    if V < 2 or L < 1:`);
  lines.push(`        raise ValueError("vocab_size >= 2 and context_length >= 1 are required")`);
  lines.push(`    ranks = np.arange(1, V + 1, dtype=np.float64)`);
  lines.push(`    out_dist = str(output_distribution)`);
  lines.push(`    if out_dist == "power_law_class_probs":`);
  lines.push(`        av = max(float(a), 1e-8)`);
  lines.push(`        un = np.power(ranks, -av)`);
  lines.push(`    elif out_dist == "exponential_class_probs":`);
  lines.push(`        av = max(float(a), 1e-8)`);
  lines.push(`        un = np.exp(-av * ranks)`);
  lines.push(`    elif out_dist == "uniform_class_probs":`);
  lines.push(`        un = np.ones((V,), dtype=np.float64)`);
  lines.push(`    else:`);
  lines.push(`        raise ValueError("output_distribution must be uniform_class_probs, power_law_class_probs, or exponential_class_probs")`);
  lines.push(`    z = float(np.sum(un))`);
  lines.push(`    if (not np.isfinite(z)) or z <= 0:`);
  lines.push(`        raise ValueError("invalid unigram token distribution")`);
  lines.push(`    probs = (un / z).astype(np.float64)`);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    def _sample(n):`);
  lines.push(`        if int(n) <= 0:`);
  lines.push(`            return None, None`);
  lines.push(`        x = rng.choice(V, size=(int(n), L), p=probs).astype(np.int64)`);
  lines.push(`        y = rng.choice(V, size=(int(n),), p=probs).astype(np.int64)`);
  lines.push(`        return x, y`);
  lines.push(`    x_train, y_train = _sample(train_size)`);
  lines.push(`    x_test, y_test = _sample(test_size)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseUnigramDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<UnigramDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<UnigramDatasetNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of p.params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") extras[camel] = val;
      continue;
    }
    paramOrder.push(camel);
    if (camel === "alpha") {
      const f = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(f)) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "Invalid alpha" };
      patch.alpha = f;
      continue;
    }
    if (camel === "outputDistribution") {
      patch.outputDistribution = String(val) as OutputDistributionId;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof UnigramDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
