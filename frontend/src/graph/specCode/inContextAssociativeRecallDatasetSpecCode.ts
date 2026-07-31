import type { InContextAssociativeRecallDatasetNodeData } from "../../components/nodes/inContextAssociativeRecallDatasetDefaults";
import { defaultInContextAssociativeRecallDatasetData } from "../../components/nodes/inContextAssociativeRecallDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "vocabSize",
  "numPairs",
  "inContextRepeat",
  "crossSampleRepeatProb",
  "repeatedTokenCount",
  "trainSize",
  "testSize",
  "seed",
]);

export const DEFAULT_ICAR_SPEC_NAME = "InContextAssociativeRecallDataset";

export const DEFAULT_ICAR_PARAM_ORDER: (keyof InContextAssociativeRecallDatasetNodeData)[] = [
  "vocabSize",
  "numPairs",
  "inContextRepeat",
  "crossSampleRepeatProb",
  "repeatedTokenCount",
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

export function generateInContextAssociativeRecallDatasetSpecCode(
  d: InContextAssociativeRecallDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_ICAR_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultInContextAssociativeRecallDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_ICAR_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const key of keys) {
    const sn = camelToSnakeCase(key);
    lines.push(`    ${sn} = ${formatPyDefault(merged[key as keyof InContextAssociativeRecallDatasetNodeData])},`);
  }
  lines.push(`):`);
  lines.push(`    # Matches comfy_research/engine/trainer_run.py: associative recall sequence generation.`);
  lines.push(`    import numpy as np`);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    N = int(num_pairs)`);
  lines.push(`    B = int(in_context_repeat)`);
  lines.push(`    p = float(cross_sample_repeat_prob)`);
  lines.push(`    K = int(repeated_token_count)`);
  lines.push(`    if V < 2 or N < 1:`);
  lines.push(`        raise ValueError("vocab_size >= 2 and num_pairs >= 1 are required")`);
  lines.push(`    L = 2 * N + 1`);
  lines.push(`    rep_pool = np.arange(min(max(K, 1), V), dtype=np.int64)`);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(``);
  lines.push(`    def _sample_one():`);
  lines.push(`        keys = rng.permutation(V)[:N].astype(np.int64)`);
  lines.push(`        vals = rng.permutation(V)[:N].astype(np.int64)`);
  lines.push(`        query = keys[int(rng.integers(0, N))]`);
  lines.push(`        if p > 0 and rep_pool.size > 0 and float(rng.random()) < p:`);
  lines.push(`            query = rep_pool[int(rng.integers(0, rep_pool.size))]`);
  lines.push(`            if query not in keys:`);
  lines.push(`                keys[int(rng.integers(0, N))] = query`);
  lines.push(`        key_positions = np.arange(N, dtype=np.int64)`);
  lines.push(`        if B > 1:`);
  lines.push(`            others = key_positions[key_positions != np.argmax(keys == query)]`);
  lines.push(`            rng.shuffle(others)`);
  lines.push(`            for pos in others[: max(0, B - 1)]:`);
  lines.push(`                keys[pos] = query`);
  lines.push(`        seq = np.empty((L,), dtype=np.int64)`);
  lines.push(`        seq[0::2][:N] = keys`);
  lines.push(`        seq[1::2][:N] = vals`);
  lines.push(`        seq[-1] = query`);
  lines.push(`        hit = np.where(keys == query)[0]`);
  lines.push(`        y = vals[int(hit[0])] if hit.size else vals[0]`);
  lines.push(`        return seq, np.int64(y)`);
  lines.push(``);
  lines.push(`    def _sample_many(n):`);
  lines.push(`        n = int(n)`);
  lines.push(`        if n <= 0:`);
  lines.push(`            return None, None`);
  lines.push(`        x = np.empty((n, L), dtype=np.int64)`);
  lines.push(`        y = np.empty((n,), dtype=np.int64)`);
  lines.push(`        for i in range(n):`);
  lines.push(`            xi, yi = _sample_one()`);
  lines.push(`            x[i] = xi`);
  lines.push(`            y[i] = yi`);
  lines.push(`        return x, y`);
  lines.push(``);
  lines.push(`    x_train, y_train = _sample_many(train_size)`);
  lines.push(`    x_test, y_test = _sample_many(test_size)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseInContextAssociativeRecallDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<InContextAssociativeRecallDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<InContextAssociativeRecallDatasetNodeData> = {};
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
    if (camel === "crossSampleRepeatProb") {
      const f = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(f)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "Invalid crossSampleRepeatProb" };
      }
      patch.crossSampleRepeatProb = f;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof InContextAssociativeRecallDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
