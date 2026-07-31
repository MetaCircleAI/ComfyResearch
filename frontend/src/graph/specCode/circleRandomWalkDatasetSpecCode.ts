import type { CircleRandomWalkDatasetNodeData } from "../../components/nodes/circleRandomWalkDatasetDefaults";
import { defaultCircleRandomWalkDatasetData } from "../../components/nodes/circleRandomWalkDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["vocabSize", "contextLength", "rightStepProb", "trainSize", "testSize", "seed"]);

export const DEFAULT_CIRCLE_RANDOM_WALK_DATASET_SPEC_NAME = "CircleRandomWalkDataset";

export const DEFAULT_CIRCLE_RANDOM_WALK_DATASET_PARAM_ORDER: (keyof CircleRandomWalkDatasetNodeData)[] = [
  "vocabSize",
  "contextLength",
  "rightStepProb",
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

export function generateCircleRandomWalkDatasetSpecCode(
  d: CircleRandomWalkDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_CIRCLE_RANDOM_WALK_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultCircleRandomWalkDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_CIRCLE_RANDOM_WALK_DATASET_PARAM_ORDER).filter((k) =>
    KNOWN_KEYS.has(k),
  );
  for (const key of keys) {
    const sn = camelToSnakeCase(key);
    lines.push(`    ${sn} = ${formatPyDefault(merged[key as keyof CircleRandomWalkDatasetNodeData])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    L = int(context_length)`);
  lines.push(`    p_right = float(right_step_prob)`);
  lines.push(`    if V < 2 or L < 1:`);
  lines.push(`        raise ValueError("vocab_size >= 2 and context_length >= 1 are required")`);
  lines.push(`    if p_right < 0.0 or p_right > 1.0:`);
  lines.push(`        raise ValueError("right_step_prob must be in [0, 1]")`);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    def _sample(n):`);
  lines.push(`        n = int(n)`);
  lines.push(`        if n <= 0:`);
  lines.push(`            return None, None`);
  lines.push(`        seq = np.empty((n, L + 1), dtype=np.int64)`);
  lines.push(`        seq[:, 0] = rng.integers(0, V, size=n, dtype=np.int64)`);
  lines.push(`        steps = np.where(rng.random((n, L)) < p_right, 1, -1).astype(np.int64)`);
  lines.push(`        for t in range(L):`);
  lines.push(`            seq[:, t + 1] = (seq[:, t] + steps[:, t]) % V`);
  lines.push(`        x = seq[:, :L]`);
  lines.push(`        y = seq[:, L]`);
  lines.push(`        return x, y`);
  lines.push(`    x_train, y_train = _sample(train_size)`);
  lines.push(`    x_test, y_test = _sample(test_size)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseCircleRandomWalkDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<CircleRandomWalkDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<CircleRandomWalkDatasetNodeData> = {};
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
    if (camel === "rightStepProb") {
      const f = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(f)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "Invalid rightStepProb" };
      }
      patch.rightStepProb = f;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof CircleRandomWalkDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
