import type { DatasetMixerNodeData } from "../../components/nodes/datasetMixerDefaults";
import { defaultDatasetMixerData } from "../../components/nodes/datasetMixerDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["trainTotalSamples", "testTotalSamples", "proportionA", "initSeed"]);

export const DEFAULT_DATASET_MIXER_SPEC_NAME = "DatasetMixerA";
export const DEFAULT_DATASET_MIXER_PARAM_ORDER: (keyof DatasetMixerNodeData)[] = [
  "trainTotalSamples",
  "testTotalSamples",
  "proportionA",
  "initSeed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(v: unknown): string {
  const s = firstScalar(v);
  return typeof s === "number" ? String(s) : JSON.stringify(s);
}

export function generateDatasetMixerSpecCode(
  d: DatasetMixerNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_DATASET_MIXER_SPEC_NAME;
  const merged = { ...defaultDatasetMixerData(), ...d };
  const keys = (order.length ? order : DEFAULT_DATASET_MIXER_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  const lines: string[] = [`def ${name}(`];
  for (const key of keys) {
    lines.push(`    ${camelToSnakeCase(key)} = ${formatPyDefault(merged[key])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(`    n_train = int(train_total_samples)`);
  lines.push(`    n_test = int(test_total_samples)`);
  lines.push(`    p_a = float(proportion_a)`);
  lines.push(`    p_b = 1.0 - p_a`);
  lines.push(`    if n_train < 1:`);
  lines.push(`        raise ValueError("train_total_samples must be >= 1")`);
  lines.push(`    if n_test < 0:`);
  lines.push(`        raise ValueError("test_total_samples must be >= 0")`);
  lines.push(`    if p_a < 0.0 or p_a > 1.0:`);
  lines.push(`        raise ValueError("proportion_a must be in [0, 1]")`);
  lines.push(`    def split(n_total: int) -> tuple[int, int]:`);
  lines.push(`        n_a = int(round(n_total * p_a))`);
  lines.push(`        n_a = max(0, min(n_total, n_a))`);
  lines.push(`        return n_a, n_total - n_a`);
  lines.push(`    n_a_tr, n_b_tr = split(n_train)`);
  lines.push(`    n_a_te, n_b_te = split(n_test)`);
  lines.push(`    return {"n_a_train": n_a_tr, "n_b_train": n_b_tr, "n_a_test": n_a_te, "n_b_test": n_b_te, "p_b": p_b}`);
  return lines.join("\n");
}

export function parseDatasetMixerSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<DatasetMixerNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<DatasetMixerNodeData> = {};
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
    if (camel === "proportionA") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "Invalid float for proportionA" };
      }
      patch.proportionA = n;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof DatasetMixerNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
