import type { ModularAdditionDatasetNodeData } from "../../components/nodes/modularAdditionDatasetDefaults";
import { defaultModularAdditionDatasetData } from "../../components/nodes/modularAdditionDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["modulus", "trainFraction", "seed"]);

export const DEFAULT_MODULAR_ADDITION_DATASET_SPEC_NAME = "ModularAdditionDataset";
export const DEFAULT_MODULAR_ADDITION_DATASET_PARAM_ORDER: (keyof ModularAdditionDatasetNodeData)[] = [
  "modulus",
  "trainFraction",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(v: unknown): string {
  const s = firstScalar(v);
  return typeof s === "number" ? String(s) : JSON.stringify(s);
}

export function generateModularAdditionDatasetSpecCode(
  d: ModularAdditionDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_MODULAR_ADDITION_DATASET_SPEC_NAME;
  const merged = { ...defaultModularAdditionDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_MODULAR_ADDITION_DATASET_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  const lines: string[] = [`def ${name}(`];
  for (const key of keys) {
    lines.push(`    ${camelToSnakeCase(key)} = ${formatPyDefault(merged[key])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(`    p = int(modulus)`);
  lines.push(`    frac = float(train_fraction)`);
  lines.push(`    if p < 2:`);
  lines.push(`        raise ValueError("modulus must be >= 2")`);
  lines.push(`    if not (0.0 < frac < 1.0):`);
  lines.push(`        raise ValueError("train_fraction must be in (0, 1)")`);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    a, b = np.meshgrid(np.arange(p, dtype=np.int64), np.arange(p, dtype=np.int64), indexing="ij")`);
  lines.push(`    x_all = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)`);
  lines.push(`    y_all = ((x_all[:, 0] + x_all[:, 1]) % p).astype(np.int64)`);
  lines.push(`    perm = rng.permutation(x_all.shape[0])`);
  lines.push(`    x_all = x_all[perm]`);
  lines.push(`    y_all = y_all[perm]`);
  lines.push(`    n_train = int(round(frac * x_all.shape[0]))`);
  lines.push(`    n_train = min(max(n_train, 1), x_all.shape[0])`);
  lines.push(`    n_test = int(x_all.shape[0] - n_train)`);
  lines.push(`    x_train = x_all[:n_train]`);
  lines.push(`    y_train = y_all[:n_train]`);
  lines.push(`    if n_test > 0:`);
  lines.push(`        x_test = x_all[n_train:]`);
  lines.push(`        y_test = y_all[n_train:]`);
  lines.push(`    else:`);
  lines.push(`        x_test = None`);
  lines.push(`        y_test = None`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseModularAdditionDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<ModularAdditionDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<ModularAdditionDatasetNodeData> = {};
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
    if (camel === "trainFraction") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "Invalid float for trainFraction" };
      }
      patch.trainFraction = n;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof ModularAdditionDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
