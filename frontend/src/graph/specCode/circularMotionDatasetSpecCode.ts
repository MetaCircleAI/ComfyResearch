import type { CircularMotionDatasetNodeData } from "../../components/nodes/circularMotionDatasetDefaults";
import { defaultCircularMotionDatasetData } from "../../components/nodes/circularMotionDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "vocabSize",
  "contextLength",
  "radiusMin",
  "radiusMax",
  "angularVelocity",
  "trainSize",
  "testSize",
  "seed",
]);

export const DEFAULT_CIRCULAR_MOTION_DATASET_SPEC_NAME = "CircularMotionDataset";

export const DEFAULT_CIRCULAR_MOTION_DATASET_PARAM_ORDER: (keyof CircularMotionDatasetNodeData)[] = [
  "vocabSize",
  "contextLength",
  "radiusMin",
  "radiusMax",
  "angularVelocity",
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

export function generateCircularMotionDatasetSpecCode(
  d: CircularMotionDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_CIRCULAR_MOTION_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultCircularMotionDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_CIRCULAR_MOTION_DATASET_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const key of keys) {
    const sn = camelToSnakeCase(key);
    lines.push(`    ${sn} = ${formatPyDefault(merged[key as keyof CircularMotionDatasetNodeData])},`);
  }
  lines.push(`):`);
  lines.push(`    import numpy as np`);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    L = int(context_length)`);
  lines.push(`    r_min = float(radius_min)`);
  lines.push(`    r_max = float(radius_max)`);
  lines.push(`    omega = float(angular_velocity)`);
  lines.push(`    if V < 2 or L < 1:`);
  lines.push(`        raise ValueError("vocab_size >= 2 and context_length >= 1 are required")`);
  lines.push(`    if r_min < 0.0 or r_max <= 0.0 or r_min > r_max or r_max >= 1.0:`);
  lines.push(`        raise ValueError("require 0 <= radius_min <= radius_max < 1")`);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    def _coord_to_token(c):`);
  lines.push(`        u = (np.clip(c, -1.0, 1.0 - 1e-8) + 1.0) * 0.5`);
  lines.push(`        return np.floor(V * u).astype(np.int64)`);
  lines.push(`    def _sample(n):`);
  lines.push(`        n = int(n)`);
  lines.push(`        if n <= 0:`);
  lines.push(`            return None, None`);
  lines.push(`        radii = rng.uniform(r_min, r_max, size=(n,)).astype(np.float64)`);
  lines.push(`        phi0 = rng.uniform(0.0, 2.0 * np.pi, size=(n,)).astype(np.float64)`);
  lines.push(`        x_ctx = np.empty((n, L), dtype=np.float64)`);
  lines.push(`        y_ctx = np.empty((n, L), dtype=np.float64)`);
  lines.push(`        for t in range(L):`);
  lines.push(`            phi = phi0 + omega * t`);
  lines.push(`            x_ctx[:, t] = radii * np.cos(phi)`);
  lines.push(`            y_ctx[:, t] = radii * np.sin(phi)`);
  lines.push(`        phi_n = phi0 + omega * L`);
  lines.push(`        x_n = radii * np.cos(phi_n)`);
  lines.push(`        y_n = radii * np.sin(phi_n)`);
  lines.push(`        pairs = np.stack([_coord_to_token(x_ctx), _coord_to_token(y_ctx)], axis=-1)`);
  lines.push(`        y_pair = np.stack([_coord_to_token(x_n), _coord_to_token(y_n)], axis=-1)`);
  lines.push(`        return pairs, y_pair`);
  lines.push(`    x_train, y_train = _sample(train_size)`);
  lines.push(`    x_test, y_test = _sample(test_size)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseCircularMotionDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<CircularMotionDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<CircularMotionDatasetNodeData> = {};
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
    if (camel === "radiusMin" || camel === "radiusMax" || camel === "angularVelocity") {
      const f = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(f)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      }
      patch[camel as keyof CircularMotionDatasetNodeData] = f as never;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof CircularMotionDatasetNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
