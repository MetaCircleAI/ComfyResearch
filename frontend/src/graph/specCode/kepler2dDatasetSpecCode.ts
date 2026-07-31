import type { Kepler2dDatasetNodeData } from "../../components/nodes/kepler2dDatasetDefaults";
import { defaultKepler2dDatasetData } from "../../components/nodes/kepler2dDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "trainSize",
  "testSize",
  "semiMajorAxisMin",
  "semiMajorAxisMax",
  "eccentricityMin",
  "eccentricityMax",
  "meanMotion",
  "outputDistribution",
  "noiseLevel",
  "seed",
]);

export const DEFAULT_KEPLER_2D_DATASET_SPEC_NAME = "Kepler2dDataset";
export const DEFAULT_KEPLER_2D_DATASET_PARAM_ORDER: (keyof Kepler2dDatasetNodeData)[] = [
  "contextLength",
  "trainSize",
  "testSize",
  "semiMajorAxisMin",
  "semiMajorAxisMax",
  "eccentricityMin",
  "eccentricityMax",
  "meanMotion",
  "outputDistribution",
  "noiseLevel",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function pyTypeForKey(key: keyof Kepler2dDatasetNodeData): string {
  if (key === "outputDistribution") return "str";
  if (
    key === "semiMajorAxisMin" ||
    key === "semiMajorAxisMax" ||
    key === "eccentricityMin" ||
    key === "eccentricityMax" ||
    key === "meanMotion" ||
    key === "noiseLevel"
  ) {
    return "float";
  }
  return "int";
}

function formatPyDefault(key: keyof Kepler2dDatasetNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "outputDistribution") return JSON.stringify(String(s));
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateKepler2dDatasetSpecCode(
  d: Kepler2dDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_KEPLER_2D_DATASET_SPEC_NAME;
  const merged = { ...defaultKepler2dDatasetData(), ...d };
  const keys = (order.length ? order : DEFAULT_KEPLER_2D_DATASET_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  const lines: string[] = [`def ${name}(`];
  for (const k of keys) {
    const ck = k as keyof Kepler2dDatasetNodeData;
    lines.push(`    ${camelToSnakeCase(String(ck))}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`):`);
  lines.push(`    """`);
  lines.push(`    Kepler-like 2D orbits with Newton solve of M = E - e sin(E).`);
  lines.push(`    Input x is [N,T,2] positions; target y is x shifted by one step (same shape).`);
  lines.push(`    """`);
  lines.push(`    import numpy as np`);
  lines.push(``);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    t = int(context_length)`);
  lines.push(`    if t < 1:`);
  lines.push(`        raise ValueError("context_length must be >= 1")`);
  lines.push(`    a_min = float(semi_major_axis_min)`);
  lines.push(`    a_max = float(semi_major_axis_max)`);
  lines.push(`    e_min = float(eccentricity_min)`);
  lines.push(`    e_max = float(eccentricity_max)`);
  lines.push(`    n = float(mean_motion)`);
  lines.push(`    if not (a_min > 0 and a_max >= a_min):`);
  lines.push(`        raise ValueError("Require 0 < semi_major_axis_min <= semi_major_axis_max.")`);
  lines.push(`    if e_min < 0 or e_max < e_min or e_max >= 0.999:`);
  lines.push(`        raise ValueError("Require 0 <= eccentricity_min <= eccentricity_max < 0.999.")`);
  lines.push(`    if n <= 0:`);
  lines.push(`        raise ValueError("mean_motion must be > 0.")`);
  lines.push(``);
  lines.push(`    def solve_kepler(mean_anomaly: np.ndarray, ecc: np.ndarray, n_iter: int = 7) -> np.ndarray:`);
  lines.push(`        e_anom = mean_anomaly.copy()`);
  lines.push(`        for _ in range(n_iter):`);
  lines.push(`            f = e_anom - ecc * np.sin(e_anom) - mean_anomaly`);
  lines.push(`            fp = 1.0 - ecc * np.cos(e_anom)`);
  lines.push(`            e_anom = e_anom - f / np.clip(fp, 1e-6, None)`);
  lines.push(`        return e_anom`);
  lines.push(``);
  lines.push(`    def sample_rows(num_rows: int):`);
  lines.push(`        num_rows = int(max(0, num_rows))`);
  lines.push(`        if num_rows <= 0:`);
  lines.push(`            z = np.zeros((0, t, 2), dtype=np.float32)`);
  lines.push(`            return z, z`);
  lines.push(`        a = rng.uniform(a_min, a_max, size=(num_rows,)).astype(np.float64)`);
  lines.push(`        e = rng.uniform(e_min, e_max, size=(num_rows,)).astype(np.float64)`);
  lines.push(`        phi = rng.uniform(0.0, 2.0 * np.pi, size=(num_rows,)).astype(np.float64)`);
  lines.push(`        m0 = rng.uniform(0.0, 2.0 * np.pi, size=(num_rows,)).astype(np.float64)`);
  lines.push(``);
  lines.push(`        m_steps = m0[:, None] + n * np.arange(t + 1, dtype=np.float64)[None, :]`);
  lines.push(`        e_mat = np.repeat(e[:, None], t + 1, axis=1)`);
  lines.push(`        e_anom = solve_kepler(m_steps, e_mat)`);
  lines.push(`        cos_e = np.cos(e_anom)`);
  lines.push(`        sin_e = np.sin(e_anom)`);
  lines.push(`        fac = np.sqrt(np.clip(1.0 - e_mat * e_mat, 1e-8, None))`);
  lines.push(``);
  lines.push(`        x_orb = a[:, None] * (cos_e - e_mat)`);
  lines.push(`        y_orb = a[:, None] * fac * sin_e`);
  lines.push(``);
  lines.push(`        cphi = np.cos(phi)[:, None]`);
  lines.push(`        sphi = np.sin(phi)[:, None]`);
  lines.push(`        x_rot = cphi * x_orb - sphi * y_orb`);
  lines.push(`        y_rot = sphi * x_orb + cphi * y_orb`);
  lines.push(``);
  lines.push(`        pts = np.stack([x_rot, y_rot], axis=-1).astype(np.float32)`);
  lines.push(`        x = pts[:, :t, :]`);
  lines.push(`        y = pts[:, 1 : t + 1, :]`);
  lines.push(`        if output_distribution == "additive_gaussian" and float(noise_level) > 0:`);
  lines.push(`            y = y.copy()`);
  lines.push(`            y[:, -1, :] = y[:, -1, :] + float(noise_level) * rng.standard_normal((num_rows, 2)).astype(np.float32)`);
  lines.push(`        return x, y`);
  lines.push(``);
  lines.push(`    x_train, y_train = sample_rows(train_size)`);
  lines.push(`    x_test, y_test = sample_rows(test_size) if int(test_size) > 0 else (None, None)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseKepler2dDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<Kepler2dDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<Kepler2dDatasetNodeData> = {};
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
    const key = camel as keyof Kepler2dDatasetNodeData;
    if (key === "outputDistribution") {
      patch[key] = String(val) as never;
      continue;
    }
    const asNumber = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(asNumber)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid number for ${camel}` };
    }
    const intKeys = new Set<keyof Kepler2dDatasetNodeData>(["contextLength", "trainSize", "testSize", "seed"]);
    if (intKeys.has(key) && !Number.isInteger(asNumber)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[key] = asNumber as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
