import type { UniformLinearMotionDatasetNodeData } from "../../components/nodes/uniformLinearMotionDatasetDefaults";
import { defaultUniformLinearMotionDatasetData } from "../../components/nodes/uniformLinearMotionDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "positionDim",
  "trainSize",
  "testSize",
  "positionDistribution",
  "velocityDistribution",
  "velocityScale",
  /** Legacy; parsed into ``velocityDistribution``. */
  "x1Distribution",
  "outputDistribution",
  "noiseLevel",
  "seed",
]);

export const DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_SPEC_NAME = "UniformLinearMotionDataset";

export const DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_PARAM_ORDER: (keyof UniformLinearMotionDatasetNodeData)[] = [
  "contextLength",
  "positionDim",
  "trainSize",
  "testSize",
  "positionDistribution",
  "velocityDistribution",
  "velocityScale",
  "outputDistribution",
  "noiseLevel",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function pyTypeForKey(key: keyof UniformLinearMotionDatasetNodeData): string {
  if (
    key === "positionDistribution" ||
    key === "velocityDistribution" ||
    key === "outputDistribution"
  )
    return "str";
  if (key === "noiseLevel" || key === "velocityScale") return "float";
  return "int";
}

function formatPyDefault(key: keyof UniformLinearMotionDatasetNodeData, v: unknown): string {
  const s = firstScalar(v);
  if ((key === "noiseLevel" || key === "velocityScale") && typeof s === "number") return String(s);
  if (
    key === "positionDistribution" ||
    key === "velocityDistribution" ||
    key === "outputDistribution"
  ) {
    return JSON.stringify(String(s));
  }
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateUniformLinearMotionDatasetSpecCode(
  d: UniformLinearMotionDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_SPEC_NAME;
  const merged = { ...defaultUniformLinearMotionDatasetData(), ...d };
  const rawKeys = (order.length ? order : DEFAULT_UNIFORM_LINEAR_MOTION_DATASET_PARAM_ORDER).filter((k) =>
    KNOWN_KEYS.has(k),
  );
  const seen = new Set<string>();
  const keys: string[] = [];
  for (const k of rawKeys) {
    const nk = k === "x1Distribution" ? "velocityDistribution" : k;
    if (seen.has(nk)) continue;
    seen.add(nk);
    keys.push(nk);
  }
  const lines: string[] = [`def ${name}(`];
  for (const k of keys) {
    const ck = k as keyof UniformLinearMotionDatasetNodeData;
    lines.push(`    ${camelToSnakeCase(String(ck))}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`):`);
  lines.push(`    """`);
  lines.push(`    Uniform linear motion: sample x0 and velocity v; x_i = x0 + v*i.`);
  lines.push(`    Input x is [N,T,D] with rows x_0..x_{T-1}; target y is x_1..x_T.`);
  lines.push(`    Optional Gaussian noise is applied only to the last target row y_{T-1} -> x_T.`);
  lines.push(`    """`);
  lines.push(`    import numpy as np`);
  lines.push(``);
  lines.push(`    rng = np.random.default_rng(seed)`);
  lines.push(`    d = int(position_dim)`);
  lines.push(`    t = int(context_length)`);
  lines.push(``);
  lines.push(`    def _sample_rows(n: int, dist_x0: str, dist_v: str, v_scale: float):`);
  lines.push(`        n = int(max(0, n))`);
  lines.push(`        if n <= 0:`);
  lines.push(`            z = np.zeros((0, t, d), dtype=np.float32)`);
  lines.push(`            return z, z`);
  lines.push(`        def _draw(dist: str, shape):`);
  lines.push(`            if dist == "uniform_neg1_1":`);
  lines.push(`                return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)`);
  lines.push(`            if dist == "uniform_0_1":`);
  lines.push(`                return rng.uniform(0.0, 1.0, size=shape).astype(np.float32)`);
  lines.push(`            return rng.standard_normal(shape).astype(np.float32)`);
  lines.push(`        x0 = _draw(dist_x0, (n, d))`);
  lines.push(`        v_raw = _draw(dist_v, (n, d))`);
  lines.push(`        v = (float(v_scale) * v_raw).astype(np.float32)`);
  lines.push(`        idx = np.arange(t + 1, dtype=np.float32)`);
  lines.push(`        traj = x0[:, np.newaxis, :] + v[:, np.newaxis, :] * idx[np.newaxis, :, np.newaxis]`);
  lines.push(`        x = traj[:, :t, :].astype(np.float32)`);
  lines.push(`        y = traj[:, 1 : t + 1, :].astype(np.float32)`);
  lines.push(`        if output_distribution == "additive_gaussian" and float(noise_level) > 0:`);
  lines.push(`            y = y.copy()`);
  lines.push(`            y[:, -1, :] = y[:, -1, :] + float(noise_level) * rng.standard_normal((n, d)).astype(np.float32)`);
  lines.push(`        return x, y`);
  lines.push(``);
  lines.push(`    x_train, y_train = _sample_rows(train_size, position_distribution, velocity_distribution, float(velocity_scale))`);
  lines.push(
    `    x_test, y_test = _sample_rows(test_size, position_distribution, velocity_distribution, float(velocity_scale)) if int(test_size) > 0 else (None, None)`,
  );
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseUniformLinearMotionDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<UniformLinearMotionDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<UniformLinearMotionDatasetNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrderRaw: string[] = [];
  for (const row of p.params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (camel === "x1Distribution" || camel === "velocityDistribution") {
      paramOrderRaw.push("velocityDistribution");
      patch.velocityDistribution = String(val) as never;
      continue;
    }
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") extras[camel] = val;
      continue;
    }
    paramOrderRaw.push(camel);
    const key = camel as keyof UniformLinearMotionDatasetNodeData;
    if (key === "positionDistribution" || key === "outputDistribution") {
      patch[key] = String(val) as never;
      continue;
    }
    if (key === "noiseLevel" || key === "velocityScale") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      patch[key] = n as never;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[key] = n as never;
  }
  const paramOrder: string[] = [];
  const seen = new Set<string>();
  for (const k of paramOrderRaw) {
    const canon = k === "x1Distribution" ? "velocityDistribution" : k;
    if (seen.has(canon)) continue;
    seen.add(canon);
    paramOrder.push(canon);
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
