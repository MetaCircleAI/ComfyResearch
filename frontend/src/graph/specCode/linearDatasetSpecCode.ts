import type { InputDistributionId, LinearDatasetNodeData, OutputDistributionId } from "../../components/nodes/linearDatasetDefaults";
import {
  defaultLinearDatasetData,
  MEMORIZATION_A_DATASET_SPEC_NAME,
  MEMORIZATION_B_DATASET_SPEC_NAME,
} from "../../components/nodes/linearDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "inputDim",
  "outputDim",
  "vocabSize",
  "inputDistribution",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "alpha",
  "seed",
]);

export const DEFAULT_LINEAR_DATASET_SPEC_NAME = "LinearDataset";

/** Field order for ``linear_dataset`` (no memorization-only ``alpha``). */
export const DEFAULT_LINEAR_DATASET_PARAM_ORDER: (keyof LinearDatasetNodeData)[] = [
  "inputDim",
  "outputDim",
  "inputDistribution",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "seed",
];

/** Memorization A: linear order with ``alpha`` before ``seed``. */
export const DEFAULT_MEMORIZATION_A_DATASET_PARAM_ORDER: (keyof LinearDatasetNodeData)[] = [
  "inputDim",
  "outputDim",
  "inputDistribution",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "alpha",
  "seed",
];

/** Memorization B: no continuous ``inputDistribution`` (inputs are class ids → one-hot rows). */
export const DEFAULT_MEMORIZATION_B_DATASET_PARAM_ORDER: (keyof LinearDatasetNodeData)[] = [
  "vocabSize",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "alpha",
  "seed",
];

function pyTypeForKey(key: keyof LinearDatasetNodeData): string {
  if (key === "inputDistribution" || key === "outputDistribution") return "str";
  if (key === "noiseLevel" || key === "alpha") return "float";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof LinearDatasetNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "inputDistribution" || key === "outputDistribution") {
    return JSON.stringify(String(s));
  }
  if (typeof s === "number" && !Number.isInteger(s)) return String(s);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function pyTypeForExtra(val: unknown): string {
  if (typeof val === "boolean") return "bool";
  if (typeof val === "string") return "str";
  if (typeof val === "number") return Number.isInteger(val) ? "int" : "float";
  return "float";
}

function formatPyExtra(val: unknown): string {
  if (typeof val === "boolean") return val ? "True" : "False";
  if (typeof val === "string") return JSON.stringify(val);
  if (typeof val === "number") return Number.isInteger(val) ? String(val) : String(val);
  return "0.0";
}

export function generateLinearDatasetSpecCode(
  d: LinearDatasetNodeData,
  order: string[],
  specName: string,
  /** React Flow node ``type`` — ``linear_dataset`` must not emit memorization spec name or ``alpha``. */
  graphNodeType: string = "linear_dataset",
): string {
  const isMemorizationA = graphNodeType === "memorization_a_dataset";
  const isMemorizationB = graphNodeType === "memorization_b_dataset";
  const isMemorization = isMemorizationA || isMemorizationB;
  let name = specName.trim() || DEFAULT_LINEAR_DATASET_SPEC_NAME;
  if (!isMemorization && name === MEMORIZATION_A_DATASET_SPEC_NAME) {
    name = DEFAULT_LINEAR_DATASET_SPEC_NAME;
  }
  if (!isMemorization && name === MEMORIZATION_B_DATASET_SPEC_NAME) {
    name = DEFAULT_LINEAR_DATASET_SPEC_NAME;
  }
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultLinearDatasetData(), ...d };
  if (isMemorizationB && (merged.vocabSize === undefined || merged.vocabSize === null)) {
    merged.vocabSize = merged.outputDim;
  }
  const defaultOrder = isMemorizationB
    ? DEFAULT_MEMORIZATION_B_DATASET_PARAM_ORDER
    : isMemorizationA
      ? DEFAULT_MEMORIZATION_A_DATASET_PARAM_ORDER
      : DEFAULT_LINEAR_DATASET_PARAM_ORDER;
  const effOrder = order.length ? order : [...defaultOrder];
  const keys: string[] = [];
  for (const k of effOrder) {
    if (!isMemorization && k === "alpha") continue;
    if (isMemorizationB && k === "inputDistribution") continue;
    if (isMemorizationB && (k === "inputDim" || k === "outputDim")) continue;
    if (KNOWN_KEYS.has(k)) keys.push(k);
    else if (merged.extras && k in merged.extras) keys.push(k);
  }
  for (const k of keys) {
    if (KNOWN_KEYS.has(k)) {
      const ck = k as keyof LinearDatasetNodeData;
      const sn = camelToSnakeCase(String(ck));
      const pyT = pyTypeForKey(ck);
      const def = formatPyDefault(ck, merged[ck]);
      lines.push(`    ${sn}: ${pyT} = ${def},`);
    } else {
      const sn = camelToSnakeCase(k);
      const ex = merged.extras as Record<string, string | number | boolean>;
      const v = ex[k];
      const pyT = pyTypeForExtra(v);
      const def = formatPyExtra(v);
      lines.push(`    ${sn}: ${pyT} = ${def},`);
    }
  }
  lines.push(`):`);
  const doc =
    isMemorizationB
      ? "Memorization B: one-hot inputs and independent class labels from the same prior family (see node info)."
      : isMemorizationA
        ? "Memorization A: random continuous x; class labels sampled independently of x from the configured prior (see node info)."
        : "Synthetic linear regression (y = x @ W.T + optional noise) when labels are Gaussian; class-prior labels if output_distribution selects uniform/power-law/exponential.";
  lines.push(`    """${doc}"""`);
  lines.push(`    import numpy as np`);
  lines.push(``);
  lines.push(`    rng = np.random.default_rng(seed)`);
  lines.push(`    eff_noise = float(noise_level)`);
  lines.push(`    is_mem_b = bool(${isMemorizationB ? "True" : "False"})`);
  lines.push(`    is_mem_a = bool(${isMemorizationA ? "True" : "False"})`);
  lines.push(`    vocab = int(vocab_size) if 'vocab_size' in locals() else int(output_dim)`);
  lines.push(`    d_in = vocab if is_mem_b else int(input_dim)`);
  lines.push(`    d_out = vocab if is_mem_b else int(output_dim)`);
  lines.push(`    alpha_v = float(alpha) if 'alpha' in locals() else 1.0`);
  lines.push(`    out_dist = str(output_distribution)`);
  lines.push(`    af_mode = "none"`);
  lines.push(
    `    is_mem = bool(is_mem_a) or (out_dist in {"uniform_class_probs", "power_law_class_probs", "exponential_class_probs"})`,
  );
  lines.push(`    w = rng.standard_normal((d_out, d_in)) / np.sqrt(max(1, d_in))`);
  lines.push(``);
  lines.push(`    def _sample_x(n: int, dist: str):`);
  lines.push(`        n = int(max(1, n))`);
  lines.push(`        if dist == "uniform_neg1_1":`);
  lines.push(`            return rng.uniform(-1.0, 1.0, size=(n, d_in)).astype(np.float32)`);
  lines.push(`        if dist == "uniform_0_1":`);
  lines.push(`            return rng.uniform(0.0, 1.0, size=(n, d_in)).astype(np.float32)`);
  lines.push(`        return rng.standard_normal((n, d_in)).astype(np.float32)`);
  lines.push(``);
  lines.push(`    def _class_probs(d: int, dist: str, av: float):`);
  lines.push(`        if d < 2:`);
  lines.push(`            raise ValueError("memorization dataset requires class dimension >= 2")`);
  lines.push(`        ranks = np.arange(1, d + 1, dtype=np.float64)`);
  lines.push(`        if dist == "power_law_class_probs":`);
  lines.push(`            a = max(float(av), 1e-8)`);
  lines.push(`            un = np.power(ranks, -a)`);
  lines.push(`        elif dist == "exponential_class_probs":`);
  lines.push(`            a = max(float(av), 1e-8)`);
  lines.push(`            un = np.exp(-a * ranks)`);
  lines.push(`        else:`);
  lines.push(`            un = np.ones((d,), dtype=np.float64)`);
  lines.push(`        z = float(np.sum(un))`);
  lines.push(`        if (not np.isfinite(z)) or z <= 0:`);
  lines.push(`            raise ValueError("invalid memorization class distribution")`);
  lines.push(`        return (un / z).astype(np.float64)`);
  lines.push(``);
  lines.push(`    def _mem_b_xy(n: int):`);
  lines.push(`        n = int(n)`);
  lines.push(`        if n <= 0:`);
  lines.push(`            return np.zeros((0, d_in), dtype=np.float32), np.zeros((0,), dtype=np.int64)`);
  lines.push(`        pin = _class_probs(d_in, out_dist, alpha_v)`);
  lines.push(`        x_idx = rng.choice(d_in, size=(n,), p=pin).astype(np.int64)`);
  lines.push(`        x_oh = np.zeros((n, d_in), dtype=np.float32)`);
  lines.push(`        x_oh[np.arange(n, dtype=np.int64), x_idx] = 1.0`);
  lines.push(`        pout = _class_probs(d_out, out_dist, alpha_v)`);
  lines.push(`        y = rng.choice(d_out, size=(n,), p=pout).astype(np.int64)`);
  lines.push(`        return x_oh, y`);
  lines.push(``);
  lines.push(`    if is_mem_b:`);
  lines.push(`        x_train, y_train = _mem_b_xy(int(train_size))`);
  lines.push(`    else:`);
  lines.push(`        x_train = _sample_x(train_size, input_distribution)`);
  lines.push(`        if af_mode == "af1":`);
  lines.push(`            base = np.tanh(x_train @ w.T).astype(np.float32)`);
  lines.push(`            y_train = ((1.0 - local_mix) * base + local_mix * np.roll(base, 1, axis=1) + 0.12 * base.mean(axis=1, keepdims=True)).astype(np.float32)`);
  lines.push(`        elif af_mode == "af2":`);
  lines.push(`            h = np.tanh(x_train @ w[:, : min(d_in, max(8, min(64, d_out)))].T).astype(np.float32)`);
  lines.push(`            pair = 0.5 * (h[:, :, None] + h[:, None, :])`);
  lines.push(`            tri = np.tanh(pair + 0.2 * np.einsum("bij,bjk->bik", pair, pair) / max(1.0, h.shape[1]))`);
  lines.push(`            y_train = tri.reshape(tri.shape[0], -1)[:, :d_out].astype(np.float32)`);
  lines.push(`            for _ in range(max(1, recycle_steps) - 1):`);
  lines.push(`                y_train = (0.6 * y_train + 0.4 * np.tanh(y_train)).astype(np.float32)`);
  lines.push(`        elif af_mode == "af3":`);
  lines.push(`            clean = np.tanh(x_train @ w.T).astype(np.float32)`);
  lines.push(`            eps = rng.standard_normal(clean.shape).astype(np.float32)`);
  lines.push(`            y_train = eps`);
  lines.push(`            if clean.shape[1] == x_train.shape[1]:`);
  lines.push(`                x_train = (clean + diffusion_noise_scale * eps).astype(np.float32)`);
  lines.push(`        elif is_mem:`);
  lines.push(`            probs = _class_probs(d_out, out_dist, alpha_v)`);
  lines.push(`            y_train = rng.choice(d_out, size=(int(train_size),), p=probs).astype(np.int64)`);
  lines.push(`        else:`);
  lines.push(`            y_train = (x_train @ w.T).astype(np.float32)`);
  lines.push(`            if out_dist == "additive_gaussian" and eff_noise > 0:`);
  lines.push(`                y_train = y_train + eff_noise * rng.standard_normal(y_train.shape).astype(np.float32)`);
  lines.push(``);
  lines.push(`    if is_mem_b:`);
  lines.push(`        x_test, y_test = _mem_b_xy(int(test_size)) if int(test_size) > 0 else (None, None)`);
  lines.push(`    else:`);
  lines.push(`        x_test = _sample_x(test_size, input_distribution) if int(test_size) > 0 else None`);
  lines.push(`        if x_test is not None:`);
  lines.push(`            if af_mode == "af1":`);
  lines.push(`                base = np.tanh(x_test @ w.T).astype(np.float32)`);
  lines.push(`                y_test = ((1.0 - local_mix) * base + local_mix * np.roll(base, 1, axis=1) + 0.12 * base.mean(axis=1, keepdims=True)).astype(np.float32)`);
  lines.push(`            elif af_mode == "af2":`);
  lines.push(`                h = np.tanh(x_test @ w[:, : min(d_in, max(8, min(64, d_out)))].T).astype(np.float32)`);
  lines.push(`                pair = 0.5 * (h[:, :, None] + h[:, None, :])`);
  lines.push(`                tri = np.tanh(pair + 0.2 * np.einsum("bij,bjk->bik", pair, pair) / max(1.0, h.shape[1]))`);
  lines.push(`                y_test = tri.reshape(tri.shape[0], -1)[:, :d_out].astype(np.float32)`);
  lines.push(`            elif af_mode == "af3":`);
  lines.push(`                clean = np.tanh(x_test @ w.T).astype(np.float32)`);
  lines.push(`                eps = rng.standard_normal(clean.shape).astype(np.float32)`);
  lines.push(`                y_test = eps`);
  lines.push(`                if clean.shape[1] == x_test.shape[1]:`);
  lines.push(`                    x_test = (clean + diffusion_noise_scale * eps).astype(np.float32)`);
  lines.push(`            elif is_mem:`);
  lines.push(`                probs = _class_probs(d_out, out_dist, alpha_v)`);
  lines.push(`                y_test = rng.choice(d_out, size=(int(test_size),), p=probs).astype(np.int64)`);
  lines.push(`            else:`);
  lines.push(`                y_test = (x_test @ w.T).astype(np.float32)`);
  lines.push(`                if out_dist == "additive_gaussian" and eff_noise > 0:`);
  lines.push(`                    y_test = y_test + eff_noise * rng.standard_normal(y_test.shape).astype(np.float32)`);
  lines.push(`        else:`);
  lines.push(`            y_test = None`);
  lines.push(``);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseLinearDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<LinearDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  if (p.params.length === 0) {
    return { specName: p.funcName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<LinearDatasetNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of p.params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
        extras[camel] = val;
        paramOrder.push(camel);
      }
      continue;
    }
    paramOrder.push(camel);
    const key = camel as keyof LinearDatasetNodeData;
    if (
      key === "inputDim" ||
      key === "outputDim" ||
      key === "vocabSize" ||
      key === "trainSize" ||
      key === "testSize" ||
      key === "seed"
    ) {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n) || !Number.isInteger(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
      }
      patch[key] = n as never;
    } else if (key === "noiseLevel" || key === "alpha") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      }
      patch[key] = n as never;
    } else if (key === "inputDistribution") {
      patch[key] = String(val) as InputDistributionId;
    } else if (key === "outputDistribution") {
      patch[key] = String(val) as OutputDistributionId;
    }
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
