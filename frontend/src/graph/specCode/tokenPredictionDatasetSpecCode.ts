import type { TokenPredictionDatasetNodeData } from "../../components/nodes/tokenPredictionDatasetDefaults";
import { defaultTokenPredictionDatasetData } from "../../components/nodes/tokenPredictionDatasetDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const RETRIEVAL_MODES = new Set(["position", "content"] as const);

const KNOWN_KEYS = new Set([
  "retrievalMode",
  "vocabSize",
  "contextLength",
  "whichToken",
  "trainSize",
  "testSize",
  "seed",
]);

export const DEFAULT_TOKEN_PREDICTION_SPEC_NAME = "TokenRetrievalDataset";

export const DEFAULT_TOKEN_PREDICTION_PARAM_ORDER: (keyof TokenPredictionDatasetNodeData)[] = [
  "retrievalMode",
  "vocabSize",
  "contextLength",
  "whichToken",
  "trainSize",
  "testSize",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof TokenPredictionDatasetNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "retrievalMode") return JSON.stringify(s === "content" ? "content" : "position");
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateTokenPredictionDatasetSpecCode(
  d: TokenPredictionDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_TOKEN_PREDICTION_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultTokenPredictionDatasetData(), ...d };
  const effOrder = order.length ? order : [...DEFAULT_TOKEN_PREDICTION_PARAM_ORDER];
  const keys: string[] = [];
  for (const k of effOrder) {
    if (KNOWN_KEYS.has(k)) keys.push(k);
    else if (merged.extras && k in merged.extras) keys.push(k);
  }
  for (const k of keys) {
    if (KNOWN_KEYS.has(k)) {
      const ck = k as keyof TokenPredictionDatasetNodeData;
      const sn = camelToSnakeCase(String(ck));
      lines.push(`    ${sn}: int = ${formatPyDefault(ck, merged[ck])},`);
    } else {
      const sn = camelToSnakeCase(k);
      const ex = merged.extras as Record<string, string | number | boolean>;
      const v = ex[k]!;
      const pyT = typeof v === "boolean" ? "bool" : typeof v === "string" ? "str" : Number.isInteger(Number(v)) ? "int" : "float";
      const def =
        typeof v === "boolean"
          ? v
            ? "True"
            : "False"
          : typeof v === "string"
            ? JSON.stringify(v)
            : String(v);
      lines.push(`    ${sn}: ${pyT} = ${def},`);
    }
  }
  lines.push(`):`);
  lines.push(`    # Same construction as comfy_research/engine/trainer_run.py (attention layer / CE path).`);
  lines.push(
    `    """Random token sequences in [0, V). Mode='position': y=x[:, which_token]. Mode='content': y is nearest prior token to last token by absolute distance."""`,
  );
  lines.push(`    import numpy as np`);
  lines.push(``);
  lines.push(`    rng = np.random.default_rng(int(seed))`);
  lines.push(`    V = int(vocab_size)`);
  lines.push(`    L = int(context_length)`);
  lines.push(`    n_train = int(train_size)`);
  lines.push(`    n_test = int(test_size)`);
  lines.push(`    x_train = rng.integers(0, V, size=(n_train, L), dtype=np.int64) if n_train > 0 else None`);
  lines.push(`    x_test = rng.integers(0, V, size=(n_test, L), dtype=np.int64) if n_test > 0 else None`);
  lines.push(`    mode = str(retrieval_mode).strip().lower()`);
  lines.push(`    def _y(x):`);
  lines.push(`        if x is None:`);
  lines.push(`            return None`);
  lines.push(`        if mode == "content":`);
  lines.push(`            if x.shape[1] < 2:`);
  lines.push(`                raise ValueError("content retrieval mode requires context_length >= 2")`);
  lines.push(`            cur = x[:, -1][:, None]`);
  lines.push(`            prev = x[:, :-1]`);
  lines.push(`            dist = np.abs(prev - cur)`);
  lines.push(`            best = np.argmin(dist, axis=1)`);
  lines.push(`            return prev[np.arange(prev.shape[0]), best].astype(np.int64)`);
  lines.push(`        idx = int(which_token)`);
  lines.push(`        return x[:, idx].astype(np.int64)`);
  lines.push(`    y_train = _y(x_train)`);
  lines.push(`    y_test = _y(x_test)`);
  lines.push(`    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}`);
  return lines.join("\n");
}

export function parseTokenPredictionDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<TokenPredictionDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  if (p.params.length === 0) {
    return { specName: p.funcName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<TokenPredictionDatasetNodeData> = {};
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
    const key = camel as keyof TokenPredictionDatasetNodeData;
    if (camel === "retrievalMode") {
      const s = typeof val === "string" ? val.toLowerCase() : String(val).toLowerCase();
      if (!RETRIEVAL_MODES.has(s as "position" | "content")) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: "retrievalMode must be 'position' or 'content'" };
      }
      patch.retrievalMode = s as "position" | "content";
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[key] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
