import type {
  InputDistributionId,
  OutputDistributionId,
} from "../../components/nodes/linearDatasetDefaults";
import type { SymbolicFuncDatasetNodeData } from "../../components/nodes/symbolicFuncDatasetDefaults";
import { defaultSymbolicFuncDatasetData } from "../../components/nodes/symbolicFuncDatasetDefaults";
import {
  camelToSnakeCase,
  parsePythonDefault,
  parsePythonFunctionSpecHeader,
  snakeToCamelCase,
} from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "equationLatex",
  "inputDim",
  "outputDim",
  "inputDistribution",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "seed",
]);

export const DEFAULT_SYMBOLIC_DATASET_SPEC_NAME = "SymbolicFuncDataset";

export const DEFAULT_SYMBOLIC_DATASET_PARAM_ORDER: (keyof SymbolicFuncDatasetNodeData)[] = [
  "equationLatex",
  "inputDim",
  "outputDim",
  "inputDistribution",
  "outputDistribution",
  "trainSize",
  "testSize",
  "noiseLevel",
  "seed",
];

function pyTypeForExtra(val: unknown): string {
  if (typeof val === "boolean") return "bool";
  if (typeof val === "string") return "str";
  // Numeric extras (bias, etc.) are always float in generated Python so `0` stays `0.0`.
  if (typeof val === "number") return "float";
  return "float";
}

function formatPyExtra(val: unknown): string {
  if (typeof val === "boolean") return val ? "True" : "False";
  if (typeof val === "string") return JSON.stringify(val);
  if (typeof val === "number") {
    if (Number.isInteger(val)) return `${val}.0`;
    return String(val);
  }
  return "0.0";
}

/** Undo accidental double-escaping of LaTeX backslashes (e.g. `\\sum` → `\sum`) from JSON/Python re-export. */
export function normalizeEquationLatex(s: string): string {
  let out = s;
  while (/\\\\[a-zA-Z]/.test(out)) {
    out = out.replace(/\\\\([a-zA-Z])/g, "\\$1");
  }
  return out;
}

function inferNumpyExpressionPreview(equationLatex: string): string {
  const clean = equationLatex
    .replace(/\s+/g, " ")
    .replace(/\\left|\\right/g, "")
    .trim();

  const toNumpyExpr = (src: string): string => {
    let out = src;

    // Basic function/operator normalization.
    out = out.replace(/\\cdot/g, "*");
    out = out.replace(/\\times/g, "*");
    out = out.replace(/\\pi/g, "np.pi");
    out = out.replace(/\\sin/g, "np.sin");
    out = out.replace(/\\cos/g, "np.cos");
    out = out.replace(/\\tan/g, "np.tan");
    out = out.replace(/\\exp/g, "np.exp");
    out = out.replace(/\\log/g, "np.log");
    out = out.replace(/\\sqrt/g, "np.sqrt");

    // x-indexed symbols.
    out = out.replace(/x_\{i\}|x_i/g, "x[:, :input_dim]");
    out = out.replace(/x_\{(\d+)\}|x_(\d+)/g, (_m, a, b) => `x[:, ${Math.max(0, Number(a ?? b) - 1)}]`);

    // \frac{a}{b}
    while (/\\frac\{[^{}]+\}\{[^{}]+\}/.test(out)) {
      out = out.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1) / ($2)");
    }

    // Common LaTeX forms for trig/log powers, e.g. \sin^{2}(...) -> np.sin(...) ** 2
    out = out.replace(
      /np\.(sin|cos|tan|exp|log|sqrt)\s*\^\{\s*([^{}]+)\s*\}\s*\(\s*([^()]*)\s*\)/g,
      (_m, fn, p, arg) => `np.${fn}(${arg}) ** (${p})`,
    );
    out = out.replace(
      /np\.(sin|cos|tan|exp|log|sqrt)\s*\^\s*([A-Za-z0-9_]+)\s*\(\s*([^()]*)\s*\)/g,
      (_m, fn, p, arg) => `np.${fn}(${arg}) ** (${p})`,
    );

    // Exponents: a^{b} and a^b.
    out = out.replace(/([A-Za-z0-9_\]\)\.:\s]+)\^\{([^{}]+)\}/g, "($1) ** ($2)");
    out = out.replace(/([A-Za-z0-9_\]\)\.:\s]+)\^([A-Za-z0-9_]+)/g, "($1) ** ($2)");

    // d symbol in sums.
    out = out.replace(/\bd\b/g, "input_dim");

    // tighten accidental extra spaces around slices.
    out = out.replace(/\s+/g, " ").trim();
    return out;
  };

  // Handle one-level sum form: \sum_{i=1}^{d} (...)  (+ optional outside arithmetic).
  const sumMatch = clean.match(/\\sum_\{i=1\}\^\{d\}\s*(.+)$/);
  if (sumMatch) {
    const after = toNumpyExpr(sumMatch[1]!.trim());
    return `y = np.sum(${after}, axis=1, keepdims=True)`;
  }

  // Generic expression fallback conversion.
  const generic = toNumpyExpr(clean);
  if (generic.length > 0) {
    return `y = ${generic}`;
  }
  return "y = symbolic_fn(x)  # compiled from equation_latex at runtime";
}

function pyTypeForKey(key: keyof SymbolicFuncDatasetNodeData): string {
  if (key === "equationLatex" || key === "inputDistribution" || key === "outputDistribution") return "str";
  if (key === "noiseLevel") return "float";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof SymbolicFuncDatasetNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "equationLatex" || key === "inputDistribution" || key === "outputDistribution") {
    return JSON.stringify(String(s));
  }
  if (typeof s === "number" && !Number.isInteger(s)) return String(s);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateSymbolicFuncDatasetSpecCode(
  d: SymbolicFuncDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_SYMBOLIC_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultSymbolicFuncDatasetData(), ...d };
  const effOrder = order.length ? order : [...DEFAULT_SYMBOLIC_DATASET_PARAM_ORDER];
  const keys: string[] = [];
  for (const k of effOrder) {
    if (KNOWN_KEYS.has(k)) keys.push(k);
    else if (merged.extras && k in merged.extras) keys.push(k);
  }
  for (const k of keys) {
    if (KNOWN_KEYS.has(k)) {
      const ck = k as keyof SymbolicFuncDatasetNodeData;
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
  lines.push(`    """Generate train/test data from a symbolic y(x) equation in LaTeX."""`);
  lines.push(`    # Equation should be self-contained (e.g. use 10*x_i instead of symbol f).`);
  lines.push(`    # Inferred NumPy preview from equation_latex:`);
  lines.push(`    ${inferNumpyExpressionPreview(String(merged.equationLatex))}`);
  lines.push(`    return {`);
  lines.push(`        "equationLatex": equation_latex,`);
  lines.push(`        "inputDim": input_dim,`);
  lines.push(`        "outputDim": output_dim,`);
  lines.push(`        "inputDistribution": input_distribution,`);
  lines.push(`        "outputDistribution": output_distribution,`);
  lines.push(`        "trainSize": train_size,`);
  lines.push(`        "testSize": test_size,`);
  lines.push(`        "noiseLevel": noise_level,`);
  lines.push(`        "seed": seed,`);
  lines.push(`    }`);
  return lines.join("\n");
}

export function parseSymbolicFuncDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<SymbolicFuncDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  if (p.params.length === 0) {
    return { specName: p.funcName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<SymbolicFuncDatasetNodeData> = {};
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
    const key = camel as keyof SymbolicFuncDatasetNodeData;
    if (key === "equationLatex") {
      patch[key] = normalizeEquationLatex(String(val)) as never;
    } else if (
      key === "inputDim" ||
      key === "outputDim" ||
      key === "trainSize" ||
      key === "testSize" ||
      key === "seed"
    ) {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n) || !Number.isInteger(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
      }
      patch[key] = n as never;
    } else if (key === "noiseLevel") {
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
