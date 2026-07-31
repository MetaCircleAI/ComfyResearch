import type {
  InputDistributionId,
  OutputDistributionId,
  RandomInputDistributionNodeData,
} from "../../components/nodes/randomInputDistributionDefaults";
import { defaultRandomInputDistributionData } from "../../components/nodes/randomInputDistributionDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const INPUT_IDS = new Set(["standard_normal", "uniform_neg1_1", "uniform_0_1"]);
const NOISE_IDS = new Set(["additive_gaussian", "deterministic"]);

const KNOWN_KEYS = new Set([
  "inputDim",
  "inputDistribution",
  "noiseDistribution",
  "noiseLevel",
  "seed",
]);

export const DEFAULT_RANDOM_INPUT_DIST_SPEC_NAME = "RandomInputDistribution";

export const DEFAULT_RANDOM_INPUT_DIST_PARAM_ORDER: (keyof RandomInputDistributionNodeData)[] = [
  "inputDim",
  "inputDistribution",
  "noiseDistribution",
  "noiseLevel",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof RandomInputDistributionNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "number") return String(s);
  return JSON.stringify(String(s));
}

export function generateRandomInputDistributionSpecCode(
  d: RandomInputDistributionNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_RANDOM_INPUT_DIST_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultRandomInputDistributionData(), ...d };
  const effOrder = order.length ? order : [...DEFAULT_RANDOM_INPUT_DIST_PARAM_ORDER];
  const keys: string[] = [];
  for (const k of effOrder) {
    if (KNOWN_KEYS.has(k)) keys.push(k);
    else if (merged.extras && k in merged.extras) keys.push(k);
  }
  for (const k of keys) {
    if (KNOWN_KEYS.has(k)) {
      const ck = k as keyof RandomInputDistributionNodeData;
      const sn = camelToSnakeCase(String(ck));
      const pyT =
        ck === "inputDistribution" || ck === "noiseDistribution"
          ? "str"
          : ck === "noiseLevel"
            ? "float"
            : "int";
      lines.push(`    ${sn}: ${pyT} = ${formatPyDefault(ck, merged[ck])},`);
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
  lines.push(
    ...[
      `    """`,
      `    Same sampling as the trainer: comfy_research.engine.random_input_distribution_runtime`,
      `    (rng_from_random_input_distribution_data + sample_x_from_random_input_dict).`,
      `    """`,
      `    from comfy_research.engine.random_input_distribution_runtime import (`,
      `        rng_from_random_input_distribution_data,`,
      `        sample_x_from_random_input_dict,`,
      `    )`,
      ``,
      `    dd = {`,
      `        "inputDim": input_dim,`,
      `        "inputDistribution": input_distribution,`,
      `        "noiseDistribution": noise_distribution,`,
      `        "noiseLevel": noise_level,`,
      `        "seed": seed,`,
      `    }`,
      `    rng = rng_from_random_input_distribution_data(dd)`,
      `    n_preview = min(32, max(1, input_dim))`,
      `    return sample_x_from_random_input_dict(dd, n_preview, rng)`,
    ],
  );
  return lines.join("\n");
}

export function parseRandomInputDistributionSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<RandomInputDistributionNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  if (p.params.length === 0) {
    return { specName: p.funcName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<RandomInputDistributionNodeData> = {};
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
    const key = camel as keyof RandomInputDistributionNodeData;
    if (key === "inputDim" || key === "seed") {
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
      const s = String(val);
      if (!INPUT_IDS.has(s)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid inputDistribution: ${s}` };
      }
      patch[key] = s as InputDistributionId;
    } else if (key === "noiseDistribution") {
      const s = String(val);
      if (!NOISE_IDS.has(s)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid noiseDistribution: ${s}` };
      }
      patch[key] = s as OutputDistributionId;
    }
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
