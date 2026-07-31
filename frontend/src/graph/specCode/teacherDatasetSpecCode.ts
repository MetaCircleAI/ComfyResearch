import type { TeacherDatasetNodeData } from "../../components/nodes/teacherDatasetDefaults";
import { defaultTeacherDatasetData } from "../../components/nodes/teacherDatasetDefaults";
import { camelToSnakeCase, parsePythonFunctionSpecHeader } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set<string>();

export const DEFAULT_TEACHER_DATASET_SPEC_NAME = "TeacherDataset";

export const DEFAULT_TEACHER_DATASET_PARAM_ORDER: (keyof TeacherDatasetNodeData)[] = [];

export function generateTeacherDatasetSpecCode(
  d: TeacherDatasetNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_TEACHER_DATASET_SPEC_NAME;
  const lines: string[] = [`def ${name}(`];
  const merged = { ...defaultTeacherDatasetData(), ...d };
  const effOrder = order.length ? order : [...DEFAULT_TEACHER_DATASET_PARAM_ORDER];
  const keys: string[] = [];
  for (const k of effOrder) {
    if (KNOWN_KEYS.has(k)) keys.push(k);
    else if (merged.extras && k in merged.extras) keys.push(k);
  }
  for (const k of keys) {
    if (!KNOWN_KEYS.has(k)) {
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
      `    Label incoming train/test sample tensors with y = f_teacher(x).`,
      `    Runtime uses comfy_research.engine.teacher_dataset_runtime.teacher_labels_numpy.`,
      `    """`,
      `    import torch.nn as nn`,
      `    from comfy_research.engine.teacher_dataset_runtime import teacher_labels_numpy`,
      ``,
      `    import numpy as np`,
      `    x = np.zeros((8, 10), dtype=np.float32)`,
      `    teacher = nn.Sequential(nn.Linear(10, 1))`,
      `    y = teacher_labels_numpy(teacher, x)`,
      `    return x, y`,
    ],
  );
  return lines.join("\n");
}

export function parseTeacherDatasetSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<TeacherDatasetNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  return { specName: p.funcName, paramOrder: [], patch: {}, extras: {} };
}
