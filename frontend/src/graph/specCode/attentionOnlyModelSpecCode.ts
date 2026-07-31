import type { AttentionOnlyModelNodeData } from "../../components/nodes/attentionOnlyModelDefaults";
import { defaultAttentionOnlyModelData } from "../../components/nodes/attentionOnlyModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "embedDim",
  "numHeads",
  "contextLength",
  "causalAttention",
  "localMixingKernel",
  "seed",
]);

export const DEFAULT_ATTENTION_ONLY_SPEC_NAME = "AttentionOnlyModel";

export const DEFAULT_ATTENTION_ONLY_PARAM_ORDER: (keyof AttentionOnlyModelNodeData)[] = [
  "embedDim",
  "numHeads",
  "contextLength",
  "causalAttention",
  "localMixingKernel",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyIntDefault(key: keyof AttentionOnlyModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function yesNoToBoolPy(v: unknown): string {
  const s = String(firstScalar(v) ?? "yes").trim().toLowerCase();
  return s === "no" || s === "false" || s === "0" ? "False" : "True";
}

function parseClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function generateAttentionOnlyModelSpecCode(
  d: AttentionOnlyModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_ATTENTION_ONLY_SPEC_NAME;
  const lines: string[] = [
    `# Self-attention on activations [B, L, d] -> [B, L, d].`,
    `# Token CE training uses AttentionTokenPredictBundle: token embed -> optional`,
    `#   residual causal depthwise mix (local_mixing_kernel) -> this block -> lm_head on last position.`,
    `# See comfy_research/engine/attention_only_model.py.`,
    ``,
    `import numpy as np`,
    `import torch`,
    `import torch.nn as nn`,
    ``,
    `class ${name}(torch.nn.Module):`,
    `    def __init__(`,
    `        self,`,
  ];
  const merged = { ...defaultAttentionOnlyModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_ATTENTION_ONLY_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof AttentionOnlyModelNodeData;
    const sn = camelToSnakeCase(String(ck));
    if (ck === "causalAttention") {
      lines.push(`        ${sn}: bool = ${yesNoToBoolPy(merged[ck])},`);
    } else {
      lines.push(`        ${sn}: int = ${formatPyIntDefault(ck, merged[ck])},`);
    }
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(
    `        # Matches comfy_research/engine/attention_only_model.py (trainer may wrap for token CE; seed via torch.manual_seed in trainer_run).`,
  );
  lines.push(`        if embed_dim < 1 or context_length < 1 or num_heads < 1:`);
  lines.push(`            raise ValueError("embed_dim, context_length, num_heads must be >= 1")`);
  lines.push(`        if embed_dim % num_heads != 0:`);
  lines.push(`            raise ValueError("embed_dim must be divisible by num_heads")`);
  lines.push(`        self.embed_dim = int(embed_dim)`);
  lines.push(`        self.context_length = int(context_length)`);
  lines.push(`        self.num_heads = int(num_heads)`);
  lines.push(`        self.head_dim = self.embed_dim // self.num_heads`);
  lines.push(`        self.causal_attention = bool(causal_attention)`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        self.local_mixing_kernel = int(local_mixing_kernel)`);
  lines.push(`        d = self.embed_dim`);
  lines.push(`        self.w_q = nn.Linear(d, d, bias=True)`);
  lines.push(`        self.w_k = nn.Linear(d, d, bias=True)`);
  lines.push(`        self.w_v = nn.Linear(d, d, bias=True)`);
  lines.push(`        self.w_o = nn.Linear(d, d, bias=True)`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(
    `        """x float [batch, seq_len, embed_dim] with seq_len == context_length. Returns same shape."""`,
  );
  lines.push(`        if x.dim() != 3:`);
  lines.push(`            raise ValueError("x must be 3D [batch, seq_len, embed_dim]")`);
  lines.push(`        b, l, d = x.shape`);
  lines.push(`        if l != self.context_length:`);
  lines.push(
    `            raise ValueError(` +
      `f"seq_len {l} != model context_length {self.context_length}")`,
  );
  lines.push(`        if d != self.embed_dim:`);
  lines.push(`            raise ValueError(f"embed dim {d} != {self.embed_dim}")`);
  lines.push(`        h, hd = self.num_heads, self.head_dim`);
  lines.push(`        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)`);
  lines.push(`        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)`);
  lines.push(`        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)`);
  lines.push(`        scale = float(hd) ** -0.5`);
  lines.push(`        att = (q @ k.transpose(-2, -1)) * scale`);
  lines.push(`        if self.causal_attention:`);
  lines.push(`            causal = torch.triu(`);
  lines.push(`                torch.ones(l, l, device=x.device, dtype=torch.bool),`);
  lines.push(`                diagonal=1,`);
  lines.push(`            )`);
  lines.push(`            att = att.masked_fill(causal, float("-inf"))`);
  lines.push(`        attn = torch.softmax(att, dim=-1)`);
  lines.push(`        mixed = attn @ v`);
  lines.push(`        y = mixed.transpose(1, 2).reshape(b, l, d)`);
  lines.push(`        return self.w_o(y)`);
  lines.push(``);
  lines.push(`    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:`);
  lines.push(`        """Weight matrices for observables (Q/K/V/O projections)."""`);
  lines.push(`        with torch.no_grad():`);
  lines.push(`            return {`);
  lines.push(`                "w_q": self.w_q.weight.detach().float().cpu().numpy(),`);
  lines.push(`                "w_k": self.w_k.weight.detach().float().cpu().numpy(),`);
  lines.push(`                "w_v": self.w_v.weight.detach().float().cpu().numpy(),`);
  lines.push(`                "w_o": self.w_o.weight.detach().float().cpu().numpy(),`);
  lines.push(`            }`);
  return lines.join("\n");
}

export function parseAttentionOnlyModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<AttentionOnlyModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseClassInitHeader(code);
  if (!cls.error) {
    specName = cls.className;
    const rows = cls.paramsChunk
      .split("\n")
      .map((l) => l.replace(/#.*$/, "").trim())
      .filter((l) => l.length > 0);
    for (const line of rows) {
      const noComma = line.endsWith(",") ? line.slice(0, -1).trim() : line;
      const pm = noComma.match(/^(\w+)\s*:\s*[^=]+=\s*(.+)$/);
      if (!pm) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Bad parameter line: ${noComma}` };
      }
      params.push({ snakeName: pm[1]!, rawValue: pm[2]! });
    }
  } else {
    const p = parsePythonFunctionSpecHeader(code);
    if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: cls.error };
    specName = p.funcName;
    params = p.params.map((x) => ({ snakeName: x.snakeName, rawValue: x.rawValue }));
  }
  if (params.length === 0) {
    return { specName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<AttentionOnlyModelNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
        extras[camel] = val;
      }
      continue;
    }
    paramOrder.push(camel);
    const key = camel as keyof AttentionOnlyModelNodeData;
    if (key === "causalAttention") {
      const b = val === true || String(val).toLowerCase() === "true" || String(val).toLowerCase() === "yes";
      patch[key] = (b ? "yes" : "no") as never;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[key] = n as never;
  }
  return { specName, paramOrder, patch, extras };
}
