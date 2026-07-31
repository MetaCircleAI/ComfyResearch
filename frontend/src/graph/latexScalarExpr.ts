import { create, all } from "mathjs";

const math = create(all, {});

function readBraced(s: string, i: number): { content: string; end: number } | null {
  if (s[i] !== "{") return null;
  let depth = 0;
  for (let k = i; k < s.length; k++) {
    const c = s[k]!;
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return { content: s.slice(i + 1, k), end: k + 1 };
    }
  }
  return null;
}

function stripDelims(s: string): string {
  let t = s.trim();
  if (t.startsWith("$$") && t.endsWith("$$") && t.length >= 4) t = t.slice(2, -2).trim();
  else if (t.startsWith("$") && t.endsWith("$") && t.length >= 2) t = t.slice(1, -1).trim();
  return t;
}

/** TeX subscripts `x_{12}` → mathjs identifier `x_12`. */
function normalizeSubscripts(s: string): string {
  return s.replace(/x_\{(\d+)\}/g, "x_$1");
}

function replaceUnaryLatexCommands(s: string): string {
  return s
    .replace(/\\left\s*/g, "")
    .replace(/\\right\s*/g, "")
    .replace(/\\cdot\b/g, "*")
    .replace(/\\times\b/g, "*")
    .replace(/\\div\b/g, "/")
    .replace(/\\pi\b/g, "pi")
    .replace(/\\infty\b/g, "Infinity")
    .replace(/\\sin\b/g, "sin")
    .replace(/\\cos\b/g, "cos")
    .replace(/\\tan\b/g, "tan")
    .replace(/\\log\b/g, "log10")
    .replace(/\\ln\b/g, "log")
    .replace(/\\exp\b/g, "exp")
    .replace(/\\min\b/g, "min")
    .replace(/\\max\b/g, "max")
    .replace(/\\mathrm\{([^}]*)\}/g, "$1")
    .replace(/\\text\{([^}]*)\}/g, "$1");
}

function collapseTeXBraces(s: string): string {
  let out = s;
  while (/\{[^{}]+\}/.test(out)) {
    out = out.replace(/\{([^{}]+)\}/g, "($1)");
  }
  return out;
}

function convertAllSqrtRaw(s: string, innerConvert: (src: string) => string): string {
  let out = s;
  for (;;) {
    const idx = out.indexOf("\\sqrt");
    if (idx < 0) return out;
    let p = idx + 5;
    while (p < out.length && /\s/.test(out[p]!)) p++;
    let rootStr: string | null = null;
    if (out[p] === "[") {
      const rb = out.indexOf("]", p);
      if (rb < 0) throw new Error("Unclosed \\sqrt index [...]");
      rootStr = out.slice(p + 1, rb);
      p = rb + 1;
      while (p < out.length && /\s/.test(out[p]!)) p++;
    }
    const body = readBraced(out, p);
    if (!body) throw new Error("\\sqrt must be followed by {...}");
    const inner = innerConvert(body.content);
    const rep =
      rootStr == null ? `sqrt(${inner})` : `nthRoot(${inner}, ${innerConvert(rootStr)})`;
    out = out.slice(0, idx) + rep + out.slice(body.end);
  }
}

function convertAllFracRaw(s: string, innerConvert: (src: string) => string): string {
  let out = s;
  for (;;) {
    const idx = out.indexOf("\\frac");
    if (idx < 0) return out;
    let p = idx + 5;
    while (p < out.length && /\s/.test(out[p]!)) p++;
    const num = readBraced(out, p);
    if (!num) throw new Error("\\frac needs {numerator}{denominator}");
    let p2 = num.end;
    while (p2 < out.length && /\s/.test(out[p2]!)) p2++;
    const den = readBraced(out, p2);
    if (!den) throw new Error("\\frac needs second {...}");
    const rep = `((${innerConvert(num.content)})/(${innerConvert(den.content)}))`;
    out = out.slice(0, idx) + rep + out.slice(den.end);
  }
}

export function latexScalarExprToMathJs(src: string): string {
  function innerConvert(raw: string): string {
    let s = stripDelims(raw.trim());
    s = normalizeSubscripts(s);
    s = s.replace(/\^\{([^}]+)\}/g, "^($1)");
    s = convertAllSqrtRaw(s, innerConvert);
    s = convertAllFracRaw(s, innerConvert);
    s = replaceUnaryLatexCommands(s);
    s = collapseTeXBraces(s);
    return s.trim();
  }
  return innerConvert(src);
}

function asFiniteNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (v && typeof v === "object" && "re" in v && typeof (v as { re: unknown }).re === "number") {
    const re = (v as { re: number }).re;
    const im = "im" in v && typeof (v as { im: unknown }).im === "number" ? (v as { im: number }).im : 0;
    if (Number.isFinite(re) && Number.isFinite(im) && Math.abs(im) < 1e-12) return re;
    return null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function evaluateLatexScalarExpr(
  latex: string,
  scope: Record<string, number>,
): { ok: true; value: number; mathJsExpr: string } | { ok: false; error: string; mathJsExpr?: string } {
  try {
    const mj = latexScalarExprToMathJs(latex);
    const node = math.parse(mj);
    const compiled = node.compile();
    const v = compiled.evaluate(scope) as unknown;
    const num = asFiniteNumber(v);
    if (num == null) return { ok: false, error: "Result is not a finite real number.", mathJsExpr: mj };
    return { ok: true, value: num, mathJsExpr: mj };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
