type CompiledRule = {
  evaluate: (x: number) => number;
  normalizedRule: string;
};

function replaceFractions(input: string): string {
  let out = input;
  const fractionPattern = /\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g;
  let changed = true;
  while (changed) {
    changed = false;
    out = out.replace(fractionPattern, (_m, num: string, den: string) => {
      changed = true;
      return `((${num})/(${den}))`;
    });
  }
  return out;
}

function normalizeLatexRule(ruleLatex: string): string {
  let expr = ruleLatex.trim();
  if (!expr) expr = "x^2";
  expr = replaceFractions(expr);
  expr = expr.replace(/\\left|\\right/g, "");
  // e.g. \left|x\right| → |x| → abs(x)
  expr = expr.replace(/\|\s*([^|]+?)\s*\|/g, "abs($1)");
  expr = expr.replace(/\\cdot|\\times/g, "*");
  expr = expr.replace(/\\pi/g, "pi");
  expr = expr.replace(/\\sqrt\s*\{([^{}]+)\}/g, "sqrt($1)");
  expr = expr.replace(/\\sin/g, "sin");
  expr = expr.replace(/\\cos/g, "cos");
  expr = expr.replace(/\\tan/g, "tan");
  expr = expr.replace(/\\exp/g, "exp");
  expr = expr.replace(/\\log/g, "log");
  expr = expr.replace(/\\ln/g, "log");
  expr = expr.replace(/\\abs/g, "abs");
  expr = expr.replace(/\{/g, "(").replace(/\}/g, ")");
  expr = expr.replace(/\s+/g, "");
  expr = expr.replace(/(\d)(x|pi|\()/g, "$1*$2");
  expr = expr.replace(/(x|pi|\))(\d|x|pi|\()/g, "$1*$2");
  expr = expr.replace(/\^/g, "**");
  return expr;
}

export function compileElementwiseLatexRule(ruleLatex: string): CompiledRule {
  const normalizedRule = normalizeLatexRule(ruleLatex);
  if (!/^[0-9a-zA-Z+\-*/().,_]*$/.test(normalizedRule)) {
    throw new Error("Rule contains unsupported characters.");
  }
  if (!/[x]/.test(normalizedRule)) {
    throw new Error("Rule must include variable x.");
  }
  const evaluateImpl = new Function(
    "x",
    "const { sin, cos, tan, exp, log, sqrt, abs } = Math; const pi = Math.PI; return (" +
      normalizedRule +
      ");",
  ) as (x: number) => number;
  const evaluate = (x: number) => {
    const y = Number(evaluateImpl(x));
    if (!Number.isFinite(y)) throw new Error("Rule produced non-finite output.");
    return y;
  };
  return { evaluate, normalizedRule };
}
