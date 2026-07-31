/**
 * Parse a minimal Python 3 ``def name(...) -> ...:`` block with one parameter per line
 * ``name: type = default`` (training / viz spec style).
 */

export type ParsedPyParam = {
  snakeName: string;
  pyType: string;
  rawValue: string;
};

export type ParsedPythonFuncSpec = {
  funcName: string;
  params: ParsedPyParam[];
  error?: string;
};

function stripPythonComments(line: string): string {
  const i = line.indexOf("#");
  if (i === -1) return line;
  return line.slice(0, i);
}

/** Parse default expression (no commas at top level in our templates). */
export function parsePythonDefault(raw: string): unknown {
  const t = raw.trim().replace(/,\s*$/, "");
  if (t === "True") return true;
  if (t === "False") return false;
  if (t === "None") return null;
  if (
    (t.startsWith('"') && t.endsWith('"')) ||
    (t.startsWith("'") && t.endsWith("'"))
  ) {
    const q = t.slice(1, -1);
    return q.replace(/\\n/g, "\n");
  }
  if (/^-?\d+$/.test(t)) return parseInt(t, 10);
  if (/^-?\d*\.\d+([eE][+-]?\d+)?$/.test(t) || /^-?\d+[eE][+-]?\d+$/.test(t)) {
    return parseFloat(t);
  }
  return t;
}

export function snakeToCamelCase(snake: string): string {
  const p = snake.split("_").filter(Boolean);
  if (p.length === 0) return snake;
  return (
    p[0] +
    p
      .slice(1)
      .map((w) => (w.length ? w[0]!.toUpperCase() + w.slice(1).toLowerCase() : ""))
      .join("")
  );
}

export function camelToSnakeCase(key: string): string {
  return key.replace(/([A-Z])/g, "_$1").replace(/^_/, "").toLowerCase();
}

/**
 * Extract ``def name(`` ... ``):`` header; parameters may span multiple lines.
 */
export function parsePythonFunctionSpecHeader(code: string): ParsedPythonFuncSpec {
  const trimmed = code.trim();
  const m = trimmed.match(/^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([\s\S]*?)\)\s*(?:->\s*[^:]+)?\s*:/);
  if (!m) {
    return {
      funcName: "",
      params: [],
      error:
        "Expected a Python function header like: def MySpec( param: int = 1, ) -> None:",
    };
  }
  const funcName = m[1]!;
  const inner = m[2]!;
  const lines = inner
    .split("\n")
    .map((l) => stripPythonComments(l).trim())
    .filter((l) => l.length > 0);

  const params: ParsedPyParam[] = [];
  for (const line of lines) {
    const noComma = line.endsWith(",") ? line.slice(0, -1).trim() : line;
    let pm = noComma.match(/^(\w+)\s*:\s*([^=]+)=\s*(.+)$/);
    if (!pm) {
      // Allow `name type = value` without a colon (e.g. `scale float = 1.0`).
      const pm2 = noComma.match(/^(\w+)\s+(\w+)\s*=\s*(.+)$/);
      if (!pm2) {
        return {
          funcName,
          params: [],
          error: `Bad parameter line: ${noComma.slice(0, 80)}`,
        };
      }
      params.push({
        snakeName: pm2[1]!.trim(),
        pyType: pm2[2]!.trim(),
        rawValue: pm2[3]!.trim(),
      });
      continue;
    }
    params.push({
      snakeName: pm[1]!.trim(),
      pyType: pm[2]!.trim(),
      rawValue: pm[3]!.trim(),
    });
  }
  return { funcName, params };
}

export function setsEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const s = new Set(a);
  for (const x of b) {
    if (!s.has(x)) return false;
  }
  return true;
}
