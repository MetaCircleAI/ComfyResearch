/** One segment from splitting a string on KaTeX delimiters ``$...$`` / ``$$...$$`` (no nesting). */
export type LatexSegment =
  | { kind: "text"; text: string }
  | { kind: "math"; latex: string; display: boolean };

/** Split prose into alternating text / math segments using ``$...$`` and ``$$...$$`` (no nesting). */
export function parseLatexDelimited(src: string): LatexSegment[] {
  const out: LatexSegment[] = [];
  let i = 0;
  while (i < src.length) {
    const dd = src.indexOf("$$", i);
    const d1 = src.indexOf("$", i);
    if (dd !== -1 && (d1 === -1 || dd <= d1)) {
      if (dd > i) out.push({ kind: "text", text: src.slice(i, dd) });
      const end = src.indexOf("$$", dd + 2);
      if (end === -1) {
        out.push({ kind: "text", text: src.slice(dd) });
        break;
      }
      out.push({ kind: "math", latex: src.slice(dd + 2, end).trim(), display: true });
      i = end + 2;
    } else if (d1 !== -1) {
      if (d1 > i) out.push({ kind: "text", text: src.slice(i, d1) });
      const end = src.indexOf("$", d1 + 1);
      if (end === -1) {
        out.push({ kind: "text", text: src.slice(d1) });
        break;
      }
      out.push({ kind: "math", latex: src.slice(d1 + 1, end).trim(), display: false });
      i = end + 1;
    } else {
      if (i < src.length) out.push({ kind: "text", text: src.slice(i) });
      break;
    }
  }
  return out;
}
