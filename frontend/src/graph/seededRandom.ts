/** Deterministic RNG in [0, 1) from a 32-bit seed (Mulberry32). */
export function createSeededRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** Parse UI seed: empty / invalid → fallback (still deterministic for a given fallback). */
export function parseGraphAssistSeed(input: string, fallback: number): number {
  const s = input.trim();
  if (!s) return fallback >>> 0;
  const n = Number(s);
  if (!Number.isFinite(n)) return fallback >>> 0;
  return Math.floor(n) >>> 0;
}

function pushRangeInclusive(out: number[], lo: number, hi: number): void {
  const a = Math.floor(lo);
  const b = Math.floor(hi);
  const start = Math.min(a, b);
  const end = Math.max(a, b);
  for (let k = start; k <= end; k++) {
    out.push(k >>> 0);
  }
}

/** Accept common dash variants users may paste from rich text. */
const RANGE_SEP_RE_SRC = String.raw`[\-\u2010\u2011\u2012\u2013\u2014\u2212]`;

/**
 * Comma / semicolon-separated chunks; inside each chunk, whitespace-separated values.
 * A hyphenated pair of integers expands inclusively, e.g. `0 - 5` → 0,1,…,5 (same as `0-5`).
 */
export function parseGraphAssistSeeds(input: string): number[] {
  const out: number[] = [];
  const numHead = /^[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/;
  const rangeHead = new RegExp(`^\\d+\\s*${RANGE_SEP_RE_SRC}\\s*\\d+`);
  const rangeSplitRe = new RegExp(`\\s*${RANGE_SEP_RE_SRC}\\s*`);

  for (const rawChunk of input.split(/[,;]+/)) {
    const chunk = rawChunk.trim();
    if (!chunk) continue;
    let i = 0;
    while (i < chunk.length) {
      while (i < chunk.length && /\s/.test(chunk[i]!)) i++;
      if (i >= chunk.length) break;
      const tail = chunk.slice(i);
      const rangeM = tail.match(rangeHead);
      if (rangeM) {
        const [aStr, bStr] = rangeM[0].split(rangeSplitRe);
        const a = Number(aStr);
        const b = Number(bStr);
        if (Number.isFinite(a) && Number.isFinite(b)) {
          pushRangeInclusive(out, a, b);
        }
        i += rangeM[0].length;
        continue;
      }
      const numM = tail.match(numHead);
      if (numM) {
        const n = Number(numM[0]);
        if (Number.isFinite(n)) {
          out.push(Math.floor(n) >>> 0);
        }
        i += numM[0].length;
        continue;
      }
      i++;
    }
  }
  return out;
}
