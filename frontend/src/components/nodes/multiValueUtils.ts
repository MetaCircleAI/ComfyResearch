/** One value or several (train runs cartesian product over multi-valued fields). */
export type ListOr1<T> = T | T[];

export function isMultiChoice<T>(v: ListOr1<T> | undefined | null): boolean {
  return Array.isArray(v) && v.length > 1;
}

export function intChoices(x: unknown, fallback: number): number[] {
  if (Array.isArray(x)) {
    const arr = x.map((v) => Number(v)).filter((n) => Number.isInteger(n));
    return arr.length ? arr : [fallback];
  }
  if (typeof x === "number" && Number.isInteger(x)) return [x];
  const n = Number(x);
  return Number.isInteger(n) ? [n] : [fallback];
}

export function floatChoices(x: unknown, fallback: number): number[] {
  if (Array.isArray(x)) {
    const arr = x.map((v) => Number(v)).filter((n) => Number.isFinite(n));
    return arr.length ? arr : [fallback];
  }
  if (typeof x === "number" && Number.isFinite(x)) return [x];
  const n = Number(x);
  return Number.isFinite(n) ? [n] : [fallback];
}

export function packIntList(values: number[]): number | number[] {
  const v = values.filter((n) => Number.isInteger(n));
  if (!v.length) return 0;
  return v.length === 1 ? v[0]! : v;
}

export function packFloatList(values: number[]): number | number[] {
  const v = values.filter((n) => Number.isFinite(n));
  if (!v.length) return 0;
  return v.length === 1 ? v[0]! : v;
}

export function parseIntList(s: string, min?: number, max?: number): number[] {
  const parts = s.split(/[,;\s]+/).map((x) => x.trim()).filter(Boolean);
  const out: number[] = [];
  for (const p of parts) {
    const n = Number(p);
    if (!Number.isInteger(n)) throw new Error(`Not an integer: ${p}`);
    if (min !== undefined && n < min) throw new Error(`Must be >= ${min}: ${n}`);
    if (max !== undefined && n > max) throw new Error(`Must be <= ${max}: ${n}`);
    out.push(n);
  }
  return out;
}

export function parseFloatList(s: string, opts?: { min?: number; max?: number; positiveOnly?: boolean }): number[] {
  const parts = s.split(/[,;\s]+/).map((x) => x.trim()).filter(Boolean);
  const out: number[] = [];
  for (const p of parts) {
    const raw = p.replaceAll(",", "");
    const n = Number(raw);
    if (!Number.isFinite(n)) throw new Error(`Not a number: ${p}`);
    if (opts?.positiveOnly && n <= 0) throw new Error(`Must be positive: ${p}`);
    if (opts?.min !== undefined && n < opts.min) throw new Error(`Must be >= ${opts.min}`);
    if (opts?.max !== undefined && n > opts.max) throw new Error(`Must be <= ${opts.max}`);
    out.push(n);
  }
  return out;
}

export function enumChoices<T extends string>(
  x: unknown,
  allowed: ReadonlySet<T>,
  fallback: T,
  /** When true, an empty array input yields `[]` instead of `[fallback]`. */
  allowEmpty = false,
): T[] {
  if (Array.isArray(x)) {
    const out: T[] = [];
    for (const v of x) {
      const s = String(v) as T;
      if (allowed.has(s)) out.push(s);
    }
    if (out.length === 0 && allowEmpty) return [];
    return out.length ? out : [fallback];
  }
  if (typeof x === "string" && allowed.has(x as T)) return [x as T];
  return [fallback];
}

export function packEnumList<T extends string>(values: T[]): ListOr1<T> {
  const uniq = [...new Set(values)];
  if (uniq.length === 0) return [];
  if (uniq.length === 1) return uniq[0]!;
  return uniq;
}
