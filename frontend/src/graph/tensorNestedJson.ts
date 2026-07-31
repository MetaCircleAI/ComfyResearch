/** Row-major flat values → nested arrays matching `shape` (same layout as model parameter JSON). */
export function reshapeFlatToNested(flat: number[], shape: number[]): unknown {
  if (!shape.length) return flat[0] ?? 0;
  const [dim, ...rest] = shape;
  const step = rest.length ? rest.reduce((acc, n) => acc * n, 1) : 1;
  const out: unknown[] = [];
  for (let i = 0; i < dim; i++) {
    const slice = flat.slice(i * step, (i + 1) * step);
    out.push(reshapeFlatToNested(slice, rest));
  }
  return out;
}

export function flattenNestedToNumbers(nested: unknown): number[] {
  if (Array.isArray(nested)) return nested.flatMap((v) => flattenNestedToNumbers(v));
  return [Number(nested)];
}
