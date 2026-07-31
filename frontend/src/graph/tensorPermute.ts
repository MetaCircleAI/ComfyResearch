/** Row-major tensor dimension permutation (numpy.transpose semantics). */

function stridesRowMajor(shape: number[]): number[] {
  const r = shape.length;
  const s = new Array<number>(r);
  let acc = 1;
  for (let i = r - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= Math.max(1, shape[i]!);
  }
  return s;
}

function ravelIndex(coord: number[], strides: number[]): number {
  let idx = 0;
  for (let i = 0; i < coord.length; i++) idx += coord[i]! * strides[i]!;
  return idx;
}

function unravelIndex(linear: number, shape: number[], strides: number[]): number[] {
  const coord = new Array<number>(shape.length);
  let rest = linear;
  for (let i = 0; i < shape.length; i++) {
    const si = strides[i]!;
    coord[i] = Math.floor(rest / si);
    rest %= si;
  }
  return coord;
}

/** `axes[outDim]` = input dimension index feeding output dimension `outDim`. */
export function normalizePermutation(axes: number[] | undefined, rank: number): number[] {
  const id = Array.from({ length: rank }, (_, i) => i);
  if (!axes || axes.length !== rank) return id;
  const seen = new Set<number>();
  for (const a of axes) {
    if (!Number.isInteger(a) || a < 0 || a >= rank) return id;
    if (seen.has(a)) return id;
    seen.add(a);
  }
  if (seen.size !== rank) return id;
  return [...axes];
}

export function permuteRowMajor(
  shape: number[],
  values: number[],
  axes: number[],
): { shape: number[]; values: number[] } {
  const r = shape.length;
  const ax = normalizePermutation(axes, r);
  if (r === 0) {
    return { shape: [], values: [...values] };
  }
  const newShape = ax.map((oldIdx) => shape[oldIdx]!);
  const inv = new Array<number>(r);
  for (let k = 0; k < r; k++) inv[ax[k]!] = k;
  const inStrides = stridesRowMajor(shape);
  const outStrides = stridesRowMajor(newShape);
  const out = new Array<number>(values.length);
  for (let ol = 0; ol < values.length; ol++) {
    const om = unravelIndex(ol, newShape, outStrides);
    const im = new Array<number>(r);
    for (let j = 0; j < r; j++) im[j] = om[inv[j]!]!;
    const il = ravelIndex(im, inStrides);
    out[ol] = values[il]!;
  }
  return { shape: newShape, values: out };
}
