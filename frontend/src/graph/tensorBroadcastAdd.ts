/** Row-major unravel: flat index → multi-index for `shape`. */
export function unravelRowMajor(flat: number, shape: number[]): number[] {
  const idx: number[] = new Array(shape.length);
  let r = flat;
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i]!;
    idx[i] = r % d;
    r = Math.floor(r / d);
  }
  return idx;
}

/** Row-major ravel: multi-index → flat index for `shape`. */
export function ravelRowMajor(index: number[], shape: number[]): number {
  let s = 0;
  let mul = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s += index[i]! * mul;
    mul *= shape[i]!;
  }
  return s;
}

/**
 * NumPy-style broadcast addition: `valuesA` / `valuesB` are C-contiguous for `shapeA` / `shapeB`.
 */
/**
 * NumPy-style broadcast result shape for two tensors (no values).
 * @throws if shapes are not broadcast-compatible.
 */
export function broadcastShapesOnly(shapeA: number[], shapeB: number[]): number[] {
  const ra = shapeA.length;
  const rb = shapeB.length;
  const R = Math.max(ra, rb);
  const pa = [...Array(R - ra).fill(1), ...shapeA];
  const pb = [...Array(R - rb).fill(1), ...shapeB];
  for (let d = 0; d < R; d++) {
    const a = pa[d]!;
    const b = pb[d]!;
    if (a !== b && a !== 1 && b !== 1) {
      throw new Error(
        `Shapes [${shapeA.join(", ")}] and [${shapeB.join(", ")}] are not broadcast-compatible.`,
      );
    }
  }
  return pa.map((a, d) => Math.max(a, pb[d]!));
}

export function broadcastAddTensorPair(
  shapeA: number[],
  valuesA: number[],
  shapeB: number[],
  valuesB: number[],
): { shape: number[]; values: number[] } {
  const ra = shapeA.length;
  const rb = shapeB.length;
  const R = Math.max(ra, rb);
  const pa = [...Array(R - ra).fill(1), ...shapeA];
  const pb = [...Array(R - rb).fill(1), ...shapeB];
  for (let d = 0; d < R; d++) {
    const a = pa[d]!;
    const b = pb[d]!;
    if (a !== b && a !== 1 && b !== 1) {
      throw new Error(
        `Shapes [${shapeA.join(", ")}] and [${shapeB.join(", ")}] are not broadcast-compatible.`,
      );
    }
  }
  const outShape = pa.map((a, d) => Math.max(a, pb[d]!));
  const count = outShape.reduce((p, n) => p * n, 1);
  const valuesOut: number[] = new Array(count);
  const iaFull = new Array<number>(R);
  const ibFull = new Array<number>(R);
  const iaPhy = new Array<number>(ra);
  const ibPhy = new Array<number>(rb);
  for (let flat = 0; flat < count; flat++) {
    const I = unravelRowMajor(flat, outShape);
    for (let d = 0; d < R; d++) {
      iaFull[d] = pa[d] === 1 ? 0 : I[d]!;
      ibFull[d] = pb[d] === 1 ? 0 : I[d]!;
    }
    for (let j = 0; j < ra; j++) iaPhy[j] = iaFull[R - ra + j]!;
    for (let j = 0; j < rb; j++) ibPhy[j] = ibFull[R - rb + j]!;
    const fa = ravelRowMajor(iaPhy, shapeA);
    const fb = ravelRowMajor(ibPhy, shapeB);
    valuesOut[flat] = valuesA[fa]! + valuesB[fb]!;
  }
  return { shape: outShape, values: valuesOut };
}
