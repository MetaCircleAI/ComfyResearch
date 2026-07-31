import type { TensorSliceSpec } from "../components/nodes/tensorSlicingDefaults";

function stridesRowMajor(shape: number[]): number[] {
  const out = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = acc;
    acc *= Math.max(1, shape[i] ?? 1);
  }
  return out;
}

function unravelIndex(linear: number, shape: number[], strides: number[]): number[] {
  const coord = new Array<number>(shape.length);
  let rest = linear;
  for (let i = 0; i < shape.length; i++) {
    const s = strides[i] ?? 1;
    coord[i] = Math.floor(rest / s);
    rest %= s;
  }
  return coord;
}

function ravelIndex(coord: number[], strides: number[]): number {
  let out = 0;
  for (let i = 0; i < coord.length; i++) out += (coord[i] ?? 0) * (strides[i] ?? 1);
  return out;
}

export function parseSliceIndices(raw: string): number[] | null {
  const parts = raw
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  if (parts.length === 0) return null;
  const out = parts.map((p) => Number(p));
  if (out.some((x) => !Number.isInteger(x))) return null;
  return out;
}

/** Non-negative indices as-is; negative count from the end (-1 = last), like Python. */
export function resolveTensorSliceIndex(i: number, axisLen: number): number | null {
  if (!Number.isInteger(axisLen) || axisLen < 1) return null;
  if (!Number.isInteger(i)) return null;
  if (i >= 0) return i < axisLen ? i : null;
  const j = axisLen + i;
  return j >= 0 ? j : null;
}

export function resolveTensorSliceIndices(raw: number[], axisLen: number): number[] | null {
  const out: number[] = [];
  for (const i of raw) {
    const r = resolveTensorSliceIndex(i, axisLen);
    if (r == null) return null;
    out.push(r);
  }
  return out;
}

export function normalizeSlices(raw: unknown): TensorSliceSpec[] {
  if (!Array.isArray(raw)) return [];
  const out: TensorSliceSpec[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const d = Number((entry as { dimension?: unknown }).dimension);
    const indices = String((entry as { indices?: unknown }).indices ?? "").trim();
    if (!Number.isInteger(d)) continue;
    out.push({ dimension: d, indices });
  }
  return out;
}

function applySliceShapeOnce(shape: number[], spec: TensorSliceSpec): number[] | null {
  const rank = shape.length;
  if (!Number.isInteger(spec.dimension) || spec.dimension < 0 || spec.dimension >= rank) return null;
  // Empty indices means "all indices" for that dimension, i.e. ignore this row.
  if (!spec.indices.trim()) return [...shape];
  const axisLen = shape[spec.dimension] ?? 0;
  if (!Number.isInteger(axisLen) || axisLen < 1) return null;
  const idx = parseSliceIndices(spec.indices);
  const idxR = idx ? resolveTensorSliceIndices(idx, axisLen) : null;
  if (!idxR) return null;
  if (idxR.length === 1) {
    return [...shape.slice(0, spec.dimension), ...shape.slice(spec.dimension + 1)];
  }
  const next = [...shape];
  next[spec.dimension] = idxR.length;
  return next;
}

export function inferTensorSliceShape(shape: number[], slices: TensorSliceSpec[]): number[] | null {
  let cur = [...shape];
  for (const s of slices) {
    const next = applySliceShapeOnce(cur, s);
    if (!next) return null;
    cur = next;
  }
  return cur;
}

function applySliceValuesOnce(shape: number[], values: number[], spec: TensorSliceSpec): { shape: number[]; values: number[] } | null {
  const rank = shape.length;
  if (!Number.isInteger(spec.dimension) || spec.dimension < 0 || spec.dimension >= rank) return null;
  // Empty indices means "all indices" for that dimension, i.e. ignore this row.
  if (!spec.indices.trim()) return { shape: [...shape], values: [...values] };
  const axisLen = shape[spec.dimension] ?? 0;
  if (!Number.isInteger(axisLen) || axisLen < 1) return null;
  const idx = parseSliceIndices(spec.indices);
  const idxR = idx ? resolveTensorSliceIndices(idx, axisLen) : null;
  if (!idxR) return null;

  const inStrides = stridesRowMajor(shape);
  if (idxR.length === 1) {
    const nextShape = [...shape.slice(0, spec.dimension), ...shape.slice(spec.dimension + 1)];
    const outLen = Math.max(1, nextShape.reduce((a, b) => a * Math.max(1, b), 1));
    const out = new Array<number>(outLen);
    const outStrides = stridesRowMajor(nextShape);
    for (let ol = 0; ol < outLen; ol++) {
      const ocoord = nextShape.length > 0 ? unravelIndex(ol, nextShape, outStrides) : [];
      const icoord = new Array<number>(rank);
      let oi = 0;
      for (let i = 0; i < rank; i++) {
        if (i === spec.dimension) icoord[i] = idxR[0]!;
        else icoord[i] = ocoord[oi++] ?? 0;
      }
      out[ol] = values[ravelIndex(icoord, inStrides)] ?? 0;
    }
    return { shape: nextShape, values: out };
  }

  const nextShape = [...shape];
  nextShape[spec.dimension] = idxR.length;
  const outLen = Math.max(1, nextShape.reduce((a, b) => a * Math.max(1, b), 1));
  const out = new Array<number>(outLen);
  const outStrides = stridesRowMajor(nextShape);
  for (let ol = 0; ol < outLen; ol++) {
    const ocoord = nextShape.length > 0 ? unravelIndex(ol, nextShape, outStrides) : [];
    const icoord = [...ocoord];
    const picked = idxR[ocoord[spec.dimension] ?? 0];
    if (picked == null) return null;
    icoord[spec.dimension] = picked;
    out[ol] = values[ravelIndex(icoord, inStrides)] ?? 0;
  }
  return { shape: nextShape, values: out };
}

export function applyTensorSlicing(
  shape: number[],
  values: number[],
  slices: TensorSliceSpec[],
): { shape: number[]; values: number[] } | null {
  let curShape = [...shape];
  let curValues = [...values];
  for (const s of slices) {
    const next = applySliceValuesOnce(curShape, curValues, s);
    if (!next) return null;
    curShape = next.shape;
    curValues = next.values;
  }
  return { shape: curShape, values: curValues };
}
