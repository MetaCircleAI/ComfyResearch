import type { StatisticsReductionOp } from "../components/nodes/statisticsDefaults";

function stridesFromShape(shape: number[]): number[] {
  const rank = shape.length;
  const st = new Array(rank).fill(1);
  for (let d = rank - 2; d >= 0; d--) {
    st[d] = st[d + 1] * shape[d + 1];
  }
  return st;
}

function linearIndex(fullCoords: number[], strides: number[]): number {
  let s = 0;
  for (let d = 0; d < fullCoords.length; d++) {
    s += fullCoords[d] * strides[d];
  }
  return s;
}

export function applyReductionOpTo1D(slice: number[], op: StatisticsReductionOp): number {
  const xs = slice.map(Number).filter((v) => Number.isFinite(v));
  if (xs.length === 0) return Number.NaN;
  switch (op) {
    case "mean":
      return xs.reduce((a, b) => a + b, 0) / xs.length;
    case "median": {
      const s = [...xs].sort((a, b) => a - b);
      const m = Math.floor(s.length / 2);
      return s.length % 2 === 1 ? s[m]! : ((s[m - 1]! + s[m]!) / 2);
    }
    case "max":
      return Math.max(...xs);
    case "min":
      return Math.min(...xs);
    case "l2_norm": {
      let t = 0;
      for (const x of xs) t += x * x;
      return Math.sqrt(t);
    }
    case "l1_norm": {
      let t = 0;
      for (const x of xs) t += Math.abs(x);
      return t;
    }
    default: {
      const _u: never = op;
      throw new Error(`Unknown reduction op: ${_u}`);
    }
  }
}

export function outputShapeAfterReduce(shape: number[], axis: number): number[] {
  return shape.filter((_, i) => i !== axis);
}

/** Output shape after removing every listed axis (each axis index refers to the **original** shape). */
export function outputShapeAfterReduceAxes(shape: number[], axes: number[]): number[] {
  const drop = new Set(axes.map((a) => Math.floor(Number(a))).filter((a) => a >= 0 && a < shape.length));
  return shape.filter((_, i) => !drop.has(i));
}

export function reduceTensorAlongAxis(
  shape: number[],
  values: number[],
  axis: number,
  op: StatisticsReductionOp,
): { shape: number[]; values: number[] } {
  const rank = shape.length;
  if (axis < 0 || axis >= rank) {
    throw new Error(`Axis ${axis} out of range for rank ${rank}.`);
  }
  const dimAlong = shape[axis]!;
  const outShape = outputShapeAfterReduce(shape, axis);
  const strides = stridesFromShape(shape);
  const outSize = outShape.reduce((a, b) => a * b, 1) || 1;
  const outStrides = stridesFromShape(outShape);
  const outValues = new Array<number>(outSize);

  const outMulti = new Array(outShape.length).fill(0);

  function walkOut(d: number): void {
    if (d === outShape.length) {
      const slice: number[] = [];
      for (let j = 0; j < dimAlong; j++) {
        const full = new Array(rank).fill(0);
        let oi = 0;
        for (let di = 0; di < rank; di++) {
          if (di === axis) full[di] = j;
          else full[di] = outMulti[oi++]!;
        }
        slice.push(values[linearIndex(full, strides)] ?? Number.NaN);
      }
      let outLin = 0;
      for (let k = 0; k < outMulti.length; k++) {
        outLin += outMulti[k]! * outStrides[k]!;
      }
      if (outShape.length === 0) outLin = 0;
      outValues[outLin] = applyReductionOpTo1D(slice, op);
      return;
    }
    for (let i = 0; i < outShape[d]!; i++) {
      outMulti[d] = i;
      walkOut(d + 1);
    }
  }

  walkOut(0);

  return { shape: outShape, values: outValues };
}

/**
 * Reduce along multiple axes (same op each time). Axes are applied in **descending** order so indices stay valid.
 */
export function reduceTensorAlongAxes(
  shape: number[],
  values: number[],
  axesIn: number[],
  op: StatisticsReductionOp,
): { shape: number[]; values: number[] } {
  const rank = shape.length;
  const uniq = [...new Set(axesIn.map((a) => Math.floor(Number(a))))].filter((a) => Number.isFinite(a) && a >= 0 && a < rank);
  if (uniq.length === 0) {
    throw new Error(`Select at least one dimension in [0, ${rank - 1}].`);
  }
  const axesDesc = [...uniq].sort((a, b) => b - a);
  let sh = [...shape];
  let vals = values;
  for (const ax of axesDesc) {
    const r = reduceTensorAlongAxis(sh, vals, ax, op);
    sh = r.shape;
    vals = r.values;
  }
  return { shape: sh, values: vals };
}
