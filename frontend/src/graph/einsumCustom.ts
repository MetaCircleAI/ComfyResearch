/**
 * Einstein / NumPy-style einsum subscripts for Statistics (custom reduction) and Statistics 2 (dot / cosine).
 * Letters a–z A–Z label axes; one letter per axis on each operand’s left-hand side.
 */

import type { StatisticsReductionOp } from "../components/nodes/statisticsDefaults";
import type { Statistics2PairOp } from "./statistics2Pair";
import { applyReductionOpTo1D } from "./statisticsReduce";

const LABEL_RE = /^[a-zA-Z]+$/;

function stridesFromShape(shape: number[]): number[] {
  const rank = shape.length;
  const st = new Array(rank).fill(1);
  for (let d = rank - 2; d >= 0; d--) {
    st[d] = st[d + 1]! * shape[d + 1]!;
  }
  return st;
}

function linearFromCoords(coords: number[], strides: number[]): number {
  let s = 0;
  for (let d = 0; d < coords.length; d++) {
    s += coords[d]! * strides[d]!;
  }
  return s;
}

function normalizeExpr(raw: string): string {
  return raw.replace(/\s+/g, "");
}

function assertLabelString(s: string, ctx: string): void {
  if (!LABEL_RE.test(s)) {
    throw new Error(`${ctx}: use only letters a–z or A–Z, one per axis (got ${JSON.stringify(s)}).`);
  }
  if (new Set(s).size !== s.length) {
    throw new Error(`${ctx}: duplicate axis labels are not supported on one tensor.`);
  }
}

export function parseSingleOperandEinstein(exprRaw: string): { lhs: string; rhs: string } {
  const expr = normalizeExpr(exprRaw);
  if (!expr.includes("->")) {
    throw new Error('Statistics notation must include "->" (e.g. ij -> j or abc->bc).');
  }
  const [lhs, rhs] = expr.split("->", 2);
  if (!lhs) {
    throw new Error("Missing left-hand side before ->.");
  }
  assertLabelString(lhs, "Left-hand side");
  const rhsNorm = rhs ?? "";
  if (rhsNorm.length > 0) {
    assertLabelString(rhsNorm, "Right-hand side");
    for (const c of rhsNorm) {
      if (!lhs.includes(c)) {
        throw new Error(`Output label ${JSON.stringify(c)} does not appear on the left.`);
      }
    }
  }
  return { lhs, rhs: rhsNorm };
}

export function parseBinaryOperandEinstein(exprRaw: string): {
  lhs0: string;
  lhs1: string;
  rhs: string;
} {
  const expr = normalizeExpr(exprRaw);
  if (!expr.includes("->")) {
    throw new Error('Statistics 2 notation must include "->" (e.g. ij,ik->jk).');
  }
  const [left, rhsPart] = expr.split("->", 2);
  const parts = left.split(",");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    throw new Error('Use two operands separated by a comma on the left, e.g. "ij,ik->jk".');
  }
  const lhs0 = parts[0]!;
  const lhs1 = parts[1]!;
  assertLabelString(lhs0, "First tensor subscripts");
  assertLabelString(lhs1, "Second tensor subscripts");
  const rhs = rhsPart ?? "";
  if (rhs.length > 0) {
    assertLabelString(rhs, "Output subscripts");
  }
  const allowed = new Set<string>();
  for (const c of lhs0) allowed.add(c);
  for (const c of lhs1) allowed.add(c);
  for (const c of rhs) {
    if (!allowed.has(c)) {
      throw new Error(`Output label ${JSON.stringify(c)} does not appear in the inputs.`);
    }
  }
  return { lhs0, lhs1, rhs };
}

/** Sorted unique letters from lhs0 + lhs1 (deterministic full index space). */
function allLetters(lhs0: string, lhs1: string): string[] {
  const u = new Set<string>();
  for (const c of lhs0) u.add(c);
  for (const c of lhs1) u.add(c);
  return [...u].sort();
}

function buildLetterSizes(lhs0: string, sh0: number[], lhs1: string, sh1: number[]): Map<string, number> {
  const sizes = new Map<string, number>();
  for (let i = 0; i < lhs0.length; i++) {
    const c = lhs0[i]!;
    const sz = sh0[i]!;
    const prev = sizes.get(c);
    if (prev !== undefined && prev !== sz) {
      throw new Error(`Label ${JSON.stringify(c)} has conflicting sizes (${prev} vs ${sz}).`);
    }
    sizes.set(c, sz);
  }
  for (let i = 0; i < lhs1.length; i++) {
    const c = lhs1[i]!;
    const sz = sh1[i]!;
    const prev = sizes.get(c);
    if (prev !== undefined && prev !== sz) {
      throw new Error(`Label ${JSON.stringify(c)} has conflicting sizes (${prev} vs ${sz}).`);
    }
    sizes.set(c, sz);
  }
  return sizes;
}

function linearWithLabels(lhs: string, strides: number[], pos: Record<string, number>): number {
  const coords = lhs.split("").map((c) => pos[c]!);
  return linearFromCoords(coords, strides);
}

export function inferBinaryOutputShape(
  shape0: number[],
  shape1: number[],
  exprRaw: string,
): number[] {
  const { lhs0, lhs1, rhs } = parseBinaryOperandEinstein(exprRaw);
  if (lhs0.length !== shape0.length || lhs1.length !== shape1.length) {
    throw new Error(
      `Subscript length must match tensor rank (got ${lhs0.length} vs rank ${shape0.length}, ${lhs1.length} vs ${shape1.length}).`,
    );
  }
  const sizes = buildLetterSizes(lhs0, shape0, lhs1, shape1);
  return rhs.split("").map((c) => {
    const s = sizes.get(c);
    if (s === undefined) throw new Error(`Missing size for ${JSON.stringify(c)}.`);
    return s;
  });
}

export function inferSingleOutputShapeFromShape(shape: number[], exprRaw: string): number[] {
  const { lhs, rhs } = parseSingleOperandEinstein(exprRaw);
  if (lhs.length !== shape.length) {
    throw new Error(`Left-hand side must have ${shape.length} letters (one per axis); got ${lhs.length}.`);
  }
  const lmap: Record<string, number> = {};
  for (let i = 0; i < lhs.length; i++) {
    lmap[lhs[i]!] = i;
  }
  return rhs.split("").map((c) => shape[lmap[c]!]!);
}

/** Shape preview when only rank is known (placeholder size 1 per output axis). */
export function inferSingleOutputShapeFromRank(rank: number, exprRaw: string): number[] | null {
  try {
    const dummy = Array.from({ length: rank }, () => 1);
    return inferSingleOutputShapeFromShape(dummy, exprRaw);
  } catch {
    return null;
  }
}

export function inferBinaryOutputShapeSafe(
  shape0: number[] | null,
  shape1: number[] | null,
  exprRaw: string,
): number[] | null {
  if (!shape0 || !shape1 || shape0.length === 0 || shape1.length === 0) return null;
  try {
    return inferBinaryOutputShape(shape0, shape1, exprRaw);
  } catch {
    return null;
  }
}

export function singleTensorEinsteinReduce(
  shape: number[],
  values: number[],
  exprRaw: string,
  op: StatisticsReductionOp,
): { shape: number[]; values: number[] } {
  const { lhs, rhs } = parseSingleOperandEinstein(exprRaw);
  if (lhs.length !== shape.length) {
    throw new Error(`Left-hand side must have ${shape.length} letters (one per axis); got ${lhs.length}.`);
  }
  const strides = stridesFromShape(shape);
  const lmap: Record<string, number> = {};
  for (let i = 0; i < lhs.length; i++) {
    lmap[lhs[i]!] = i;
  }
  const outLetters = rhs.split("");
  const reducedLetters = [...new Set(lhs.split("").filter((c) => !rhs.includes(c)))].sort();

  const outShape = inferSingleOutputShapeFromShape(shape, exprRaw);
  const outSize = outShape.reduce((a, b) => a * b, 1) || 1;
  const outStrides = stridesFromShape(outShape.length ? outShape : [1]);
  const outValues = new Array<number>(outSize);

  const pos: Record<string, number> = {};
  const outMulti = new Array(outLetters.length).fill(0);

  function outLinearFromMulti(): number {
    if (outLetters.length === 0) return 0;
    let lin = 0;
    for (let k = 0; k < outLetters.length; k++) {
      lin += outMulti[k]! * outStrides[k]!;
    }
    return lin;
  }

  function walkOut(d: number): void {
    if (d === outLetters.length) {
      for (let k = 0; k < outLetters.length; k++) {
        pos[outLetters[k]!] = outMulti[k]!;
      }
      const oLin = outLinearFromMulti();
      if (reducedLetters.length === 0) {
        const coords = lhs.split("").map((c) => pos[c]!);
        const v = values[linearFromCoords(coords, strides)] ?? Number.NaN;
        outValues[oLin] = applyReductionOpTo1D([v], op);
        return;
      }
      const slice: number[] = [];
      function walkRed(rd: number): void {
        if (rd === reducedLetters.length) {
          outValues[oLin] = applyReductionOpTo1D(slice, op);
          return;
        }
        const L = reducedLetters[rd]!;
        const ax = lmap[L]!;
        for (let v = 0; v < shape[ax]!; v++) {
          pos[L] = v;
          const coords = lhs.split("").map((c) => pos[c]!);
          slice.push(values[linearFromCoords(coords, strides)] ?? Number.NaN);
          walkRed(rd + 1);
          slice.pop();
        }
      }
      walkRed(0);
      return;
    }
    const L = outLetters[d]!;
    const ax = lmap[L]!;
    for (let i = 0; i < shape[ax]!; i++) {
      outMulti[d] = i;
      walkOut(d + 1);
    }
  }

  walkOut(0);

  return { shape: outShape, values: outValues };
}

export function binaryTensorEinsteinPair(
  shape0: number[],
  values0: number[],
  shape1: number[],
  values1: number[],
  exprRaw: string,
  pairOp: Statistics2PairOp,
): { shape: number[]; values: number[] } {
  const { lhs0, lhs1, rhs } = parseBinaryOperandEinstein(exprRaw);
  if (lhs0.length !== shape0.length || lhs1.length !== shape1.length) {
    throw new Error(
      `Subscript length must match tensor rank (got ${lhs0.length} vs ${shape0.length}, ${lhs1.length} vs ${shape1.length}).`,
    );
  }
  const sizes = buildLetterSizes(lhs0, shape0, lhs1, shape1);
  const letters = allLetters(lhs0, lhs1);
  const st0 = stridesFromShape(shape0);
  const st1 = stridesFromShape(shape1);
  const outShape = rhs.split("").map((c) => sizes.get(c)!);
  const outSize = outShape.reduce((a, b) => a * b, 1) || 1;
  const outStrides = stridesFromShape(outShape.length ? outShape : [1]);

  const pos: Record<string, number> = {};
  const dotAcc = new Map<number, number>();
  const naAcc = new Map<number, number>();
  const nbAcc = new Map<number, number>();

  function outLinear(): number {
    if (rhs.length === 0) return 0;
    let lin = 0;
    for (let k = 0; k < rhs.length; k++) {
      const c = rhs[k]!;
      lin += pos[c]! * outStrides[k]!;
    }
    return lin;
  }

  function recurseLetter(li: number): void {
    if (li === letters.length) {
      const ia = linearWithLabels(lhs0, st0, pos);
      const ib = linearWithLabels(lhs1, st1, pos);
      const va = values0[ia] ?? Number.NaN;
      const vb = values1[ib] ?? Number.NaN;
      const ol = outLinear();
      if (pairOp === "dot") {
        dotAcc.set(ol, (dotAcc.get(ol) ?? 0) + va * vb);
      } else {
        dotAcc.set(ol, (dotAcc.get(ol) ?? 0) + va * vb);
        naAcc.set(ol, (naAcc.get(ol) ?? 0) + va * va);
        nbAcc.set(ol, (nbAcc.get(ol) ?? 0) + vb * vb);
      }
      return;
    }
    const L = letters[li]!;
    const n = sizes.get(L)!;
    for (let v = 0; v < n; v++) {
      pos[L] = v;
      recurseLetter(li + 1);
    }
  }

  recurseLetter(0);

  const outValues = new Array<number>(outSize);
  const eps = 1e-12;
  for (let i = 0; i < outSize; i++) {
    if (pairOp === "dot") {
      outValues[i] = dotAcc.get(i) ?? 0;
    } else {
      const dot = dotAcc.get(i) ?? 0;
      const na = naAcc.get(i) ?? 0;
      const nb = nbAcc.get(i) ?? 0;
      outValues[i] = dot / (Math.sqrt(na) * Math.sqrt(nb) + eps);
    }
  }

  return { shape: outShape.length ? outShape : [], values: outValues };
}
