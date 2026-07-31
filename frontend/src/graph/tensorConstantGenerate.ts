import type { TensorConstantInit } from "../components/nodes/tensorConstantDefaults";
import { intChoices, type ListOr1 } from "../components/nodes/multiValueUtils";

const MAX_ELEMENTS = 500_000;

/**
 * Parse a human shape string such as `[2, 3]`, `[2,3]`, or `2, 3` into positive integer sizes.
 */
export function parseTensorShapeInput(raw: string): number[] {
  let t = raw.trim();
  if (t.startsWith("[") && t.endsWith("]")) {
    t = t.slice(1, -1).trim();
  }
  if (!t) {
    throw new Error("Shape is empty. Example: [2, 3] or 2, 3.");
  }
  const parts = t
    .split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (!parts.length) {
    throw new Error("Shape is empty. Example: [2, 3] or 2, 3.");
  }
  const out: number[] = [];
  for (const p of parts) {
    const n = Number(p);
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 1) {
      throw new Error(`Invalid shape entry "${p}": use positive integers (e.g. [2, 3]).`);
    }
    out.push(n);
  }
  return out;
}

function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Mulberry32 PRNG; deterministic for a given 32-bit seed. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) | 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianPair(rand: () => number): [number, number] {
  const u = Math.max(Number.EPSILON, rand());
  const v = rand();
  const mag = Math.sqrt(-2 * Math.log(u));
  return [mag * Math.cos(2 * Math.PI * v), mag * Math.sin(2 * Math.PI * v)];
}

function primaryInitSeed(initSeed: ListOr1<number> | undefined): number {
  const s = intChoices(initSeed, 0);
  return s[0] ?? 0;
}

/**
 * Build a dense row-major tensor from shape and init policy.
 * @throws If shape invalid or product exceeds {@link MAX_ELEMENTS}.
 */
export function generateTensorConstantValues(
  shape: number[],
  init: TensorConstantInit,
  initSeed: ListOr1<number> | undefined,
): number[] {
  if (!shape.length) throw new Error("Shape must have at least one dimension.");
  for (const s of shape) {
    if (!Number.isInteger(s) || s < 1) throw new Error("Each shape entry must be a positive integer.");
  }
  const n = product(shape);
  if (n > MAX_ELEMENTS) {
    throw new Error(`Tensor has ${n} elements (max ${MAX_ELEMENTS}). Reduce shape.`);
  }
  const out = new Array<number>(n);
  if (init === "zero") {
    out.fill(0);
    return out;
  }
  const rand = mulberry32(primaryInitSeed(initSeed));
  if (init === "uniform_m11") {
    for (let i = 0; i < n; i++) out[i] = rand() * 2 - 1;
    return out;
  }
  /* gaussian — standard normal N(0,1), independent draws */
  let i = 0;
  while (i < n) {
    const [a, b] = gaussianPair(rand);
    out[i++] = a;
    if (i < n) out[i++] = b;
  }
  return out;
}
