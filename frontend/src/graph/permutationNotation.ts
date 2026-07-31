import { normalizePermutation } from "./tensorPermute";

/** Labels for input axes 0,1,2,… in order (i, j, k, …). */
const AXIS_LABELS = "ijklmnopqrstuvwxyz";

export function defaultAxisLabels(rank: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < rank; i++) {
    if (i < AXIS_LABELS.length) out.push(AXIS_LABELS[i]!);
    else out.push(`d${i}`);
  }
  return out;
}

export function tokenizeEinsteinSide(s: string): string[] {
  return s
    .trim()
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter(Boolean);
}

/** Renders current permutation as `i j k -> k j i` (numpy `axes` / transpose order). */
export function formatPermutationAsEinstein(axes: number[] | undefined, rank: number): string {
  if (rank <= 0) return "";
  const labels = defaultAxisLabels(rank);
  const ax = normalizePermutation(axes, rank);
  const left = labels.join(" ");
  const right = ax.map((inDim) => labels[inDim]!).join(" ");
  return `${left} -> ${right}`;
}

export type ParseEinsteinResult =
  | { ok: true; axes: number[] }
  | { ok: false; message: string };

/**
 * Parse `left -> right`: left lists labels for input dims 0..rank-1; right lists which input label
 * feeds each output dim (same multiset as left).
 */
export function parseEinsteinPermutation(spec: string, rank: number): ParseEinsteinResult {
  if (rank <= 0) return { ok: false, message: "No dimensions to permute." };
  const trimmed = spec.trim();
  if (!trimmed) return { ok: false, message: "Enter a spec like “i j k -> k j i”." };
  const parts = trimmed.split(/\s*->\s*/i);
  if (parts.length !== 2) {
    return { ok: false, message: "Use “->” between input and output labels (e.g. i j k -> k j i)." };
  }
  const left = tokenizeEinsteinSide(parts[0]!);
  const right = tokenizeEinsteinSide(parts[1]!);
  if (left.length !== rank) {
    return { ok: false, message: `Input side must have ${rank} label(s), got ${left.length}.` };
  }
  if (right.length !== rank) {
    return { ok: false, message: `Output side must have ${rank} label(s), got ${right.length}.` };
  }
  const labelToIn = new Map<string, number>();
  for (let i = 0; i < left.length; i++) {
    const t = left[i]!;
    if (labelToIn.has(t)) return { ok: false, message: `Duplicate label on input side: ${t}` };
    labelToIn.set(t, i);
  }
  const axes = new Array<number>(rank);
  const seenIn = new Set<number>();
  for (let j = 0; j < right.length; j++) {
    const t = right[j]!;
    const inDim = labelToIn.get(t);
    if (inDim === undefined) return { ok: false, message: `Unknown label on output side: ${t}` };
    if (seenIn.has(inDim)) return { ok: false, message: "Output must use each input label exactly once." };
    seenIn.add(inDim);
    axes[j] = inDim;
  }
  if (seenIn.size !== rank) {
    return { ok: false, message: "Output must use each input label exactly once." };
  }
  return { ok: true, axes: normalizePermutation(axes, rank) };
}
