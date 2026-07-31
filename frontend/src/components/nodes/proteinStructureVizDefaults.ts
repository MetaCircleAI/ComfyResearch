export type ProteinStructureDisplayerNodeData = {
  instanceTitle?: string;
  coordsFlat?: string;
  resolvedCoordsFlat?: string;
  showPolyline?: boolean;
  sampleIndex?: number;
};

export type ProteinStructureComparisonVizNodeData = {
  instanceTitle?: string;
  predCoordsFlat?: string;
  trueCoordsFlat?: string;
  sampleIndex?: number;
};

export function defaultProteinStructureDisplayerData(
  partial?: Partial<ProteinStructureDisplayerNodeData>,
): ProteinStructureDisplayerNodeData {
  return {
    coordsFlat: "",
    resolvedCoordsFlat: "",
    showPolyline: true,
    sampleIndex: 0,
    ...partial,
  };
}

export function defaultProteinStructureComparisonVizData(
  partial?: Partial<ProteinStructureComparisonVizNodeData>,
): ProteinStructureComparisonVizNodeData {
  return {
    predCoordsFlat: "",
    trueCoordsFlat: "",
    sampleIndex: 0,
    ...partial,
  };
}

export function parseCoordsFlat(raw: string | undefined | null): number[][] {
  const t = String(raw ?? "").trim();
  if (!t) return [];
  const rows = t
    .split(/\s*;\s*|\n+/)
    .map((r) => r.trim())
    .filter(Boolean);
  const out: number[][] = [];
  for (const row of rows) {
    const vals = row
      .split(/\s*,\s*|\s+/)
      .map((v) => Number.parseFloat(v))
      .filter((v) => Number.isFinite(v));
    if (vals.length >= 2) {
      out.push([vals[0]!, vals[1]!, vals[2] ?? 0]);
    }
  }
  return out;
}

export function coordsFromTensorLike(data: Record<string, unknown> | undefined): number[][] {
  if (!data) return [];
  const flat = data.coordsFlat;
  if (typeof flat === "string") return parseCoordsFlat(flat);
  const values = data.values;
  if (Array.isArray(values)) {
    const nums = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (nums.length >= 6) {
      const out: number[][] = [];
      for (let i = 0; i + 2 < nums.length; i += 3) out.push([nums[i]!, nums[i + 1]!, nums[i + 2]!]);
      return out;
    }
  }
  return [];
}

function clampSampleIndex(sampleIndex: number, sampleCount: number): number {
  if (!Number.isFinite(sampleIndex)) return 0;
  const i = Math.floor(sampleIndex);
  if (sampleCount <= 1) return 0;
  if (i < 0) return 0;
  if (i >= sampleCount) return sampleCount - 1;
  return i;
}

function rowsToCoords(values: number[], rowCount: number, rowWidth: number): number[][] {
  const out: number[][] = [];
  if (rowWidth < 2) return out;
  for (let r = 0; r < rowCount; r += 1) {
    const off = r * rowWidth;
    const x = values[off];
    const y = values[off + 1];
    const z = rowWidth >= 3 ? (values[off + 2] ?? 0) : 0;
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      out.push([x!, y!, z!]);
    }
  }
  return out;
}

export type ExtractedStructureCoords = {
  coords: number[][];
  sampleIndexUsed: number;
  sampleCount: number;
};

/**
 * Extract one structure sample from common tensor layouts:
 * - [N, 3], [N, 2], [B, N, 3], [B, N, 2]
 * - flattened [3N], [2N], [B, 3N], [B, 2N]
 */
export function extractStructureCoordsFromTensor(
  shape: number[],
  values: number[],
  sampleIndex: number,
): ExtractedStructureCoords | null {
  const dims = shape.map((x) => Math.floor(Number(x))).filter((x) => Number.isFinite(x) && x > 0);
  if (dims.length === 0 || values.length < 2) return null;

  if (dims.length === 1) {
    const l = dims[0]!;
    if (l % 3 === 0) {
      return { coords: rowsToCoords(values.slice(0, l), l / 3, 3), sampleIndexUsed: 0, sampleCount: 1 };
    }
    if (l % 2 === 0) {
      return { coords: rowsToCoords(values.slice(0, l), l / 2, 2), sampleIndexUsed: 0, sampleCount: 1 };
    }
    return null;
  }

  if (dims.length === 2) {
    const [a, b] = dims;
    if (b === 3 || b === 2) {
      const need = a * b;
      if (values.length < need) return null;
      return { coords: rowsToCoords(values.slice(0, need), a, b), sampleIndexUsed: 0, sampleCount: 1 };
    }
    if (a === 3 || a === 2) {
      const need = a * b;
      if (values.length < need) return null;
      const rowMajor = values.slice(0, need);
      const out: number[][] = [];
      for (let j = 0; j < b; j += 1) {
        const x = rowMajor[j];
        const y = rowMajor[b + j];
        const z = a >= 3 ? (rowMajor[2 * b + j] ?? 0) : 0;
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          out.push([x!, y!, z!]);
        }
      }
      return { coords: out, sampleIndexUsed: 0, sampleCount: 1 };
    }
    const width = b;
    if (width % 3 === 0) {
      const s = clampSampleIndex(sampleIndex, a);
      const sampleLen = width;
      const off = s * sampleLen;
      if (values.length < off + sampleLen) return null;
      return {
        coords: rowsToCoords(values.slice(off, off + sampleLen), width / 3, 3),
        sampleIndexUsed: s,
        sampleCount: a,
      };
    }
    if (width % 2 === 0) {
      const s = clampSampleIndex(sampleIndex, a);
      const sampleLen = width;
      const off = s * sampleLen;
      if (values.length < off + sampleLen) return null;
      return {
        coords: rowsToCoords(values.slice(off, off + sampleLen), width / 2, 2),
        sampleIndexUsed: s,
        sampleCount: a,
      };
    }
    return null;
  }

  if (dims.length >= 3) {
    const [b, n, c] = dims;
    if (c === 3 || c === 2) {
      const sampleCount = b;
      const s = clampSampleIndex(sampleIndex, sampleCount);
      const sampleLen = n * c;
      const off = s * sampleLen;
      if (values.length < off + sampleLen) return null;
      return {
        coords: rowsToCoords(values.slice(off, off + sampleLen), n, c),
        sampleIndexUsed: s,
        sampleCount,
      };
    }
    if (n === 3 || n === 2) {
      const sampleCount = b;
      const s = clampSampleIndex(sampleIndex, sampleCount);
      const sampleLen = n * c;
      const off = s * sampleLen;
      if (values.length < off + sampleLen) return null;
      const rowMajor = values.slice(off, off + sampleLen);
      const out: number[][] = [];
      for (let j = 0; j < c; j += 1) {
        const x = rowMajor[j];
        const y = rowMajor[c + j];
        const z = n >= 3 ? (rowMajor[2 * c + j] ?? 0) : 0;
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          out.push([x!, y!, z!]);
        }
      }
      return { coords: out, sampleIndexUsed: s, sampleCount };
    }
  }

  return null;
}

export function projectCoords2d(coords: number[][], w: number, h: number): { x: number; y: number }[] {
  if (!coords.length) return [];
  const xs = coords.map((c) => c[0]!);
  const ys = coords.map((c) => c[1]!);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const pad = 8;
  const innerW = Math.max(1, w - 2 * pad);
  const innerH = Math.max(1, h - 2 * pad);
  return coords.map((c) => ({
    x: pad + ((c[0]! - minX) / spanX) * innerW,
    y: h - pad - ((c[1]! - minY) / spanY) * innerH,
  }));
}

export function centeredRmsd(pred: number[][], target: number[][]): number | null {
  const n = Math.min(pred.length, target.length);
  if (n < 2) return null;
  let px = 0;
  let py = 0;
  let pz = 0;
  let tx = 0;
  let ty = 0;
  let tz = 0;
  for (let i = 0; i < n; i += 1) {
    px += pred[i]![0]!;
    py += pred[i]![1]!;
    pz += pred[i]![2]!;
    tx += target[i]![0]!;
    ty += target[i]![1]!;
    tz += target[i]![2]!;
  }
  px /= n;
  py /= n;
  pz /= n;
  tx /= n;
  ty /= n;
  tz /= n;
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = (pred[i]![0]! - px) - (target[i]![0]! - tx);
    const dy = (pred[i]![1]! - py) - (target[i]![1]! - ty);
    const dz = (pred[i]![2]! - pz) - (target[i]![2]! - tz);
    acc += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(acc / n);
}

