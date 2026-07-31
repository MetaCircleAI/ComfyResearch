import type {
  LazyTensorOp,
  Resolved,
  ResolvedLazyActivation,
  ResolvedLazyDataset,
  ResolvedNone,
  ResolvedOk,
} from "./resolveUpstreamTensor";
import { applyTensorSlicing, normalizeSlices } from "./tensorSlice";
import { normalizePermutation, permuteRowMajor } from "./tensorPermute";

function tensorElementCount(shape: number[]): number {
  return shape.reduce((acc, v) => acc * Math.max(0, Math.floor(v)), 1);
}

function flattenOutputShapeRowMajor(shape: number[], exceptDim: number | null): number[] | null {
  const rank = shape.length;
  if (rank < 1) return null;
  if (exceptDim == null) {
    let p = 1;
    for (const s of shape) p *= Math.max(0, Math.floor(s));
    return [p];
  }
  let ax = exceptDim;
  if (ax < 0) ax = rank + ax;
  if (!Number.isInteger(ax) || ax < 0 || ax >= rank) return null;
  let prod = 1;
  for (let i = 0; i < rank; i += 1) {
    if (i !== ax) prod *= Math.max(0, Math.floor(shape[i]!));
  }
  return [Math.max(0, Math.floor(shape[ax]!)), prod];
}

function applyFlattenRowMajorOk(
  shape: number[],
  values: number[],
  exceptDim: number | null,
): { shape: number[]; values: number[] } | null {
  const outShape = flattenOutputShapeRowMajor(shape, exceptDim);
  if (!outShape) return null;
  const exp = tensorElementCount(shape);
  if (exp !== values.length) return null;
  if (exceptDim == null) {
    return { shape: outShape, values: [...values] };
  }
  const rank = shape.length;
  let ax = exceptDim;
  if (ax < 0) ax = rank + ax;
  if (ax !== 0 && ax !== rank - 1) return null;
  return { shape: outShape, values: [...values] };
}

function applyLazyOps(base: ResolvedOk, ops: LazyTensorOp[] | undefined): ResolvedOk | ResolvedNone {
  if (!ops || ops.length === 0) return base;
  let shape = [...base.shape];
  let values = [...base.values];
  let sourceSummary = base.sourceSummary;
  for (const op of ops) {
    if (op.kind === "slice") {
      const out = applyTensorSlicing(shape, values, normalizeSlices(op.slices ?? []));
      if (!out) return { kind: "none", detail: "Invalid slicing config: check dimensions and index ranges." };
      shape = out.shape;
      values = out.values;
      sourceSummary = `${sourceSummary} · slice`;
      continue;
    }
    if (op.kind === "permute") {
      const ax = normalizePermutation(op.axes, shape.length);
      const { shape: pshape, values: pvalues } = permuteRowMajor(shape, values, ax);
      shape = pshape;
      values = pvalues;
      sourceSummary = `${sourceSummary} · permute(${ax.join(",")})`;
      continue;
    }
    if (op.kind === "flatten") {
      const flat = applyFlattenRowMajorOk(shape, values, op.exceptDim);
      if (!flat) {
        return {
          kind: "none",
          detail: "Flatten in chain: invalid config or middle-axis except (use null, first, or last axis).",
        };
      }
      shape = flat.shape;
      values = flat.values;
      sourceSummary = `${sourceSummary} · flatten`;
      continue;
    }
  }
  return { ...base, rank: shape.length, shape, values, sourceSummary };
}

export async function hydrateResolved(resolved: Resolved): Promise<ResolvedOk | ResolvedNone> {
  if (resolved.kind === "none") return resolved;
  if (resolved.kind === "ok") return resolved;
  if (resolved.kind === "lazy_activation") return fetchActivationTensorAsOk(resolved);
  return fetchDatasetTensorAsOk(resolved);
}

export async function fetchActivationTensorAsOk(
  lazy: ResolvedLazyActivation,
): Promise<ResolvedOk | ResolvedNone> {
  const params = new URLSearchParams({ run_id: lazy.runId, rep_id: lazy.repId });
  const res = await fetch(`/api/activation_tensor?${params}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j?.detail != null) {
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      }
    } catch {
      const t = await res.text().catch(() => "");
      if (t) detail = t.slice(0, 400);
    }
    return { kind: "none", detail };
  }
  const shapeHeader = res.headers.get("X-Tensor-Shape");
  let shape: number[];
  try {
    shape = shapeHeader ? (JSON.parse(shapeHeader) as number[]) : lazy.shape;
  } catch {
    shape = lazy.shape;
  }
  const buf = await res.arrayBuffer();
  const values = Array.from(new Float32Array(buf));
  const rank = shape.length;
  const base: ResolvedOk = {
    kind: "ok",
    rank,
    shape,
    values,
    sourceSummary: lazy.sourceSummary,
  };
  return applyLazyOps(base, lazy.ops);
}

export async function fetchDatasetTensorAsOk(lazy: ResolvedLazyDataset): Promise<ResolvedOk | ResolvedNone> {
  const rawData = (lazy.datasetData ?? {}) as Record<string, unknown>;
  const datasetData =
    lazy.ops && lazy.ops.length > 0
      ? { ...rawData, inspectFormat: "id", format: "id" }
      : rawData;
  const res = await fetch("/api/dataset_tensor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_node_id: lazy.datasetNodeId,
      dataset_node_type: lazy.datasetNodeType,
      dataset_data: datasetData,
      graph_nodes: lazy.graphNodes ?? [],
      graph_edges: lazy.graphEdges ?? [],
      split: lazy.split,
      tensor_key: lazy.tensorKey,
    }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j?.detail != null) {
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      }
    } catch {
      const t = await res.text().catch(() => "");
      if (t) detail = t.slice(0, 400);
    }
    return { kind: "none", detail };
  }
  const inspectHdr = res.headers.get("X-Dataset-Inspect");
  if (inspectHdr === "word") {
    try {
      const j = (await res.json()) as {
        shape?: unknown;
        lines?: unknown;
        note?: unknown;
        tensorKey?: unknown;
        split?: unknown;
        inspect?: unknown;
      };
      const shape = Array.isArray(j.shape) ? j.shape.map((x) => Number(x)) : [];
      const lines = Array.isArray(j.lines) ? j.lines.map((x) => String(x)) : [];
      const textPreview = JSON.stringify(
        {
          inspect: j.inspect ?? "word",
          split: j.split,
          tensorKey: j.tensorKey,
          shape,
          lines,
          note: j.note,
        },
        null,
        2,
      );
      const n = Math.max(1, shape.reduce((a, b) => a * Math.max(0, Math.floor(b)), 1));
      const base: ResolvedOk = {
        kind: "ok",
        rank: shape.length,
        shape,
        values: new Array(Math.min(n, 50_000)).fill(0),
        sourceSummary: lazy.sourceSummary,
        textPreview,
      };
      return applyLazyOps(base, lazy.ops);
    } catch {
      return { kind: "none", detail: "Word inspect response could not be parsed." };
    }
  }

  const shapeHeader = res.headers.get("X-Tensor-Shape");
  let shape: number[] = [];
  try {
    shape = shapeHeader ? (JSON.parse(shapeHeader) as number[]) : [];
  } catch {
    shape = [];
  }
  const buf = await res.arrayBuffer();
  const values = Array.from(new Float32Array(buf));
  const base: ResolvedOk = {
    kind: "ok",
    rank: shape.length,
    shape,
    values,
    sourceSummary: lazy.sourceSummary,
  };
  return applyLazyOps(base, lazy.ops);
}
