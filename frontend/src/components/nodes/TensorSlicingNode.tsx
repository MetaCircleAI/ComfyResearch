import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { resolveUpstreamTensor, type FlowEdge, type FlowNodeBare, type Resolved } from "../../graph/resolveUpstreamTensor";
import { defaultTensorSlicingNodeData, type TensorSliceSpec, type TensorSlicingNodeData } from "./tensorSlicingDefaults";
import {
  inferTensorSliceShape,
  normalizeSlices,
  parseSliceIndices,
  resolveTensorSliceIndex,
} from "../../graph/tensorSlice";

function shapeFmt(shape: number[] | null): string {
  if (!shape) return "—";
  return `[${shape.join(", ")}]`;
}

function resolvedStableKey(resolved: Resolved): string {
  if (resolved.kind === "none") return `n:${resolved.detail}`;
  if (resolved.kind === "lazy_activation") return `l:${resolved.runId}:${resolved.repId}:${resolved.shape.join(",")}`;
  if (resolved.kind === "lazy_dataset") {
    return `d:${resolved.datasetNodeId}:${resolved.datasetNodeType}:${resolved.split}:${resolved.tensorKey}:${resolved.sourceSummary}`;
  }
  return `o:${resolved.rank}:${resolved.shape.join(",")}:${resolved.values.length}:${resolved.sourceSummary}`;
}

function patchData(
  id: string,
  patch: Partial<TensorSlicingNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = (n.data ?? {}) as Partial<TensorSlicingNodeData>;
      const merged: TensorSlicingNodeData = {
        ...defaultTensorSlicingNodeData(),
        ...cur,
        ...patch,
        slices: normalizeSlices(patch.slices ?? cur.slices ?? []),
      };
      return { ...n, data: merged };
    }),
  );
}

function rowError(slice: TensorSliceSpec, inShape: number[] | null): string | null {
  if (!inShape) return null;
  if (!Number.isInteger(slice.dimension) || slice.dimension < 0 || slice.dimension >= inShape.length) {
    return `dimension must be in [0, ${Math.max(0, inShape.length - 1)}]`;
  }
  if (!slice.indices.trim()) return null;
  const idx = parseSliceIndices(slice.indices);
  if (!idx) return "indices must be comma-separated integers";
  const axisLen = inShape[slice.dimension] ?? 0;
  const bad = idx.find((v) => resolveTensorSliceIndex(v, axisLen) == null);
  if (bad != null) return `index ${bad} is out of range for dim ${slice.dimension} (size ${axisLen})`;
  return null;
}

export function TensorSlicingNode({ id, data, selected }: NodeProps) {
  const d = {
    ...defaultTensorSlicingNodeData(),
    ...(data as Partial<TensorSlicingNodeData>),
    slices: normalizeSlices((data as Partial<TensorSlicingNodeData>)?.slices ?? []),
  } satisfies TensorSlicingNodeData;
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<TensorSlicingNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  const resolved = useStore(
    useCallback(
      (state) => resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
    (a, b) => resolvedStableKey(a) === resolvedStableKey(b),
  );

  const inShape = useMemo((): number[] | null => {
    if (resolved.kind !== "ok" && resolved.kind !== "lazy_activation") return null;
    return resolved.shape.map((x) => Number(x));
  }, [resolved]);
  const outShape = useMemo(() => (inShape ? inferTensorSliceShape(inShape, d.slices) : null), [inShape, d.slices]);

  return (
    <div
      className={`cr-node cr-node--tensor-slicing${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor slicing")}</span>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
          <div className="cr-tviz-socket-row__left">
            <Handle
              type="target"
              position={Position.Left}
              id="tensor"
              className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
            />
            <span className="cr-tviz-socket-label">tensor</span>
          </div>
          <div className="cr-tviz-socket-row__right cr-tviz-socket-row--dual">
            <div className="cr-tviz-socket-pair">
              <span className="cr-tviz-socket-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
              />
            </div>
          </div>
        </div>
        <p className="cr-dim-perm__hint">
          Each slice picks one dimension and indices (comma-separated). Negative indices count from the end (-1 is last).
          Leave indices blank to keep all values for that dimension.
        </p>

        <div className="nodrag nopan">
          <div className="cr-comfy-widget cr-comfy-widget--flush">
            <span className="cr-comfy-widget__label" />
            <div className="cr-comfy-widget__control-col">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 6 }}>
                <span className="cr-node__hint cr-node__hint--extras">dimension</span>
                <span className="cr-node__hint cr-node__hint--extras" style={{ justifySelf: "start", marginLeft: -10 }}>
                  index/indices
                </span>
                <span />
              </div>
            </div>
          </div>
          {d.slices.map((slice, i) => {
            const err = rowError(slice, inShape);
            return (
              <div key={`slice-${i}`} className="cr-comfy-widget cr-comfy-widget--flush">
                <span className="cr-comfy-widget__label">{`slice ${i + 1}`}</span>
                <div className="cr-comfy-widget__control-col">
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 6 }}>
                    <input
                      type="number"
                      className="cr-input cr-comfy-widget__control nodrag nopan"
                      value={String(slice.dimension)}
                      aria-label={`Slice ${i + 1} dimension`}
                      onChange={(e) => {
                        const next = [...d.slices];
                        const n = Number(e.target.value);
                        next[i] = { ...slice, dimension: Number.isFinite(n) ? Math.trunc(n) : 0 };
                        update({ slices: next });
                      }}
                    />
                    <input
                      type="text"
                      className="cr-input cr-comfy-widget__control nodrag nopan"
                      value={slice.indices}
                      aria-label={`Slice ${i + 1} indices`}
                      placeholder="e.g. 1 or 0,2,4"
                      onChange={(e) => {
                        const next = [...d.slices];
                        next[i] = { ...slice, indices: e.target.value };
                        update({ slices: next });
                      }}
                    />
                    <button
                      type="button"
                      className="cr-dim-perm__reset"
                      onClick={() => update({ slices: d.slices.filter((_, idx) => idx !== i) })}
                      disabled={d.slices.length <= 1}
                    >
                      Remove
                    </button>
                  </div>
                  {err ? <p className="cr-dim-perm__error">{err}</p> : null}
                </div>
              </div>
            );
          })}
          <div className="cr-dim-perm__actions">
            <button
              type="button"
              className="cr-dim-perm__reset"
              onClick={() => update({ slices: [...d.slices, { dimension: 0, indices: "0" }] })}
            >
              Add slice
            </button>
          </div>
        </div>

        <div className="cr-statistics-shape-footer cr-dim-perm__shape-footer" aria-live="polite">
          {shapeFmt(inShape)}
          {" → "}
          {shapeFmt(outShape)}
        </div>
      </div>
    </div>
  );
}
