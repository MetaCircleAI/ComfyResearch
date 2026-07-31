import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  defaultDimensionPermutatorData,
  type DimensionPermutatorNodeData,
} from "./dimensionPermutatorDefaults";
import {
  resolveUpstreamTensor,
  type FlowEdge,
  type FlowNodeBare,
  type Resolved,
} from "../../graph/resolveUpstreamTensor";
import { formatPermutationAsEinstein, parseEinsteinPermutation } from "../../graph/permutationNotation";
import { normalizePermutation } from "../../graph/tensorPermute";

function patchDimensionPermutatorData(
  id: string,
  patch: Partial<DimensionPermutatorNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultDimensionPermutatorData();
      const cur = (n.data ?? {}) as Partial<DimensionPermutatorNodeData>;
      const prev: DimensionPermutatorNodeData = {
        axes: Array.isArray(cur.axes) ? [...cur.axes!] : [...def.axes],
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function resolvedRank(resolved: Resolved): number {
  if (resolved.kind === "ok") return resolved.rank;
  if (resolved.kind === "lazy_activation") return resolved.shape.length;
  return -1;
}

function resolvedStableKey(resolved: Resolved): string {
  if (resolved.kind === "none") return `n:${resolved.detail}`;
  if (resolved.kind === "lazy_activation") return `l:${resolved.runId}:${resolved.repId}:${resolved.shape.join(",")}`;
  if (resolved.kind === "lazy_dataset") {
    return `d:${resolved.datasetNodeId}:${resolved.datasetNodeType}:${resolved.split}:${resolved.tensorKey}:${resolved.sourceSummary}`;
  }
  return `o:${resolved.rank}:${resolved.shape.join(",")}:${resolved.values.length}:${resolved.sourceSummary}`;
}

function shapeFmt(shape: number[] | null): string {
  if (!shape || shape.length === 0) return "—";
  return `[${shape.join(", ")}]`;
}

export function DimensionPermutatorNode({ id, data, selected }: NodeProps) {
  const def = defaultDimensionPermutatorData();
  const raw = (data ?? {}) as Partial<DimensionPermutatorNodeData>;
  const d: DimensionPermutatorNodeData = {
    axes: Array.isArray(raw.axes) ? [...raw.axes] : [...def.axes],
  };
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<DimensionPermutatorNodeData>) => patchDimensionPermutatorData(id, patch, setNodes),
    [id, setNodes],
  );

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor_in"),
      [id],
    ),
    (a, b) => resolvedStableKey(a) === resolvedStableKey(b),
  );

  const rank = resolvedRank(resolved);
  const axes = useMemo(
    () => normalizePermutation(d.axes.length === rank && rank > 0 ? d.axes : undefined, Math.max(0, rank)),
    [rank, d.axes.join(",")],
  );
  const axesKey = axes.join(",");

  useEffect(() => {
    if (rank < 0) {
      if (d.axes.length > 0) update({ axes: [] });
      return;
    }
    if (rank === 0) {
      if (d.axes.length > 0) update({ axes: [] });
      return;
    }
    if (d.axes.length !== rank) {
      update({ axes: Array.from({ length: rank }, (_, i) => i) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- resync axes only when upstream tensor rank changes
  }, [rank]);

  const [draft, setDraft] = useState("");
  const [notationError, setNotationError] = useState<string | null>(null);
  const prevSyncKey = useRef<string>("");

  useEffect(() => {
    if (rank <= 0) {
      setDraft("");
      setNotationError(null);
      prevSyncKey.current = "";
      return;
    }
    const key = `${rank}:${axesKey}`;
    if (key === prevSyncKey.current) return;
    prevSyncKey.current = key;
    setDraft(formatPermutationAsEinstein(axes, rank));
    setNotationError(null);
  }, [rank, axesKey]);

  const commitDraft = useCallback(() => {
    if (rank <= 0) return;
    const r = parseEinsteinPermutation(draft, rank);
    if (!r.ok) {
      setNotationError(r.message);
      return;
    }
    setNotationError(null);
    update({ axes: r.axes });
  }, [draft, rank, update]);

  const resetIdentity = useCallback(() => {
    if (rank <= 0) return;
    update({ axes: Array.from({ length: rank }, (_, i) => i) });
  }, [rank, update]);

  const summary =
    rank <= 0
      ? resolved.kind === "none"
        ? "Connect a tensor on the left."
        : "Scalar — no dimensions to permute."
      : "Letters name input axes in order; after “->” list which input feeds each output axis (Einstein-style).";

  const upstreamShapeKey =
    resolved.kind === "ok" || resolved.kind === "lazy_activation" ? resolved.shape.join("\x1f") : "";

  const inShape = useMemo((): number[] | null => {
    if (resolved.kind !== "ok" && resolved.kind !== "lazy_activation") return null;
    return resolved.shape.map((x) => Number(x));
  }, [resolved.kind, upstreamShapeKey]);

  const outShapePreview = useMemo(() => {
    if (!inShape || rank <= 0 || inShape.length !== rank) return null;
    return axes.map((inDim) => inShape[inDim]!);
  }, [inShape, rank, axes]);

  return (
    <div
      className={`cr-node cr-node--dim-perm${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Dimension permutator</span>
          <button
            type="button"
            className="cr-trainer-train-btn nodrag nopan"
            disabled={rank <= 0}
            onClick={() => commitDraft()}
          >
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
          <div className="cr-tviz-socket-row__left">
            <Handle
              type="target"
              position={Position.Left}
              id="tensor_in"
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
                id="tensor_out"
                className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
              />
            </div>
          </div>
        </div>

        <p className="cr-dim-perm__hint">{summary}</p>

        {rank > 0 ? (
          <div className="cr-dim-perm__editor nodrag nopan">
            <input
              type="text"
              className="cr-dim-perm__ein-input cr-statistics-ein-input nodrag nopan"
              value={draft}
              spellCheck={false}
              aria-label="Permutation as i j k -> k j i"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitDraft();
                }
              }}
            />
            {notationError ? <p className="cr-dim-perm__error">{notationError}</p> : null}
            <div className="cr-dim-perm__actions">
              <button type="button" className="cr-dim-perm__reset" onClick={() => resetIdentity()}>
                Identity
              </button>
            </div>
            <p className="cr-dim-perm__help">
              Example: <code className="cr-dim-perm__code">i j k -&gt; k j i</code>. Commas optional.{" "}
              <kbd className="cr-dim-perm__kbd">Enter</kbd> or <strong>Compute</strong> commits the permutation.
            </p>
          </div>
        ) : null}

        <div className="cr-statistics-shape-footer cr-dim-perm__shape-footer" aria-live="polite">
          {shapeFmt(inShape)}
          {" → "}
          {shapeFmt(outShapePreview)}
        </div>
      </div>
    </div>
  );
}
