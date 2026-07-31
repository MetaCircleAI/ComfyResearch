import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useId, useState, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";
import { parseTensorShapeInput } from "../../graph/tensorConstantGenerate";
import { runFakeTensorShapeCheck } from "../../graph/fakeTensorShapePropagation";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { useShapeCheckOverlay } from "../../context/shapeCheckOverlayContext";
import { SHAPE_CHECK_LABEL_DATA_KEY } from "../edges/ResearchDefaultEdge";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import { defaultFakeTensorData, type FakeTensorDtype, type FakeTensorNodeData } from "./fakeTensorDefaults";

const DTYPE_OPTIONS: { id: FakeTensorDtype; label: string }[] = [
  { id: "long", label: "long" },
  { id: "float", label: "float" },
];

function formatShapeForInput(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

function patchFakeTensorData(
  id: string,
  patch: Partial<FakeTensorNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultFakeTensorData();
      const cur = (n.data ?? {}) as Partial<FakeTensorNodeData>;
      const prev: FakeTensorNodeData = {
        shape: Array.isArray(cur.shape) && cur.shape.length ? [...cur.shape] : def.shape,
        dtype: cur.dtype === "long" || cur.dtype === "float" ? cur.dtype : def.dtype,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function FakeTensorNode({ id, data, selected }: NodeProps) {
  const def = defaultFakeTensorData();
  const raw = (data ?? {}) as Partial<FakeTensorNodeData>;
  const d: FakeTensorNodeData = {
    shape: Array.isArray(raw.shape) && raw.shape.length ? [...raw.shape] : def.shape,
    dtype: raw.dtype === "long" || raw.dtype === "float" ? raw.dtype : def.dtype,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes, getNodes, getEdges, setEdges } = useReactFlow();
  const shapeOverlay = useShapeCheckOverlay();
  const update = useCallback(
    (patch: Partial<FakeTensorNodeData>) => patchFakeTensorData(id, patch, setNodes),
    [id, setNodes],
  );

  const shapeKey = d.shape.join("x");
  const [modalOpen, setModalOpen] = useState(false);
  const [shapeDraft, setShapeDraft] = useState(() => formatShapeForInput(d.shape));
  const [checking, setChecking] = useState(false);
  const titleId = useId().replace(/:/g, "");

  useEffect(() => {
    setShapeDraft(formatShapeForInput(d.shape));
  }, [shapeKey]);

  const commitShapeFromDraft = useCallback(() => {
    try {
      const parsed = parseTensorShapeInput(shapeDraft);
      update({ shape: parsed, lastError: null });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      update({ lastError: msg });
      setShapeDraft(formatShapeForInput(d.shape));
    }
  }, [shapeDraft, update, d.shape]);

  const endShapeCheck = useCallback(() => {
    shapeOverlay?.clearShapeCheckErrors();
    setEdges((eds) =>
      eds.map((e) => {
        const cur = (e.data ?? {}) as Record<string, unknown>;
        if (cur[SHAPE_CHECK_LABEL_DATA_KEY] === undefined) return e;
        const next = { ...cur };
        delete next[SHAPE_CHECK_LABEL_DATA_KEY];
        return { ...e, data: Object.keys(next).length ? next : undefined };
      }),
    );
    setChecking(false);
  }, [setEdges, shapeOverlay]);

  const startShapeCheck = useCallback(() => {
    const nodes = getNodes();
    const edges = getEdges();
    const { errorNodeIds, edgeIdToLabel } = runFakeTensorShapeCheck(id, nodes, edges);
    shapeOverlay?.setShapeCheckErrors(errorNodeIds);
    setEdges((eds) =>
      eds.map((e) => {
        const lab = edgeIdToLabel[e.id];
        if (!lab) {
          const cur = (e.data ?? {}) as Record<string, unknown>;
          if (cur[SHAPE_CHECK_LABEL_DATA_KEY] === undefined) return e;
          const next = { ...cur };
          delete next[SHAPE_CHECK_LABEL_DATA_KEY];
          return { ...e, data: Object.keys(next).length ? next : undefined };
        }
        return {
          ...e,
          type: e.type ?? "research_default",
          data: { ...((e.data as Record<string, unknown>) ?? {}), [SHAPE_CHECK_LABEL_DATA_KEY]: lab },
        };
      }),
    );
    setChecking(true);
  }, [getEdges, getNodes, id, setEdges, shapeOverlay]);

  const onBackdrop = useCallback((e: ReactMouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) setModalOpen(false);
  }, []);

  useEffect(() => {
    if (!modalOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setModalOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  const modal =
    modalOpen &&
    createPortal(
      <div className="cr-modal-backdrop" style={{ zIndex: 10028 }} role="presentation" onMouseDown={onBackdrop}>
        <div
          className="cr-modal cr-modal--tensor-constant"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div className="cr-modal--tensor-constant__header">
            <h2 id={titleId} className="cr-modal__title">
              Fake tensor · shape
            </h2>
            <button type="button" className="cr-modal--code-view__close nodrag nopan" aria-label="Close" onClick={() => setModalOpen(false)}>
              ×
            </button>
          </div>
          <div className="cr-modal--tensor-constant__body nodrag nopan">
            <p className="cr-node__hint">
              For transformers, a typical rank-3 shape is <code>[batch, context, embedding]</code> (e.g.{" "}
              <code>[2, 3, 4]</code>).
            </p>
            <label className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
              <span className="cr-comfy-widget__label">shape</span>
              <div className="cr-comfy-widget__control-col">
                <input
                  type="text"
                  className="cr-input cr-comfy-widget__control cr-tensor-constant-shape-input nodrag nopan"
                  value={shapeDraft}
                  onChange={(e) => setShapeDraft(e.target.value)}
                  onBlur={commitShapeFromDraft}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      (e.target as HTMLInputElement).blur();
                    }
                  }}
                  spellCheck={false}
                  aria-label="Fake tensor shape"
                />
              </div>
            </label>
          </div>
        </div>
      </div>,
      document.body,
    );

  return (
    <div
      className={`cr-node cr-node--fake-tensor${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Fake tensor")}</div>
        </div>
      </div>
      <div className="cr-node__body">
        <SourceSocketRow handleId="tensor" label="tensor" />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label" title="Row-major sizes, e.g. [2, 3, 4]. Press Enter or blur to apply.">
              shape
            </span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="text"
                className="cr-input cr-comfy-widget__control cr-tensor-constant-shape-input nodrag nopan"
                value={shapeDraft}
                onChange={(e) => setShapeDraft(e.target.value)}
                onBlur={commitShapeFromDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    (e.target as HTMLInputElement).blur();
                  }
                }}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                aria-label="Tensor shape, e.g. [2, 3, 4]"
              />
            </div>
          </div>
        </div>
        <DiscreteMultiSelect<FakeTensorDtype>
          label="data type"
          options={DTYPE_OPTIONS}
          value={d.dtype}
          onCommit={(next) =>
            update({
              dtype: (typeof next === "string" ? next : next[0] ?? d.dtype) as FakeTensorDtype,
            })
          }
          ariaLabel="Fake tensor dtype"
          singleSelect
        />
        <div className="cr-tensor-constant-footer nodrag nopan">
          <div className="cr-tensor-constant-footer__actions cr-tensor-constant-footer__actions--split">
            <button type="button" className="cr-tensor-constant-footer__btn nodrag nopan" onClick={() => setModalOpen(true)}>
              Shape help
            </button>
            <button
              type="button"
              className="cr-trainer-train-btn nodrag nopan"
              onClick={() => {
                if (checking) endShapeCheck();
                else startShapeCheck();
              }}
            >
              {checking ? "End check" : "Check"}
            </button>
          </div>
        </div>
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
        {modal}
      </div>
    </div>
  );
}
