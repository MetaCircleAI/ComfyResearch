import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";
import { ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { intChoices, packIntList } from "./multiValueUtils";
import { SourceSocketRow } from "./SourceSocketRow";
import {
  defaultTensorConstantData,
  type TensorConstantInit,
  type TensorConstantNodeData,
} from "./tensorConstantDefaults";
import { generateTensorConstantValues, parseTensorShapeInput } from "../../graph/tensorConstantGenerate";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

const PREVIEW_CAP = 5;
const LARGE_DIM_THRESHOLD = 20;

function rowMajorStrides(shape: number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i]!;
  }
  return s;
}

function flatIndex(shape: number[], strides: number[], idx: number[]): number {
  let o = 0;
  for (let d = 0; d < shape.length; d++) o += idx[d]! * strides[d]!;
  return o;
}

function fmtCell(n: number): string {
  if (!Number.isFinite(n)) return "NaN";
  if (Number.isInteger(n) && Math.abs(n) < 1e15) return String(n);
  return String(Number(n.toPrecision(6)));
}

/**
 * Row-major tensor as nested bracket text (e.g. 2×2 matrix `[[0, 1],\n [2, 3]]`).
 * When {@link maxPerDim} is set, only the first {@link maxPerDim} entries along each axis are included; larger axes are cut with no in-matrix markers.
 */
function formatTensorBracketMatrix(
  shape: number[],
  values: number[],
  maxPerDim: number | null,
): string {
  if (!shape.length) return fmtCell(values[0] ?? 0);
  const strides = rowMajorStrides(shape);
  const idx = shape.map(() => 0);

  function joinBracketChildren(parts: string[]): string {
    if (parts.length === 0) return "[]";
    const preferRows =
      parts.length > 1 && parts.every((p) => p.startsWith("[") && p.endsWith("]") && !p.includes("\n"));
    const multiline = preferRows || parts.some((p) => p.includes("\n"));
    if (!multiline) return `[${parts.join(", ")}]`;
    const body = parts.map((p) => p.replace(/^/gm, "  ")).join(",\n");
    return `[\n${body}\n]`;
  }

  function rec(dim: number): string {
    if (dim >= shape.length) {
      return fmtCell(values[flatIndex(shape, strides, idx)]!);
    }
    const len = shape[dim]!;
    const cap = maxPerDim == null ? len : Math.min(len, maxPerDim);
    const parts: string[] = [];
    for (let i = 0; i < cap; i++) {
      idx[dim] = i;
      parts.push(rec(dim + 1));
    }
    return joinBracketChildren(parts);
  }

  return rec(0);
}

function formatShapeForInput(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

function patchTensorConstantData(
  id: string,
  patch: Partial<TensorConstantNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTensorConstantData();
      const cur = (n.data ?? {}) as Partial<TensorConstantNodeData>;
      const prev: TensorConstantNodeData = {
        shape: Array.isArray(cur.shape) && cur.shape.length ? [...cur.shape] : def.shape,
        init: cur.init ?? def.init,
        initSeed: cur.initSeed !== undefined ? cur.initSeed : def.initSeed,
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

const INIT_OPTIONS: { id: TensorConstantInit; label: string }[] = [
  { id: "zero", label: "Zeros" },
  { id: "uniform_m11", label: "Uniform [-1, 1]" },
  { id: "gaussian", label: "Gaussian N(0, 1)" },
];

export function TensorConstantNode({ id, data, selected }: NodeProps) {
  const def = defaultTensorConstantData();
  const raw = (data ?? {}) as Partial<TensorConstantNodeData>;
  const d: TensorConstantNodeData = {
    shape: Array.isArray(raw.shape) && raw.shape.length ? [...raw.shape] : def.shape,
    init: raw.init ?? def.init,
    initSeed: raw.initSeed !== undefined ? raw.initSeed : def.initSeed,
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<TensorConstantNodeData>) => patchTensorConstantData(id, patch, setNodes),
    [id, setNodes],
  );

  const regenSig = `${d.init}|${d.shape.join("x")}|${intChoices(d.initSeed, 0).join(",")}`;
  const stableSig = useRef<string>("__never__");
  const lastEmittedErr = useRef<string | null>(null);

  useEffect(() => {
    try {
      const vals = generateTensorConstantValues(d.shape, d.init, d.initSeed);
      const ot = d.outputTensor;
      const n = vals.length;
      if (
        stableSig.current === regenSig &&
        ot &&
        ot.values.length === n &&
        ot.shape.length === d.shape.length &&
        d.shape.every((s, i) => s === ot.shape[i]!)
      ) {
        return;
      }
      /* First mount with a persisted tensor: keep values (do not re-draw random init). */
      if (
        stableSig.current === "__never__" &&
        ot &&
        ot.values.length === n &&
        ot.shape.length === d.shape.length &&
        d.shape.every((s, i) => s === ot.shape[i]!)
      ) {
        stableSig.current = regenSig;
        lastEmittedErr.current = null;
        return;
      }
      stableSig.current = regenSig;
      lastEmittedErr.current = null;
      update({ outputTensor: { shape: [...d.shape], values: vals }, lastError: null });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (lastEmittedErr.current === msg) return;
      lastEmittedErr.current = msg;
      stableSig.current = "";
      update({ lastError: msg, outputTensor: null });
    }
  }, [regenSig, update]);

  const shapeKey = d.shape.join("x");
  const [modalOpen, setModalOpen] = useState(false);
  const [shapeDraft, setShapeDraft] = useState(() => formatShapeForInput(d.shape));
  const [showFull, setShowFull] = useState(false);
  const titleId = useId().replace(/:/g, "");

  useEffect(() => {
    setShapeDraft(formatShapeForInput(d.shape));
  }, [shapeKey]);

  useEffect(() => {
    if (!modalOpen) return;
    setShowFull(false);
  }, [modalOpen]);

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

  const previewMatrix = useMemo(() => {
    const t = d.outputTensor;
    if (!t?.shape?.length || !t.values?.length) return "— (no tensor yet)";
    return formatTensorBracketMatrix(t.shape, t.values, PREVIEW_CAP);
  }, [d.outputTensor]);

  const fullMatrix = useMemo(() => {
    const t = d.outputTensor;
    if (!t?.shape?.length || !t.values?.length) return "";
    return formatTensorBracketMatrix(t.shape, t.values, null);
  }, [d.outputTensor]);

  const fullViewLarge = useMemo(() => {
    const t = d.outputTensor?.shape;
    return Array.isArray(t) && t.some((x) => x > LARGE_DIM_THRESHOLD);
  }, [d.outputTensor]);

  const previewTruncates = useMemo(() => {
    const t = d.outputTensor?.shape;
    return Array.isArray(t) && t.length > 0 && t.some((s) => s > PREVIEW_CAP);
  }, [d.outputTensor]);

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
              Tensor · values
            </h2>
            <button type="button" className="cr-modal--code-view__close nodrag nopan" aria-label="Close" onClick={() => setModalOpen(false)}>
              ×
            </button>
          </div>
          <div className="cr-modal--tensor-constant__body nodrag nopan">
            <div className="cr-tensor-constant-preview-block">
              <h3 className="cr-tensor-constant-preview-title">Preview</h3>
              {previewTruncates ? (
                <p className="cr-tensor-constant-preview-note">
                  Only the first {PREVIEW_CAP} entries along each dimension are shown; the rest are omitted.
                </p>
              ) : null}
              <pre className="cr-tensor-constant-json cr-tensor-constant-json--matrix" tabIndex={0}>
                {previewMatrix}
              </pre>
            </div>

            <label className="cr-tensor-constant-expand-check nodrag nopan">
              <input type="checkbox" checked={showFull} onChange={(e) => setShowFull(e.target.checked)} />
              <span>Expand full tensor</span>
            </label>

            {showFull ? (
              <div className="cr-tensor-constant-full-block">
                {fullViewLarge ? (
                  <p className="cr-trainer-train-err" role="alert">
                    Warning: at least one dimension is longer than {LARGE_DIM_THRESHOLD}. Showing the full matrix may be
                    slow or hard to read in the browser.
                  </p>
                ) : null}
                <pre className="cr-tensor-constant-json cr-tensor-constant-json--full cr-tensor-constant-json--matrix" tabIndex={0}>
                  {fullMatrix || "—"}
                </pre>
              </div>
            ) : null}
          </div>
        </div>
      </div>,
      document.body,
    );

  return (
    <div
      className={`cr-node cr-node--tensor-constant${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor constant")}</div>
        </div>
      </div>
      <div className="cr-node__body">
        <SourceSocketRow handleId="tensor" label="tensor" />
        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label" title="Row-major sizes, e.g. [2, 3] or 2, 3. Press Enter or blur to apply.">
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
                aria-label="Tensor shape, e.g. [2, 3]"
              />
            </div>
          </div>
        </div>
        <DiscreteMultiSelect<TensorConstantInit>
          label="initialization"
          options={INIT_OPTIONS}
          value={d.init}
          onCommit={(next) =>
            update({
              init: (typeof next === "string" ? next : next[0] ?? d.init) as TensorConstantInit,
            })
          }
          ariaLabel="Tensor initialization"
          singleSelect
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(d.initSeed, 0)}
          min={0}
          title="RNG seed for uniform / Gaussian fills (ignored for zeros)."
          onCommit={(vals) => update({ initSeed: packIntList(vals) })}
          ariaLabel="Tensor init RNG seed"
        />
        <div className="cr-tensor-constant-footer nodrag nopan">
          <div className="cr-tensor-constant-footer__actions">
            <button type="button" className="cr-tensor-constant-footer__btn nodrag nopan" onClick={() => setModalOpen(true)}>
              View / edit parameters
            </button>
          </div>
        </div>
        {modal}
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
