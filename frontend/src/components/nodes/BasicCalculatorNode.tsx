import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import katex from "katex";
import { useCallback, useEffect, useMemo, useState } from "react";
import { evaluateLatexScalarExpr } from "../../graph/latexScalarExpr";
import type { FlowEdge, FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { resolveUpstreamTensor } from "../../graph/resolveUpstreamTensor";
import {
  BASIC_CALCULATOR_INPUT_MAX,
  clampBasicCalculatorInputCount,
  defaultBasicCalculatorData,
  type BasicCalculatorNodeData,
} from "./basicCalculatorDefaults";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

function patchBasicCalculatorData(
  id: string,
  patch: Partial<BasicCalculatorNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultBasicCalculatorData();
      const cur = (n.data ?? {}) as Partial<BasicCalculatorNodeData>;
      const prev: BasicCalculatorNodeData = {
        inputCount: clampBasicCalculatorInputCount(cur.inputCount ?? def.inputCount),
        equationLatex: typeof cur.equationLatex === "string" ? cur.equationLatex : def.equationLatex,
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      const next: BasicCalculatorNodeData = { ...prev, ...patch };
      if (
        next.inputCount === prev.inputCount &&
        next.equationLatex === prev.equationLatex &&
        next.outputTensor === prev.outputTensor &&
        next.lastError === prev.lastError
      ) {
        return n;
      }
      return { ...n, data: next };
    }),
  );
}

function tensorAsSingleScalar(shape: number[], values: number[]): number | null {
  if (values.length !== 1) return null;
  const prod = shape.reduce((a, b) => a * b, 1);
  if (prod !== 1) return null;
  const v = values[0]!;
  return Number.isFinite(v) ? v : null;
}

function LatexPreview({ latex }: { latex: string }) {
  const html = useMemo(() => {
    const t = latex.trim();
    if (!t) return "";
    try {
      return katex.renderToString(t, { throwOnError: false, displayMode: true });
    } catch {
      return "";
    }
  }, [latex]);

  if (!latex.trim()) {
    return <p className="cr-node__hint">Preview: enter LaTeX (use x_1, x_2, … in the formula).</p>;
  }
  if (!html) {
    return <p className="cr-node__hint">Preview: KaTeX could not render this string.</p>;
  }
  return (
    <div
      className="cr-symbolic-dataset-katex nodrag nopan"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function BasicCalculatorNode({ id, data, selected }: NodeProps) {
  const def = defaultBasicCalculatorData();
  const raw = (data ?? {}) as Partial<BasicCalculatorNodeData>;
  const d: BasicCalculatorNodeData = {
    inputCount: clampBasicCalculatorInputCount(raw.inputCount ?? def.inputCount),
    equationLatex: typeof raw.equationLatex === "string" ? raw.equationLatex : def.equationLatex,
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };
  const inputCount = d.inputCount;
  const { setNodes } = useReactFlow();
  const [latexDraft, setLatexDraft] = useState(d.equationLatex);

  useEffect(() => {
    setLatexDraft(d.equationLatex);
  }, [d.equationLatex]);

  const handles = useMemo(
    () => Array.from({ length: inputCount }, (_, i) => ({ id: `tensor_${i + 1}`, label: `x_${i + 1}` })),
    [inputCount],
  );

  const resolvedList = useStore(
    useCallback(
      (state) => {
        const nodes = state.nodes as FlowNodeBare[];
        const edges = state.edges as FlowEdge[];
        const out: ReturnType<typeof resolveUpstreamTensor>[] = [];
        for (let i = 1; i <= inputCount; i++) {
          out.push(resolveUpstreamTensor(nodes, edges, id, `tensor_${i}`));
        }
        return out;
      },
      [id, inputCount],
    ),
  );

  const update = useCallback(
    (patch: Partial<BasicCalculatorNodeData>) => patchBasicCalculatorData(id, patch, setNodes),
    [id, setNodes],
  );

  useEffect(() => {
    if (latexDraft === d.equationLatex) return;
    const t = window.setTimeout(() => {
      update({ equationLatex: latexDraft });
    }, 250);
    return () => window.clearTimeout(t);
  }, [latexDraft, d.equationLatex, update]);

  useEffect(() => {
    const anyDisconnected = resolvedList.some((r) => r.kind === "none");
    if (!anyDisconnected) return;
    if (!d.outputTensor?.values?.length) return;
    const parts = resolvedList.map((r, i) => (r.kind === "none" ? `x_${i + 1}: ${r.detail}` : null)).filter(Boolean);
    update({ outputTensor: null, lastError: parts.join(" · ") || "Missing inputs." });
  }, [resolvedList, d.outputTensor, update]);

  const compute = useCallback(async () => {
    const scalars: number[] = [];
    for (let i = 0; i < inputCount; i++) {
      const r = resolvedList[i]!;
      const h = await hydrateResolved(r);
      if (h.kind !== "ok") {
        update({ lastError: `x_${i + 1}: ${h.detail}`, outputTensor: null });
        return;
      }
      const sh = h.shape.map((x) => Number(x));
      const sc = tensorAsSingleScalar(sh, h.values);
      if (sc == null) {
        update({
          lastError: `x_${i + 1} must be a scalar (single value, total size 1); got shape [${sh.join(", ")}].`,
          outputTensor: null,
        });
        return;
      }
      scalars.push(sc);
    }

    const scope: Record<string, number> = {};
    for (let i = 0; i < scalars.length; i++) {
      scope[`x_${i + 1}`] = scalars[i]!;
    }

    const ev = evaluateLatexScalarExpr(d.equationLatex, scope);
    if (!ev.ok) {
      update({
        lastError: ev.mathJsExpr ? `${ev.error} (→ ${ev.mathJsExpr})` : ev.error,
        outputTensor: null,
      });
      return;
    }
    update({
      outputTensor: { shape: [], values: [ev.value] },
      lastError: null,
    });
  }, [resolvedList, inputCount, d.equationLatex, update]);

  return (
    <div
      className={`cr-node cr-node--basic-calculator${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">
            {readInstanceTitle(data as Record<string, unknown>, "Basic calculator")}
          </span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Basic calculator inputs and output">
          {handles.map((h, idx) => (
            <div key={h.id} className="cr-trainer-io-row">
              <div className="cr-trainer-io-row__leftwrap">
                <Handle
                  type="target"
                  position={Position.Left}
                  id={h.id}
                  className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
                />
                <span className="cr-trainer-socket-label">{h.label}</span>
              </div>
              {idx === handles.length - 1 ? (
                <div className="cr-trainer-io-row__rightwrap">
                  <span className="cr-trainer-output-label">tensor</span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id="tensor"
                    className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
                  />
                </div>
              ) : (
                <div className="cr-trainer-io-row__rightwrap" />
              )}
            </div>
          ))}
        </div>

        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">scalar inputs</span>
            <div className="cr-comfy-widget__control-col">
              <input
                type="number"
                min={1}
                max={BASIC_CALCULATOR_INPUT_MAX}
                step={1}
                className="cr-input cr-comfy-widget__control nodrag nopan"
                value={inputCount}
                aria-label="Number of scalar inputs"
                onChange={(e) =>
                  patchBasicCalculatorData(
                    id,
                    { inputCount: clampBasicCalculatorInputCount(e.target.value) },
                    setNodes,
                  )
                }
              />
            </div>
          </div>
        </div>

        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span className="cr-comfy-widget__label">equation (LaTeX)</span>
            <textarea
              className="cr-input cr-symbolic-dataset-latex cr-comfy-widget__control nodrag nopan"
              rows={3}
              value={latexDraft}
              spellCheck={false}
              aria-label="LaTeX equation"
              onChange={(e) => setLatexDraft(e.target.value)}
            />
          </div>
        </div>

        <LatexPreview latex={latexDraft} />

        <p className="cr-node__hint">
          Each input must be a scalar tensor (rank-0 or all dimensions 1). Use{" "}
          <code className="cr-code">x_1</code>, <code className="cr-code">x_2</code>, … in the formula (supports{" "}
          <code className="cr-code">\frac</code>, <code className="cr-code">\sqrt</code>, <code className="cr-code">\cdot</code>, trig,
          etc.).
        </p>

        <div className="cr-statistics-shape-footer" aria-live="polite">
          {d.outputTensor?.shape?.length === 0 && d.outputTensor.values?.length
            ? `scalar → ${d.outputTensor.values[0]}`
            : "scalar → —"}
        </div>

        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
