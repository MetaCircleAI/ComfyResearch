import { Handle, Position, useReactFlow, useStore, type NodeProps } from "@xyflow/react";
import katex from "katex";
import { useCallback, useEffect, useMemo, useState } from "react";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { compileElementwiseLatexRule } from "../../graph/elementwiseTransformRule";
import { resolveUpstreamTensor, type FlowEdge, type FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import {
  defaultElementwiseTransformData,
  type ElementwiseTransformNodeData,
} from "./elementwiseTransformDefaults";

/** LaTeX placed in the rule field; KaTeX preview + `compileElementwiseLatexRule` must accept them. */
const ELEMENTWISE_TRANSFORM_PRESETS: readonly { id: string; label: string; latex: string; title: string }[] = [
  { id: "x2", label: "x^2", latex: "x^2", title: "Square: x^2" },
  { id: "abs", label: "|x|", latex: "\\left|x\\right|", title: "Absolute value: |x|" },
  { id: "sin", label: "sin(x)", latex: "\\sin(x)", title: "Sine: sin(x)" },
  { id: "exp", label: "exp(x)", latex: "\\exp(x)", title: "Exponential: exp(x)" },
];

function patchElementwiseTransformData(
  id: string,
  patch: Partial<ElementwiseTransformNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = defaultElementwiseTransformData((n.data ?? {}) as Partial<ElementwiseTransformNodeData>);
      return { ...n, data: { ...cur, ...patch } };
    }),
  );
}

function shapeFmt(shape: number[] | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return `[${shape.join(", ")}]`;
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
    return <p className="cr-node__hint">Preview: enter a rule in x.</p>;
  }
  if (!html) {
    return <p className="cr-node__hint">Preview: KaTeX could not render.</p>;
  }
  return (
    <div
      className="cr-symbolic-dataset-katex nodrag nopan"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function ElementwiseTransformNode({ id, data, selected }: NodeProps) {
  const d = defaultElementwiseTransformData(data as Partial<ElementwiseTransformNodeData>);
  const [ruleDraft, setRuleDraft] = useState(d.ruleLatex);
  const { setNodes } = useReactFlow();

  useEffect(() => {
    setRuleDraft(d.ruleLatex);
  }, [d.ruleLatex]);

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
  );
  const inShape = resolved.kind === "ok" || resolved.kind === "lazy_activation" ? resolved.shape : null;

  const update = useCallback(
    (patch: Partial<ElementwiseTransformNodeData>) => patchElementwiseTransformData(id, patch, setNodes),
    [id, setNodes],
  );

  const applyPresetLatex = useCallback(
    (latex: string) => {
      const next = latex.trim() || "x^2";
      setRuleDraft(next);
      update({ ruleLatex: next });
    },
    [update],
  );

  useEffect(() => {
    if (resolved.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    update({ outputTensor: null, lastError: resolved.detail });
  }, [resolved, d.outputTensor, update]);

  const compute = useCallback(async () => {
    const hydrated = await hydrateResolved(resolved);
    if (hydrated.kind !== "ok") {
      update({ outputTensor: null, lastError: hydrated.detail });
      return;
    }
    try {
      const compiled = compileElementwiseLatexRule(ruleDraft);
      const outValues = hydrated.values.map((value) => compiled.evaluate(Number(value)));
      update({
        ruleLatex: ruleDraft.trim() || "x^2",
        outputTensor: { shape: [...hydrated.shape], values: outValues },
        lastError: null,
      });
    } catch (error) {
      update({
        outputTensor: null,
        lastError: error instanceof Error ? error.message : String(error),
      });
    }
  }, [resolved, ruleDraft, update]);

  return (
    <div
      className={`cr-node cr-node--elementwise-transform${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">
            {readInstanceTitle(data as Record<string, unknown>, "Elementwise transform")}
          </span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
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

        <div className="cr-comfy-widget cr-comfy-widget--stack cr-elementwise-rule">
          <div className="cr-elementwise-rule__head">
            <span className="cr-comfy-widget__label">f(x) = (LaTeX)</span>
            <div className="cr-elementwise-presets" role="group" aria-label="Preset transforms">
              {ELEMENTWISE_TRANSFORM_PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="cr-elementwise-presets__btn nodrag nopan"
                  title={p.title}
                  onClick={() => applyPresetLatex(p.latex)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            className="cr-input cr-symbolic-dataset-latex cr-comfy-widget__control nodrag nopan"
            rows={3}
            value={ruleDraft}
            onChange={(e) => setRuleDraft(e.target.value)}
            onBlur={() => update({ ruleLatex: ruleDraft.trim() || "x^2" })}
            spellCheck={false}
            aria-label="Elementwise transform rule in x"
          />
        </div>
        <LatexPreview latex={ruleDraft} />

        <p className="cr-node__hint">
          Apply one scalar rule to each tensor element (1D function in <code>x</code>), for example{" "}
          <code>x^2</code>, <code>\sin(x)</code>, or <code>\frac{"{x}"}{"{1+x^2}"}</code>.
        </p>

        <div className="cr-statistics-shape-footer" aria-live="polite">
          {shapeFmt(inShape)} {" → "} {shapeFmt(d.outputTensor?.shape ?? null)}
        </div>

        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
