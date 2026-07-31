import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import katex from "katex";
import {
  defaultSymbolicFuncDatasetData,
  type SymbolicFuncDatasetNodeData,
} from "./symbolicFuncDatasetDefaults";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  INPUT_DISTRIBUTION_OPTIONS,
  OUTPUT_DISTRIBUTION_OPTIONS,
  type DatasetSamplingMode,
} from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  DEFAULT_SYMBOLIC_DATASET_PARAM_ORDER,
  DEFAULT_SYMBOLIC_DATASET_SPEC_NAME,
  generateSymbolicFuncDatasetSpecCode,
} from "../../graph/specCode/symbolicFuncDatasetSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

const FIXED_SYMBOLIC_OUTPUT_DIM = 1;
const SYMBOLIC_EVALUATION_PRECISION_OPTIONS = [
  { id: "input" as const, label: "Input dtype" },
  { id: "float64" as const, label: "Float64" },
];

function replaceNodeData(
  id: string,
  data: SymbolicFuncDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: SymbolicFuncDatasetNodeData,
  patch: Partial<SymbolicFuncDatasetNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function inferInputDimFromEquationLatex(latex: string): number | null {
  const re = /x_(?:\{\s*)?(\d+)(?:\s*\})?/g;
  let maxIdx = 0;
  let m: RegExpExecArray | null;
  do {
    m = re.exec(latex);
    if (!m) break;
    const idx = Number(m[1]);
    if (Number.isInteger(idx) && idx > maxIdx) maxIdx = idx;
  } while (m);
  return maxIdx > 0 ? maxIdx : null;
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
    return <p className="cr-node__hint">Preview: enter LaTeX for y(x).</p>;
  }
  if (!html) {
    return <p className="cr-node__hint">Preview: KaTeX could not render (still may parse on server).</p>;
  }
  return (
    <div
      className="cr-symbolic-dataset-katex nodrag nopan"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function SymbolicFuncDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultSymbolicFuncDatasetData();
  const d = { ...defs, ...(data as Partial<SymbolicFuncDatasetNodeData>) } as SymbolicFuncDatasetNodeData;
  const outputDim = intChoices(d.outputDim, FIXED_SYMBOLIC_OUTPUT_DIM)[0] ?? FIXED_SYMBOLIC_OUTPUT_DIM;
  const { setNodes } = useReactFlow();
  const [latexDraft, setLatexDraft] = useState(d.equationLatex);
  const [latexEditorTab, setLatexEditorTab] = useState<"write" | "preview">("preview");

  const order = useMemo(
    () => (d.paramOrder?.length ? d.paramOrder : [...DEFAULT_SYMBOLIC_DATASET_PARAM_ORDER]),
    [d.paramOrder],
  );
  const specName = d.specCodeName ?? DEFAULT_SYMBOLIC_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateSymbolicFuncDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );

  useEffect(() => {
    setLatexDraft(d.equationLatex);
  }, [d.equationLatex]);

  const update = useCallback(
    (patch: Partial<SymbolicFuncDatasetNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  useEffect(() => {
    if (outputDim !== FIXED_SYMBOLIC_OUTPUT_DIM) {
      update({ outputDim: FIXED_SYMBOLIC_OUTPUT_DIM });
    }
  }, [outputDim, update]);

  const onLatexBlur = useCallback(() => {
    const patch: Partial<SymbolicFuncDatasetNodeData> = {};
    if (latexDraft !== d.equationLatex) {
      patch.equationLatex = latexDraft;
    }
    const inferredDim = inferInputDimFromEquationLatex(latexDraft);
    if (inferredDim != null && intChoices(d.inputDim, 2)[0] !== inferredDim) {
      patch.inputDim = inferredDim;
    }
    if (Object.keys(patch).length > 0) update(patch);
  }, [d.equationLatex, d.inputDim, latexDraft, update]);

  const onLatexChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    const nextLatex = e.target.value;
    setLatexDraft(nextLatex);
    const inferredDim = inferInputDimFromEquationLatex(nextLatex);
    if (inferredDim != null && intChoices(d.inputDim, 2)[0] !== inferredDim) {
      update({ inputDim: inferredDim });
    }
  }, [d.inputDim, update]);

  return (
    <div
      className={`cr-node cr-node--symbolic-func-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="symbolic_func_dataset"
        nodeId={id}
        graphNodeType="symbolic_func_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Symbolic func dataset")}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <DatasetSourceSockets />

        <DiscreteMultiSelect<DatasetSamplingMode>
          label="train data sampling"
          options={DATASET_SAMPLING_MODE_OPTIONS}
          value={d.samplingMode ?? "fixed"}
          onCommit={(v) =>
            update({
              samplingMode: (typeof v === "string" ? v : v[0] ?? "fixed") as DatasetSamplingMode,
            })
          }
          ariaLabel="Train data sampling mode for the trainer"
          singleSelect
        />

        <div className="cr-comfy-widget cr-comfy-widget--stack cr-symbolic-dataset-equation">
          <div className="cr-symbolic-dataset-editor">
            <div className="cr-symbolic-dataset-editor__tabs" role="tablist" aria-label="Equation editor mode">
              <button
                type="button"
                role="tab"
                aria-selected={latexEditorTab === "write"}
                className={`cr-symbolic-dataset-editor__tab${latexEditorTab === "write" ? " is-active" : ""}`}
                onClick={() => setLatexEditorTab("write")}
              >
                Write
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={latexEditorTab === "preview"}
                className={`cr-symbolic-dataset-editor__tab${latexEditorTab === "preview" ? " is-active" : ""}`}
                onClick={() => setLatexEditorTab("preview")}
              >
                Preview
              </button>
            </div>
            {latexEditorTab === "write" ? (
              <textarea
                className="cr-input cr-symbolic-dataset-latex cr-comfy-widget__control nodrag nopan"
                rows={4}
                value={latexDraft}
                onChange={onLatexChange}
                onBlur={onLatexBlur}
                spellCheck={false}
                aria-label="LaTeX for target y as function of x"
              />
            ) : (
              <div className="cr-symbolic-dataset-preview-panel">
                <LatexPreview latex={latexDraft} />
              </div>
            )}
          </div>
        </div>

        <ComfyIntListField
          label={String.raw`input dim $d_{\mathrm in}$`}
          values={intChoices(d.inputDim, 2)}
          min={1}
          onCommit={(vals) => update({ inputDim: packIntList(vals) })}
          ariaLabel="Input dimension d"
        />

        <div className="cr-comfy-field">
          <div className="cr-comfy-widget cr-comfy-widget--flush">
            <span className="cr-comfy-widget__label" title="Fixed for symbolic scalar targets in v1">output dimension</span>
            <input
              className="cr-input cr-comfy-widget__control cr-comfy-widget__control--num"
              value={String(FIXED_SYMBOLIC_OUTPUT_DIM)}
              readOnly
              aria-label="Output dimension (fixed to 1)"
            />
          </div>
        </div>

        <DiscreteMultiSelect
          label="input distribution"
          options={INPUT_DISTRIBUTION_OPTIONS}
          value={typeof d.inputDistribution === "string" ? d.inputDistribution : "standard_normal"}
          onCommit={(next) =>
            update({
              inputDistribution: (typeof next === "string" ? next : next[0] ?? "standard_normal") as SymbolicFuncDatasetNodeData["inputDistribution"],
            })
          }
          ariaLabel="Input distribution"
          singleSelect
        />

        <DiscreteMultiSelect<"input" | "float64">
          label="target evaluation"
          options={SYMBOLIC_EVALUATION_PRECISION_OPTIONS}
          value={d.evaluationPrecision ?? "input"}
          onCommit={(next) =>
            update({
              evaluationPrecision:
                typeof next === "string" ? next : next[0] ?? "input",
            })
          }
          ariaLabel="Symbolic target evaluation precision"
          singleSelect
        />

        <DiscreteMultiSelect
          label="output noise"
          options={OUTPUT_DISTRIBUTION_OPTIONS}
          value={typeof d.outputDistribution === "string" ? d.outputDistribution : "deterministic"}
          onCommit={(next) =>
            update({
              outputDistribution: (typeof next === "string" ? next : next[0] ?? "deterministic") as SymbolicFuncDatasetNodeData["outputDistribution"],
            })
          }
          ariaLabel="Output distribution"
          singleSelect
        />

        <ComfyIntListField
          label="train size"
          values={intChoices(d.trainSize, 500)}
          min={1}
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
          ariaLabel="Training set size"
        />

        <ComfyIntListField
          label="test size"
          values={intChoices(d.testSize, 0)}
          min={0}
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
          ariaLabel="Test set size"
        />

        <ComfyFloatListField
          label="noise level (σ)"
          values={floatChoices(d.noiseLevel, 0)}
          min={0}
          positiveOnly={false}
          title="Gaussian noise on y when output noise is enabled"
          onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
          ariaLabel="Noise level sigma"
        />

        <ComfyIntListField
          label="seed"
          values={intChoices(d.seed, 0)}
          onCommit={(vals) => update({ seed: packIntList(vals) })}
          ariaLabel="Random seed"
        />

        {order
          .filter((k) => !["equationLatex", "inputDim", "outputDim", "inputDistribution", "evaluationPrecision", "outputDistribution", "trainSize", "testSize", "noiseLevel", "seed"].includes(k))
          .map((key) => {
            const ex = d.extras;
            if (!ex || !(key in ex)) return null;
            const v = ex[key as keyof typeof ex];
            if (typeof v === "number") {
              return (
                <ComfyFloatListField
                  key={key}
                  label={key}
                  values={floatChoices(v, 0)}
                  positiveOnly={false}
                  title={`${key} (from spec)`}
                  onCommit={(vals) => update({ extras: { ...ex, [key]: vals.length <= 1 ? (vals[0] ?? 0) : vals[0] } })}
                  ariaLabel={key}
                />
              );
            }
            if (typeof v === "boolean") {
              return (
                <div key={key} className="cr-comfy-widget">
                  <span className="cr-comfy-widget__label">{key}</span>
                  <label className="cr-comfy-widget__control">
                    <input
                      type="checkbox"
                      checked={v}
                      onChange={(e) => update({ extras: { ...ex, [key]: e.target.checked } })}
                    />
                  </label>
                </div>
              );
            }
            if (typeof v === "string") {
              return (
                <div key={key} className="cr-comfy-widget">
                  <span className="cr-comfy-widget__label">{key}</span>
                  <input
                    className="cr-input cr-comfy-widget__control"
                    value={v}
                    onChange={(e) => update({ extras: { ...ex, [key]: e.target.value } })}
                    aria-label={key}
                  />
                </div>
              );
            }
            return null;
          })}

      </div>
    </div>
  );
}
