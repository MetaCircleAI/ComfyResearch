import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo } from "react";
import type { FlowEdge, FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { resolveUpstreamTensor } from "../../graph/resolveUpstreamTensor";
import { binaryTensorEinsteinPair, inferBinaryOutputShapeSafe, parseBinaryOperandEinstein } from "../../graph/einsumCustom";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultStatistics2Data,
  resolvedStatistics2Einsum,
  type Statistics2NodeData,
  type Statistics2NodeDataWithLegacy,
} from "./statistics2Defaults";
import type { Statistics2PairOp } from "../../graph/statistics2Pair";

const PAIR_OPTIONS: { value: Statistics2PairOp; label: string }[] = [
  { value: "dot", label: "Dot product" },
  { value: "cosine_similarity", label: "Cosine similarity" },
];

function shapeFmt(shape: number[] | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return `[${shape.join(", ")}]`;
}

function patchStatistics2Data(
  id: string,
  patch: Partial<Statistics2NodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultStatistics2Data();
      const cur = (n.data ?? {}) as Partial<Statistics2NodeDataWithLegacy>;
      const prev: Statistics2NodeData = {
        einsumSubscripts: (typeof cur.einsumSubscripts === "string" && cur.einsumSubscripts.trim()
          ? cur.einsumSubscripts
          : def.einsumSubscripts
        ).trim(),
        pairReduction: cur.pairReduction ?? def.pairReduction,
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function Statistics2Node({ id, data, selected }: NodeProps) {
  const def = defaultStatistics2Data();
  const raw = (data ?? {}) as Partial<Statistics2NodeDataWithLegacy>;
  const rawExpr = typeof raw.einsumSubscripts === "string" ? raw.einsumSubscripts : "";
  const d: Statistics2NodeData = {
    einsumSubscripts: rawExpr.trim().length > 0 ? rawExpr : def.einsumSubscripts,
    pairReduction: raw.pairReduction ?? def.pairReduction,
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes } = useReactFlow();

  const r1 = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor_1"),
      [id],
    ),
  );
  const r2 = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor_2"),
      [id],
    ),
  );

  const rank1 =
    r1.kind === "ok" ? r1.rank : r1.kind === "lazy_activation" ? r1.shape.length : null;
  const rank2 =
    r2.kind === "ok" ? r2.rank : r2.kind === "lazy_activation" ? r2.shape.length : null;

  const shape1 = r1.kind === "ok" || r1.kind === "lazy_activation" ? r1.shape : null;
  const shape2 = r2.kind === "ok" || r2.kind === "lazy_activation" ? r2.shape : null;

  const rankGuess = rank1 ?? rank2;
  const normalizedExpr = useMemo(() => {
    const t = rawExpr.trim().replace(/\s+/g, "");
    return t.length > 0 ? t : resolvedStatistics2Einsum(raw, rankGuess, def);
  }, [rawExpr, raw, rankGuess, def]);

  const displayExpr = rawExpr.trim().length > 0 ? rawExpr : normalizedExpr;

  const parseErr = useMemo((): string | null => {
    try {
      const { lhs0, lhs1 } = parseBinaryOperandEinstein(normalizedExpr);
      if (rank1 != null && lhs0.length !== rank1) {
        return `First tensor: left side has ${lhs0.length} letters but rank is ${rank1}.`;
      }
      if (rank2 != null && lhs1.length !== rank2) {
        return `Second tensor: left side has ${lhs1.length} letters but rank is ${rank2}.`;
      }
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, [normalizedExpr, rank1, rank2]);

  const update = useCallback(
    (patch: Partial<Statistics2NodeData>) => patchStatistics2Data(id, patch, setNodes),
    [id, setNodes],
  );

  const outShapePreview = useMemo(() => {
    if (parseErr || !shape1 || !shape2) return null;
    return inferBinaryOutputShapeSafe(shape1, shape2, normalizedExpr);
  }, [parseErr, shape1, shape2, normalizedExpr]);

  const inputShapesLine = `${shapeFmt(shape1)} x ${shapeFmt(shape2)}`;

  useEffect(() => {
    if (r1.kind !== "none" && r2.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    const detail =
      r1.kind === "none" && r2.kind === "none"
        ? `${r1.detail} / ${r2.detail}`
        : r1.kind === "none"
          ? `Tensor 1: ${r1.detail}`
          : `Tensor 2: ${r2.detail}`;
    update({ outputTensor: null, lastError: detail });
  }, [r1, r2, d.outputTensor, update]);

  const compute = useCallback(async () => {
    const h1 = await hydrateResolved(r1);
    const h2 = await hydrateResolved(r2);
    if (h1.kind !== "ok") {
      update({ lastError: `Tensor 1: ${h1.detail}`, outputTensor: null });
      return;
    }
    if (h2.kind !== "ok") {
      update({ lastError: `Tensor 2: ${h2.detail}`, outputTensor: null });
      return;
    }
    const rawD = (data ?? {}) as Partial<Statistics2NodeDataWithLegacy>;
    const rs = typeof rawD.einsumSubscripts === "string" ? rawD.einsumSubscripts.trim().replace(/\s+/g, "") : "";
    const eff = rs.length > 0 ? rs : resolvedStatistics2Einsum(rawD, h1.rank, def);
    if (!eff.trim()) {
      update({ lastError: "Enter einsum subscripts (e.g. ij, ik -> jk).", outputTensor: null });
      return;
    }
    const sh1 = h1.shape.map((x) => Number(x));
    const sh2 = h2.shape.map((x) => Number(x));
    try {
      const { shape: outShape, values } = binaryTensorEinsteinPair(
        sh1,
        h1.values,
        sh2,
        h2.values,
        eff,
        d.pairReduction,
      );
      update({
        outputTensor: { shape: outShape, values },
        lastError: null,
      });
    } catch (e) {
      update({
        lastError: e instanceof Error ? e.message : String(e),
        outputTensor: null,
      });
    }
  }, [data, def, d.pairReduction, r1, r2, update]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Statistics 2</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Statistics2 tensor inputs and output">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_1"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor 1</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap" />
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_2"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor 2</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
              />
            </div>
          </div>
        </div>

        <fieldset className="cr-statistics-fieldset nodrag nopan">
          <legend className="cr-statistics-legend">Einstein / einsum (pair)</legend>
          <p className="cr-statistics-hint">
            Two-operand NumPy-style subscripts. Shared letters contract (multiply, then sum over those
            indices). Pair reduction applies per output cell: dot matches{" "}
            <code className="cr-statistics-code">numpy.einsum</code> on the same pattern; cosine divides by
            both L2 norms along the contracted indices.
          </p>
          <textarea
            className="cr-statistics-ein-input nodrag nopan"
            rows={2}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            value={displayExpr}
            onChange={(e) => update({ einsumSubscripts: e.target.value })}
            aria-label="Einsum subscripts for two tensors"
          />
          {parseErr ? <p className="cr-trainer-train-err" style={{ marginTop: 6 }}>{parseErr}</p> : null}
        </fieldset>

        <DiscreteMultiSelect<Statistics2PairOp>
          label="pair reduction"
          options={PAIR_OPTIONS.map((o) => ({ id: o.value, label: o.label }))}
          value={d.pairReduction}
          onCommit={(next) =>
            update({
              pairReduction: (typeof next === "string" ? next : next[0] ?? d.pairReduction) as Statistics2PairOp,
            })
          }
          ariaLabel="Pair reduction"
          singleSelect
        />

        <div className="cr-statistics-shape-footer" aria-live="polite">
          {inputShapesLine}
          {" → "}
          {shapeFmt(outShapePreview)}
        </div>

        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
