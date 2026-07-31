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
import {
  inferSingleOutputShapeFromShape,
  inferSingleOutputShapeFromRank,
  parseSingleOperandEinstein,
  singleTensorEinsteinReduce,
} from "../../graph/einsumCustom";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultStatisticsData,
  resolvedEinsumSubscripts,
  type StatisticsNodeData,
  type StatisticsNodeDataWithLegacy,
  type StatisticsReductionOp,
} from "./statisticsDefaults";

const OP_OPTIONS: { value: StatisticsReductionOp; label: string }[] = [
  { value: "mean", label: "mean" },
  { value: "median", label: "median" },
  { value: "max", label: "max" },
  { value: "min", label: "min" },
  { value: "l2_norm", label: "L2 norm" },
  { value: "l1_norm", label: "L1 norm" },
];

function shapeFmt(shape: number[] | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return `[${shape.join(", ")}]`;
}

function patchStatisticsData(
  id: string,
  patch: Partial<StatisticsNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultStatisticsData();
      const cur = (n.data ?? {}) as Partial<StatisticsNodeDataWithLegacy>;
      const prev: StatisticsNodeData = {
        einsumSubscripts: (typeof cur.einsumSubscripts === "string" && cur.einsumSubscripts.trim()
          ? cur.einsumSubscripts
          : def.einsumSubscripts
        ).trim(),
        reductionOp: cur.reductionOp ?? def.reductionOp,
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function StatisticsNode({ id, data, selected }: NodeProps) {
  const def = defaultStatisticsData();
  const raw = (data ?? {}) as Partial<StatisticsNodeDataWithLegacy>;
  const rawExpr = typeof raw.einsumSubscripts === "string" ? raw.einsumSubscripts : "";
  const d: StatisticsNodeData = {
    einsumSubscripts: rawExpr.trim().length > 0 ? rawExpr : def.einsumSubscripts,
    reductionOp: raw.reductionOp ?? def.reductionOp,
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
  );

  const rank =
    resolved.kind === "ok"
      ? resolved.rank
      : resolved.kind === "lazy_activation"
        ? resolved.shape.length
        : null;
  const hasTensor = rank !== null && rank >= 1;

  const normalizedExpr = useMemo(() => {
    const t = rawExpr.trim().replace(/\s+/g, "");
    return t.length > 0 ? t : resolvedEinsumSubscripts(raw, rank, def);
  }, [rawExpr, raw, rank, def]);

  const displayExpr = rawExpr.trim().length > 0 ? rawExpr : normalizedExpr;

  const subscriptParseError = useMemo((): string | null => {
    try {
      parseSingleOperandEinstein(normalizedExpr);
      if (rank != null) {
        const { lhs } = parseSingleOperandEinstein(normalizedExpr);
        if (lhs.length !== rank) {
          return `Left side needs ${rank} axis letters for this tensor (got ${lhs.length}).`;
        }
      }
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, [normalizedExpr, rank]);

  const previewShape =
    resolved.kind === "ok" || resolved.kind === "lazy_activation" ? resolved.shape : null;

  const previewOutShape = useMemo(() => {
    if (!previewShape || !rank || subscriptParseError) return null;
    try {
      return inferSingleOutputShapeFromShape(previewShape, normalizedExpr);
    } catch {
      return inferSingleOutputShapeFromRank(rank, normalizedExpr);
    }
  }, [previewShape, rank, subscriptParseError, normalizedExpr]);

  const update = useCallback(
    (patch: Partial<StatisticsNodeData>) => patchStatisticsData(id, patch, setNodes),
    [id, setNodes],
  );

  useEffect(() => {
    if (resolved.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    update({ outputTensor: null, lastError: resolved.detail });
  }, [resolved, d.outputTensor, update]);

  const compute = useCallback(async () => {
    const r = await hydrateResolved(resolved);
    if (r.kind !== "ok") {
      update({ lastError: r.detail, outputTensor: null });
      return;
    }
    if (r.rank < 1) {
      update({ lastError: "Input must be a tensor with rank ≥ 1.", outputTensor: null });
      return;
    }
    const rawD = (data ?? {}) as Partial<StatisticsNodeDataWithLegacy>;
    const rs = typeof rawD.einsumSubscripts === "string" ? rawD.einsumSubscripts.trim().replace(/\s+/g, "") : "";
    const eff = rs.length > 0 ? rs : resolvedEinsumSubscripts(rawD, r.rank, def);
    if (!eff.trim()) {
      update({ lastError: "Enter einsum subscripts (e.g. ij -> j).", outputTensor: null });
      return;
    }
    try {
      const { shape: outShape, values } = singleTensorEinsteinReduce(
        r.shape,
        r.values,
        eff,
        d.reductionOp,
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
  }, [data, def, d.reductionOp, resolved, update]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">Statistics</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={compute}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Statistics tensor in and out">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
              />
              <span className="cr-trainer-socket-label">tensor</span>
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
          <legend className="cr-statistics-legend">Einstein / einsum</legend>
          <p className="cr-statistics-hint">
            Single-operand NumPy-style subscripts (letters per axis in order). Labels that appear only on
            the left of <code className="cr-statistics-code">{"->"}</code> are collapsed using the reduction
            below (not summed).
          </p>
          <textarea
            className="cr-statistics-ein-input nodrag nopan"
            rows={2}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            value={displayExpr}
            onChange={(e) => update({ einsumSubscripts: e.target.value })}
            aria-label="Einsum subscripts for one tensor"
          />
          {!hasTensor ? (
            <p className="cr-statistics-hint">Connect a tensor to validate letters vs rank.</p>
          ) : subscriptParseError ? (
            <p className="cr-trainer-train-err" style={{ marginTop: 6 }}>
              {subscriptParseError}
            </p>
          ) : null}
        </fieldset>

        <DiscreteMultiSelect<StatisticsReductionOp>
          label="reduction"
          options={OP_OPTIONS.map((o) => ({ id: o.value, label: o.label }))}
          value={d.reductionOp}
          onCommit={(next) =>
            update({
              reductionOp: (typeof next === "string" ? next : next[0] ?? d.reductionOp) as StatisticsReductionOp,
            })
          }
          ariaLabel="Reduction operation"
          singleSelect
        />

        <div
          className="cr-statistics-shape-footer"
          aria-live="polite"
          title={
            d.outputTensor
              ? "Downstream nodes use the last successful Compute."
              : "Shape preview from the connected tensor; click Compute to write the output."
          }
        >
          {shapeFmt(previewShape)}
          {" → "}
          {shapeFmt(previewOutShape)}
        </div>

        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
