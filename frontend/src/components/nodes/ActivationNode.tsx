import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useState } from "react";
import { ActivationWirePickerModal } from "../ActivationWirePickerModal";
import {
  resolveActivationWireModel,
  supportsActivationWirePicker,
} from "../../graph/resolveActivationWireModel";
import {
  buildRepresentationsForResidualDepth,
  defaultActivationData,
  type ActivationNodeData,
  type ActivationWirePick,
} from "./activationDefaults";
import { resolveModelForActivation } from "./resolveMlpForActivation";

type CollectActivationsResponse = {
  activation_run_id: string;
  manifest: Record<string, { shape: number[] }>;
  summary: string;
};

function patchActivationData(
  id: string,
  updater: (prev: ActivationNodeData) => ActivationNodeData,
  setNodes: (fn: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultActivationData();
      const cur = (n.data ?? {}) as Partial<ActivationNodeData>;
      const prev: ActivationNodeData = {
        representationOptions: cur.representationOptions ?? def.representationOptions,
        selectedRepresentationIds: cur.selectedRepresentationIds ?? def.selectedRepresentationIds,
        activationWirePicks: cur.activationWirePicks ?? def.activationWirePicks,
        scanMessage: cur.scanMessage ?? def.scanMessage,
        collectedActivations: cur.collectedActivations ?? def.collectedActivations,
        collectSummary: cur.collectSummary ?? def.collectSummary,
        activationRunId: cur.activationRunId ?? def.activationRunId,
        activationManifest: cur.activationManifest ?? def.activationManifest,
      };
      return { ...n, data: updater(prev) };
    }),
  );
}

export function ActivationNode({ id, data, selected }: NodeProps) {
  const def = defaultActivationData();
  const raw = (data ?? {}) as Partial<ActivationNodeData>;
  const d: ActivationNodeData = {
    representationOptions: raw.representationOptions ?? def.representationOptions,
    selectedRepresentationIds: raw.selectedRepresentationIds ?? def.selectedRepresentationIds,
    activationWirePicks: raw.activationWirePicks ?? def.activationWirePicks,
    scanMessage: raw.scanMessage ?? def.scanMessage,
    collectedActivations: raw.collectedActivations ?? def.collectedActivations,
    collectSummary: raw.collectSummary ?? def.collectSummary,
    activationRunId: raw.activationRunId ?? def.activationRunId,
    activationManifest: raw.activationManifest ?? def.activationManifest,
  };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [collectLoading, setCollectLoading] = useState(false);
  const [collectError, setCollectError] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);

  const wireTarget = resolveActivationWireModel(id, getNodes() as Node[], getEdges() as Edge[]);
  const canWirePick = supportsActivationWirePicker(wireTarget);

  const activationModelDigest = useStore(
    useCallback(
      (state) => {
        const nodes = state.nodes as Node[];
        const edges = state.edges as Edge[];
        const resolved = resolveModelForActivation(id, nodes, edges as Edge[]);
        if (!resolved) return "";
        const depth = Math.max(1, Math.floor(Number(resolved.data.depth) || 1));
        return `${resolved.nodeId}:${resolved.modelType}:${depth}`;
      },
      [id],
    ),
  );

  useEffect(() => {
    const nodes = getNodes() as Node[];
    const edges = getEdges() as Edge[];
    const resolved = resolveModelForActivation(id, nodes, edges);
    patchActivationData(
      id,
      (prev) => {
        if (!resolved) {
          return {
            ...prev,
            representationOptions: [],
            selectedRepresentationIds: [],
            scanMessage: null,
          };
        }
        const depth = Math.max(1, Math.floor(Number(resolved.data.depth) || 1));
        // MLP activations use wire picks only (no checkbox representations in the node UI).
        const representationOptions =
          resolved.modelType === "residual_ln_model" ? buildRepresentationsForResidualDepth(depth) : [];
        const allowed = new Set(representationOptions.map((o) => o.id));
        const selectedRepresentationIds = prev.selectedRepresentationIds.filter((x) =>
          allowed.has(x),
        );
        return {
          ...prev,
          representationOptions,
          selectedRepresentationIds,
          scanMessage: null,
        };
      },
      setNodes,
    );
  }, [activationModelDigest, getEdges, getNodes, id, setNodes]);

  const clearWirePicks = useCallback(() => {
    patchActivationData(id, (prev) => ({ ...prev, activationWirePicks: [] }), setNodes);
  }, [id, setNodes]);

  const saveWirePicks = useCallback(
    (picks: ActivationWirePick[]) => {
      patchActivationData(id, (prev) => ({ ...prev, activationWirePicks: picks }), setNodes);
    },
    [id, setNodes],
  );

  const toggleRep = useCallback(
    (repId: string, checked: boolean) => {
      patchActivationData(
        id,
        (prev) => {
          const set = new Set(prev.selectedRepresentationIds);
          if (checked) set.add(repId);
          else set.delete(repId);
          return { ...prev, selectedRepresentationIds: [...set] };
        },
        setNodes,
      );
    },
    [id, setNodes],
  );

  const selectAllReps = useCallback(() => {
    patchActivationData(
      id,
      (prev) => ({
        ...prev,
        selectedRepresentationIds: prev.representationOptions.map((o) => o.id),
      }),
      setNodes,
    );
  }, [id, setNodes]);

  const deselectAllReps = useCallback(() => {
    patchActivationData(id, (prev) => ({ ...prev, selectedRepresentationIds: [] }), setNodes);
  }, [id, setNodes]);

  const runCollect = useCallback(async () => {
    setCollectError(null);
    setCollectLoading(true);
    try {
      const nodes = getNodes().map((n) => ({
        id: n.id,
        type: n.type as string,
        position: n.position,
        parentId: n.parentId ?? undefined,
        data: (n.data as Record<string, unknown>) ?? {},
      }));
      const edges = getEdges().map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle ?? null,
        targetHandle: e.targetHandle ?? null,
      }));
      const res = await fetch("/api/collect_activations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          activation_node_id: id,
          nodes,
          edges,
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
          /* ignore */
        }
        throw new Error(detail);
      }
      const body = (await res.json()) as CollectActivationsResponse;
      patchActivationData(
        id,
        (prev) => ({
          ...prev,
          activationRunId: body.activation_run_id,
          activationManifest: body.manifest,
          collectedActivations: null,
          collectSummary: body.summary,
        }),
        setNodes,
      );
    } catch (e) {
      setCollectError(e instanceof Error ? e.message : String(e));
    } finally {
      setCollectLoading(false);
    }
  }, [getEdges, getNodes, id, setNodes]);

  const selectedSet = new Set(d.selectedRepresentationIds);

  return (
    <div
      className={`cr-node cr-node--activation${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--activation">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">Activation</span>
          <button
            type="button"
            className="cr-activation-collect-btn nodrag nopan"
            disabled={collectLoading}
            onClick={() => void runCollect()}
          >
            {collectLoading ? "…" : "Collect"}
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Activation inputs and tensor list output">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model"
              />
              <span className="cr-trainer-socket-label">model</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">tensor list</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor_list"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor-list"
              />
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset</span>
            </div>
          </div>
        </div>

        <div className="cr-activation-actions cr-activation-actions--wire nodrag nopan">
          <button
            type="button"
            className="cr-activation-wire-btn"
            disabled={!canWirePick}
            title={
              canWirePick
                ? "Open read-only graph to pick tensors along wires"
                : "Wire picking needs MLP, combined model, or atomic chain on the model socket"
            }
            onClick={() => setPickerOpen(true)}
          >
            Pick wires
          </button>
          {d.activationWirePicks.length > 0 ? (
            <div className="cr-activation-gauge-names" aria-label="Saved wire gauge names">
              {d.activationWirePicks.map((p, i) => (
                <span key={p.pickId || `${p.tensorKey}-${p.afterModuleIndex}-${i}`} className="cr-activation-gauge-chip">
                  {p.label}
                </span>
              ))}
            </div>
          ) : null}
        </div>

        {canWirePick && wireTarget ? (
          <ActivationWirePickerModal
            open={pickerOpen}
            onClose={() => setPickerOpen(false)}
            onSave={saveWirePicks}
            resolved={wireTarget}
            nodes={getNodes() as Node[]}
            edges={getEdges() as Edge[]}
            initialPicks={d.activationWirePicks}
          />
        ) : null}

        {d.activationWirePicks.length > 0 ? (
          <div className="cr-activation-wire-summary nodrag nopan">
            <p className="cr-activation-summary">
              {d.activationWirePicks.length} wire pick(s) — Collect runs hooks on the matching{' '}
              <code>nn.Sequential</code> path (canvas unchanged).
            </p>
            <button type="button" className="cr-activation-select-btn" onClick={clearWirePicks}>
              clear wire picks
            </button>
          </div>
        ) : null}

        {d.scanMessage ? <p className="cr-activation-scan-msg">{d.scanMessage}</p> : null}
        {d.representationOptions.length > 0 && !d.scanMessage && d.activationWirePicks.length === 0 ? (
          <p className="cr-activation-summary">
            {d.representationOptions.length} representations — checked items are collected on the train set when
            you press Collect.
          </p>
        ) : null}

        {d.collectSummary ? <p className="cr-activation-collect-summary">{d.collectSummary}</p> : null}

        {collectError ? <p className="cr-activation-scan-msg">{collectError}</p> : null}

        {d.representationOptions.length > 0 && d.activationWirePicks.length === 0 ? (
          <div className="cr-activation-select-actions nodrag nopan" aria-label="Representation selection">
            <button
              type="button"
              className="cr-activation-select-btn"
              onClick={selectAllReps}
              title="Check every representation"
            >
              select all
            </button>
            <button
              type="button"
              className="cr-activation-select-btn"
              onClick={deselectAllReps}
              title="Clear all checks"
            >
              deselect all
            </button>
          </div>
        ) : null}

        {d.representationOptions.length > 0 && d.activationWirePicks.length === 0 ? (
          <ul className="cr-activation-rep-list nodrag nopan" aria-label="Hidden representations">
            {d.representationOptions.map((opt) => (
              <li key={opt.id}>
                <label className="cr-activation-rep-check">
                  <input
                    type="checkbox"
                    checked={selectedSet.has(opt.id)}
                    onChange={(e) => toggleRep(opt.id, e.target.checked)}
                  />
                  <span>{opt.label}</span>
                </label>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}
