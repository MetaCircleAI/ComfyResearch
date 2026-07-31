import { Handle, Position, useReactFlow, useStore, useUpdateNodeInternals, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { normalizeTensorSelectorData, type TensorSelectorNodeData } from "./tensorSelectorDefaults";
import { readActivationManifest, readActivationRunId } from "../../graph/activationNodeData";
import {
  orderedSelectedTensorKeysForPicker,
  tensorChoicesForTensorsInput,
  type FlowNodeBare,
  type TensorListChoice,
} from "../../graph/resolveUpstreamTensor";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { enumChoices, type ListOr1 } from "./multiValueUtils";

function patchTensorSelectorData(
  id: string,
  patch: Partial<TensorSelectorNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = (n.data ?? {}) as Partial<TensorSelectorNodeData>;
      const prev = normalizeTensorSelectorData(cur);
      const merged: TensorSelectorNodeData = { ...prev, ...patch };
      if (Array.isArray(merged.selectedTensorKeys)) {
        merged.selectedTensorKey = merged.selectedTensorKeys[0] ?? "";
      } else if (merged.selectedTensorKey) {
        merged.selectedTensorKeys = [merged.selectedTensorKey];
      }
      return { ...n, data: merged };
    }),
  );
}

function choicesEqual(a: TensorListChoice[], b: TensorListChoice[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i]!.id !== b[i]!.id || a[i]!.label !== b[i]!.label) return false;
  }
  return true;
}

const SWEEP_MS = 520;

export function TensorSelectorNode({ id, data, selected }: NodeProps) {
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const tensorChoices = useStore(
    useCallback(
      (state) => tensorChoicesForTensorsInput(state.nodes as FlowNodeBare[], state.edges, id),
      [id],
    ),
    choicesEqual,
  );

  const orderedKeys = useMemo(
    () => orderedSelectedTensorKeysForPicker((data ?? {}) as Partial<TensorSelectorNodeData>, tensorChoices),
    [data, tensorChoices],
  );

  /** Reference updates when the upstream activation node’s `data` changes (e.g. Collect). */
  const activationUpstreamData = useStore(
    useCallback((state) => {
      const edge = state.edges.find((e) => e.target === id && e.targetHandle === "tensor_list");
      if (!edge?.source) return null;
      const act = state.nodes.find((n) => n.id === edge.source);
      if (!act || act.type !== "activation") return null;
      return act.data ?? null;
    }, [id]),
  );

  const fetchGenRef = useRef(0);

  const update = useCallback(
    (patch: Partial<TensorSelectorNodeData>) => patchTensorSelectorData(id, patch, setNodes),
    [id, setNodes],
  );

  useEffect(() => {
    if (tensorChoices.length === 0) return;
    const ids = new Set(tensorChoices.map((c) => c.id));
    const raw = (data ?? {}) as Partial<TensorSelectorNodeData>;
    if (Array.isArray(raw.selectedTensorKeys) && raw.selectedTensorKeys.length === 0) {
      return;
    }
    const ord = orderedSelectedTensorKeysForPicker(raw, tensorChoices);
    if (ord.length > 0 && ord.every((k) => ids.has(k))) return;
    const nk = tensorChoices[0]!.id;
    update({ selectedTensorKeys: [nk], selectedTensorKey: nk });
  }, [tensorChoices, data, update]);

  useEffect(() => {
    const keys = orderedSelectedTensorKeysForPicker((data ?? {}) as Partial<TensorSelectorNodeData>, tensorChoices);
    const nodes = getNodes();
    const edges = getEdges();
    const selfNode = nodes.find((n) => n.id === id);
    const curNorm = normalizeTensorSelectorData((selfNode?.data ?? {}) as Partial<TensorSelectorNodeData>);
    const existingCaches = { ...(curNorm.activationTensorCaches ?? {}) };

    const edge = edges.find((e) => e.target === id && e.targetHandle === "tensor_list");
    if (!edge?.source || keys.length === 0) {
      if (Object.keys(existingCaches).length > 0 || curNorm.activationTensorCache !== null) {
        update({ activationTensorCaches: {}, activationTensorCache: null });
      }
      fetchGenRef.current += 1;
      return;
    }
    const act = nodes.find((n) => n.id === edge.source);
    if (!act || act.type !== "activation") {
      if (Object.keys(existingCaches).length > 0 || curNorm.activationTensorCache !== null) {
        update({ activationTensorCaches: {}, activationTensorCache: null });
      }
      fetchGenRef.current += 1;
      return;
    }

    const actRaw = (act.data ?? {}) as Record<string, unknown>;
    const runId = readActivationRunId(actRaw);
    const manifest = readActivationManifest(actRaw);

    const gen = ++fetchGenRef.current;
    let cancelled = false;

    const needFetch: string[] = [];
    for (const key of keys) {
      const entry = manifest?.[key];
      const shapeMeta = entry?.shape;
      if (!runId || !Array.isArray(shapeMeta) || shapeMeta.length === 0) {
        continue;
      }
      const shapeHint = shapeMeta.map((x) => Number(x));
      const expectedLen = shapeHint.reduce((a, b) => a * (Number.isFinite(b) && b > 0 ? b : 1), 1);
      const c = existingCaches[key];
      if (
        c &&
        c.runId === runId &&
        c.tensorKey === key &&
        c.values.length === expectedLen &&
        c.shape.length === shapeHint.length &&
        c.shape.every((v, i) => v === shapeHint[i]!)
      ) {
        continue;
      }
      needFetch.push(key);
    }

    if (needFetch.length === 0) {
      return;
    }

    (async () => {
      const merged: Partial<Record<string, (typeof existingCaches)[string]>> = { ...existingCaches };
      for (const key of needFetch) {
        const entry = manifest?.[key];
        const shapeMeta = entry?.shape;
        if (!runId || !Array.isArray(shapeMeta) || shapeMeta.length === 0) continue;
        const shapeHint = shapeMeta.map((x) => Number(x));
        try {
          const params = new URLSearchParams({ run_id: runId, rep_id: key });
          const res = await fetch(`/api/activation_tensor?${params}`);
          if (!res.ok) {
            if (res.status === 404) {
              // Run removed server-side; drop every cached slice for this run (not only the failing key).
              for (const k of Object.keys(merged)) {
                const ent = merged[k];
                if (ent && ent.runId === runId) delete merged[k];
              }
            } else {
              delete merged[key];
            }
            continue;
          }
          const buf = await res.arrayBuffer();
          const values = Array.from(new Float32Array(buf));
          const shapeHeader = res.headers.get("X-Tensor-Shape");
          let shape: number[];
          try {
            shape = shapeHeader ? (JSON.parse(shapeHeader) as number[]) : shapeHint;
          } catch {
            shape = shapeHint;
          }
          merged[key] = { runId, tensorKey: key, shape, values };
        } catch {
          delete merged[key];
        }
      }
      if (cancelled || gen !== fetchGenRef.current) return;
      const first = keys[0];
      const primary = first && merged[first] ? merged[first]! : null;
      update({
        activationTensorCaches: merged as TensorSelectorNodeData["activationTensorCaches"],
        activationTensorCache: primary,
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [activationUpstreamData, id, data, tensorChoices, update, getNodes, getEdges]);

  const idSet = useMemo(() => new Set(tensorChoices.map((c) => c.id)), [tensorChoices]);
  const fallbackId = tensorChoices[0]?.id ?? "";
  const multiValue: ListOr1<string> =
    orderedKeys.length === 0 ? [] : orderedKeys.length === 1 ? orderedKeys[0]! : orderedKeys;

  const [sweeping, setSweeping] = useState(false);
  const sweepTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sweepSeqRef = useRef(0);

  const stopSweep = useCallback(() => {
    if (sweepTimerRef.current != null) {
      clearInterval(sweepTimerRef.current);
      sweepTimerRef.current = null;
    }
    setSweeping(false);
    update({ tensorSelectorSweeping: false, tensorSelectorSweepSnapshots: {} });
  }, [update]);

  useEffect(() => () => stopSweep(), [stopSweep]);

  const toggleSweep = useCallback(() => {
    if (tensorChoices.length < 2 || orderedKeys.length === 0) return;
    if (sweepTimerRef.current != null) {
      stopSweep();
      return;
    }
    const allIds = tensorChoices.map((c) => c.id);
    const w = Math.max(1, orderedKeys.length);
    let start = 0;
    const first = orderedKeys[0];
    if (first) {
      const ix = allIds.indexOf(first);
      if (ix >= 0) start = ix;
    }
    sweepSeqRef.current = 0;
    setSweeping(true);
    update({ tensorSelectorSweepSnapshots: {} });
    const tick = () => {
      if (start + w > allIds.length) {
        stopSweep();
        return;
      }
      sweepSeqRef.current += 1;
      const seq = sweepSeqRef.current;
      const slice = allIds.slice(start, start + w);
      const primary = slice[0] ?? "";
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== id) return n;
          const prev = normalizeTensorSelectorData((n.data ?? {}) as Partial<TensorSelectorNodeData>);
          const snaps: Partial<Record<number, string[]>> = { ...(prev.tensorSelectorSweepSnapshots ?? {}) };
          snaps[seq] = [...slice];
          const seqNums = Object.keys(snaps)
            .map((k) => Number(k))
            .filter((x) => Number.isFinite(x))
            .sort((a, b) => a - b);
          while (seqNums.length > 500) {
            const k = seqNums.shift();
            if (k !== undefined) delete snaps[k];
          }
          return {
            ...n,
            data: {
              ...prev,
              selectedTensorKeys: slice,
              selectedTensorKey: primary,
              tensorSelectorSweeping: true,
              tensorSelectorSweepSeq: seq,
              tensorSelectorSweepSnapshots: snaps,
            },
          };
        }),
      );
      start += 1;
    };
    tick();
    sweepTimerRef.current = setInterval(tick, SWEEP_MS);
  }, [tensorChoices, orderedKeys, update, stopSweep, setNodes, id]);

  const outputCount = tensorChoices.length === 0 ? 1 : orderedKeys.length;

  useEffect(() => {
    // Tensor selector exposes a dynamic number of source handles (`tensor_1`, `tensor_2`, ...).
    // Keep React Flow's internal handle registry in sync after list/selection changes.
    updateNodeInternals(id);
  }, [id, outputCount, updateNodeInternals]);

  return (
    <div
      className={`cr-node cr-node--tensor-selector${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">Tensor selector</span>
          <button
            type="button"
            className="cr-activation-collect-btn nodrag nopan"
            disabled={tensorChoices.length < 2 || orderedKeys.length === 0}
            title={
              orderedKeys.length === 0
                ? "Select at least one tensor to sweep"
                : tensorChoices.length < 2
                ? "Need at least two upstream tensors to sweep"
                : sweeping
                  ? "Stop sweep"
                  : "Sweep selection downward along the list"
            }
            onClick={toggleSweep}
          >
            {sweeping ? "Stop" : "Sweep"}
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Tensor selector sockets">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="tensor_list"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor-list"
              />
              <span className="cr-trainer-socket-label">tensor list</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              {outputCount === 0 && tensorChoices.length > 0 ? (
                <span className="cr-trainer-output-label cr-trainer-output-label--muted">no outputs</span>
              ) : (
                <>
                  <span className="cr-trainer-output-label">tensor 1</span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id="tensor_1"
                    className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
                  />
                </>
              )}
            </div>
          </div>
          {outputCount > 1
            ? Array.from({ length: outputCount - 1 }, (_, i) => {
                const labelN = i + 2;
                const hid = `tensor_${labelN}`;
                return (
                  <div key={hid} className="cr-trainer-io-row">
                    <div className="cr-trainer-io-row__leftwrap" />
                    <div className="cr-trainer-io-row__rightwrap">
                      <span className="cr-trainer-output-label">{`tensor ${labelN}`}</span>
                      <Handle
                        type="source"
                        position={Position.Right}
                        id={hid}
                        className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
                      />
                    </div>
                  </div>
                );
              })
            : null}
        </div>

        <div className="nodrag nopan">
          {tensorChoices.length === 0 ? (
            <label className="cr-comfy-widget cr-comfy-widget--flush" htmlFor={`${id}-tensor-select`}>
              <span className="cr-comfy-widget__label">tensors</span>
              <div className="cr-comfy-widget__control-col">
                <span className="cr-comfy-widget__warn">
                  Connect a tensor list from Activation, Model weight tensors, a viz panel, Table viz (multi-line), or the
                  trainer’s loss / observable handles. Dataset nodes’ unified “dataset” output is also accepted (pick train vs test tensors in the list).
                </span>
              </div>
            </label>
          ) : (
            <DiscreteMultiSelect
              label="tensors"
              options={tensorChoices}
              value={multiValue}
              allowEmpty
              onCommit={(next) => {
                if (sweepTimerRef.current != null) {
                  clearInterval(sweepTimerRef.current);
                  sweepTimerRef.current = null;
                }
                setSweeping(false);
                sweepSeqRef.current = 0;
                const picked = enumChoices(next, idSet, fallbackId, true);
                const primary = picked[0] ?? "";
                update({
                  selectedTensorKeys: picked,
                  selectedTensorKey: primary,
                  tensorSelectorSweeping: false,
                  tensorSelectorSweepSeq: 0,
                  tensorSelectorSweepSnapshots: {},
                });
              }}
              ariaLabel="Select upstream tensors"
            />
          )}
        </div>
      </div>
    </div>
  );
}
