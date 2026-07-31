import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { useCallback, useEffect } from "react";
import type { FlowEdge, FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { resolveUpstreamTensor } from "../../graph/resolveUpstreamTensor";
import { broadcastAddTensorPair } from "../../graph/tensorBroadcastAdd";
import { defaultTensorAddData, type TensorAddNodeData } from "./tensorAddDefaults";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

function patchTensorAddData(
  id: string,
  patch: Partial<TensorAddNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTensorAddData();
      const cur = (n.data ?? {}) as Partial<TensorAddNodeData>;
      const prev: TensorAddNodeData = {
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function shapeFmt(shape: number[] | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return `[${shape.join(", ")}]`;
}

export function TensorAddNode({ id, data, selected }: NodeProps) {
  const def = defaultTensorAddData();
  const raw = (data ?? {}) as Partial<TensorAddNodeData>;
  const d: TensorAddNodeData = {
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

  const shape1 = r1.kind === "ok" || r1.kind === "lazy_activation" ? r1.shape : null;
  const shape2 = r2.kind === "ok" || r2.kind === "lazy_activation" ? r2.shape : null;

  const update = useCallback(
    (patch: Partial<TensorAddNodeData>) => patchTensorAddData(id, patch, setNodes),
    [id, setNodes],
  );

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
    const sh1 = h1.shape.map((x) => Number(x));
    const sh2 = h2.shape.map((x) => Number(x));
    try {
      const { shape, values } = broadcastAddTensorPair(sh1, h1.values, sh2, h2.values);
      update({
        outputTensor: { shape, values },
        lastError: null,
      });
    } catch (e) {
      update({
        lastError: e instanceof Error ? e.message : String(e),
        outputTensor: null,
      });
    }
  }, [r1, r2, update]);

  return (
    <div
      className={`cr-node cr-node--tensor-add${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor add")}</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Tensor add inputs and output">
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

        <p className="cr-node__hint">
          Element-wise sum with NumPy-style broadcasting (e.g. <code>[10]</code> + <code>[3, 10]</code>).
        </p>

        <div className="cr-statistics-shape-footer" aria-live="polite">
          {shapeFmt(shape1)} + {shapeFmt(shape2)}
          {d.outputTensor?.shape?.length ? ` → ${shapeFmt(d.outputTensor.shape)}` : " → —"}
        </div>

        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
