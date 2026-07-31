import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useState, type ReactNode } from "react";
import { USER_OBSERVABLES_CHANGED } from "../../dnd";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
  type FlowNodeBare,
} from "../../graph/resolveUpstreamTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import { formatTensorScalarDisplay } from "./tensorVizScalarFormat";
import { TensorVizObsAddStrip } from "./TensorVizObsAddStrip";
import { VizSocketsBar } from "./VizSocketsBar";
import { defaultObservableUserData } from "./observableUserDefaults";

type TensorViz0dData = {
  optionalMetricName?: string;
};

export function TensorViz0dNode({ id, data, selected }: NodeProps) {
  const [addObsBusy, setAddObsBusy] = useState(false);
  const [addObsError, setAddObsError] = useState<string | null>(null);
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const d0 = (data ?? {}) as TensorViz0dData;
  const nameDraft = typeof d0.optionalMetricName === "string" ? d0.optionalMetricName : "";
  const setNameDraft = useCallback(
    (v: string) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...((n.data ?? {}) as Record<string, unknown>), optionalMetricName: v } } : n,
        ),
      );
    },
    [id, setNodes],
  );

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
    resolvedTensorEqual,
  );

  const { display, loading: tensorLoading } = useHydratedResolved(resolved);
  const hasObservableOut = useStore(
    useCallback(
      (state) =>
        state.edges.some((e) => e.source === id && (e.sourceHandle ?? "") === "observable"),
      [id],
    ),
  );

  const canAddObs =
    display.kind === "ok" && display.rank === 0 && display.values.length > 0;

  const handleAddObs = useCallback(async () => {
    if (!canAddObs) return;
    setAddObsError(null);
    setAddObsBusy(true);
    try {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const res = await fetch("/api/user-observables", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tensor_viz_node_id: id,
          nodes: g.nodes,
          edges: g.edges,
          ...(nameDraft.trim() ? { label: nameDraft.trim() } : {}),
        }),
      });
      if (res.ok) {
        const j = (await res.json()) as { item?: { id?: string; label?: string } };
        const itemId = (j.item?.id ?? "").trim();
        const itemLabel = (j.item?.label ?? "").trim() || nameDraft.trim() || "User observable";
        if (itemId) {
          const nodes = getNodes();
          const edges = getEdges();
          const existingObs = nodes.find(
            (n) =>
              n.type === "observable_user" &&
              String((n.data as { userObservableId?: string } | undefined)?.userObservableId ?? "") === itemId,
          );
          if (!existingObs) {
            const obsId = `observable_user-${Math.random().toString(36).slice(2, 10)}`;
            const self = nodes.find((n) => n.id === id);
            const obsPos = self
              ? { x: self.position.x + 280, y: self.position.y + 110 }
              : { x: 200, y: 200 };
            const newObs: Node = {
              id: obsId,
              type: "observable_user",
              position: obsPos,
              data: defaultObservableUserData({
                userObservableId: itemId,
                label: itemLabel,
                tensorVizNodeId: id,
              }),
            };
            const nextNodes = [...nodes, newObs];
            const nextEdges = [
              ...edges,
              {
                id: `e-${id}-observable-${obsId}`,
                source: id,
                target: obsId,
                sourceHandle: "observable",
                targetHandle: "observable_in",
              },
            ];
            setNodes(nextNodes);
            setEdges(nextEdges);
          }
        }
        window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
        return;
      }
      const t = await res.text().catch(() => "");
      setAddObsError(t ? t.slice(0, 220) : `Request failed (${res.status})`);
    } catch {
      setAddObsError("Network error while saving observable.");
    } finally {
      setAddObsBusy(false);
    }
  }, [canAddObs, getEdges, getNodes, id, nameDraft, setEdges, setNodes]);

  let body: ReactNode;
  if (display.kind === "none") {
    body = (
      <p className="cr-tensor-viz__hint">
        {tensorLoading ? "Loading tensor from server…" : display.detail}
      </p>
    );
  } else if (display.rank !== 0) {
    body = (
      <p className="cr-tensor-viz__hint">
        This node expects a 0-D (scalar) tensor. Connected: rank {display.rank}
        {display.shape.length ? `, shape [${display.shape.join(", ")}]` : ""}.
      </p>
    );
  } else if (display.values.length === 0) {
    body = <p className="cr-tensor-viz__hint">Scalar tensor has no stored value.</p>;
  } else {
    const v = display.values[0]!;
    body = (
      <div className="cr-tensor-viz__plot cr-tensor-viz-0d">
        <div className="cr-tensor-viz-0d__value" title={String(v)}>
          {formatTensorScalarDisplay(v)}
        </div>
        <p className="cr-tensor-viz__meta">{display.sourceSummary} · []</p>
      </div>
    );
  }

  return (
    <div
      className={`cr-node cr-node--tensor-viz cr-node--tensor-viz-0d${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">0D tensor viz</div>
      <TensorVizObsAddStrip
        nodeId={id}
        nameDraft={nameDraft}
        onNameChange={setNameDraft}
        canAdd={canAddObs}
        busy={addObsBusy}
        onAdd={handleAddObs}
      />
      <div className="cr-node__body cr-node__body--compact">
        {addObsError ? (
          <p className="cr-tensor-viz__hint" role="alert">
            {addObsError}
          </p>
        ) : null}
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="0D tensor viz input and outputs">
          <VizSocketsBar
            middleRight={
              hasObservableOut ? (
                <>
                  <span className="cr-tviz-socket-label">observable</span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id="observable"
                    className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--observable"
                  />
                </>
              ) : undefined
            }
          />
        </div>
        <div className="cr-tensor-viz__body nodrag nopan">{body}</div>
      </div>
    </div>
  );
}
