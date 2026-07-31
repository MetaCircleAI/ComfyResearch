import { useReactFlow, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useState } from "react";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";
import {
  defaultObservableUserData,
  type ObservableUserNodeData,
} from "./observableUserDefaults";

type UserObservableApiItem = {
  human_chain?: string;
  tensor_selector_node_id?: string;
};

export function ObservableUserNode({ id, selected, data }: NodeProps) {
  const raw = (data ?? {}) as Partial<ObservableUserNodeData>;
  const d = { ...defaultObservableUserData(), ...raw } as ObservableUserNodeData;
  const label = (d.label ?? "User observable").trim() || "User observable";
  const userObservableId = (d.userObservableId ?? "").trim();
  const tensorSelectorId = (d.tensorSelectorNodeId ?? "").trim();
  const tensorVizId = (d.tensorVizNodeId ?? "").trim();

  const { getNodes, getEdges } = useReactFlow();
  const [humanChain, setHumanChain] = useState<string | null>(null);
  const [defError, setDefError] = useState<string | null>(null);

  const runDescribePath = useCallback(
    async (pathAnchorId: string) => {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const res = await fetch("/api/user-observables/describe-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes: g.nodes,
          edges: g.edges,
          tensor_viz_node_id: pathAnchorId,
        }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const j = (await res.json()) as { human_chain?: string };
      setHumanChain((typeof j.human_chain === "string" && j.human_chain) || null);
      setDefError(null);
    },
    [getEdges, getNodes],
  );

  useEffect(() => {
    let cancelled = false;

    const waitForAnchorThenDescribe = async (anchor: string) => {
      setDefError(null);
      const deadline = Date.now() + 6000;
      while (!cancelled && Date.now() < deadline) {
        if (getNodes().some((n) => n.id === anchor)) {
          try {
            await runDescribePath(anchor);
          } catch (e) {
            if (!cancelled) {
              setDefError(e instanceof Error ? e.message : String(e));
              setHumanChain(null);
            }
          }
          return;
        }
        await new Promise((r) => setTimeout(r, 80));
      }
      if (!cancelled) {
        setDefError(`Observable anchor “${anchor}” is not in this graph (missing or outdated reference).`);
        setHumanChain(null);
      }
    };

    const run = async () => {
      if (userObservableId) {
        try {
          const res = await fetch(`/api/user-observables/${encodeURIComponent(userObservableId)}`);
          if (!res.ok) {
            let msg = res.statusText;
            try {
              const j = (await res.json()) as { detail?: unknown };
              if (j?.detail != null) {
                msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
              }
            } catch {
              /* ignore */
            }
            if (!cancelled) {
              setDefError(msg);
              setHumanChain(null);
            }
            return;
          }
          const j = (await res.json()) as { item?: UserObservableApiItem };
          const item = j.item;
          if (cancelled || !item) return;
          const hc = typeof item.human_chain === "string" ? item.human_chain : "";
          if (hc.trim()) {
            setHumanChain(hc.trim());
            setDefError(null);
            return;
          }
          const ts = (item.tensor_selector_node_id ?? "").trim();
          if (ts) {
            await waitForAnchorThenDescribe(ts);
            return;
          }
          if (!cancelled) {
            setDefError("Saved observable has no definition snapshot; re-add from the Observables panel.");
            setHumanChain(null);
          }
          return;
        } catch (e) {
          if (!cancelled) {
            setDefError(e instanceof Error ? e.message : String(e));
            setHumanChain(null);
          }
          return;
        }
      }

      const anchor = tensorSelectorId || tensorVizId;
      if (!anchor) {
        if (!cancelled) {
          setHumanChain(null);
          setDefError(null);
        }
        return;
      }
      await waitForAnchorThenDescribe(anchor);
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [userObservableId, tensorSelectorId, tensorVizId, runDescribePath]);

  return (
    <div
      className={`cr-node cr-node--observable-user cr-node--observable-weight-l2${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader id={id} graphNodeType="observable_user" title={label} />
      <div className="cr-node__body cr-node__body--compact nodrag nopan">
        <ObservableSourceStrip withTargetObservable />
        {defError ? (
          <p className="cr-observable-hint cr-observable-hint--error">{defError}</p>
        ) : (
          <p className="cr-observable-hint cr-observable-hint--chain">
            {humanChain ??
              "Connect a valid tensor pipeline; summary appears here (from saved definition or the graph)."}
          </p>
        )}
      </div>
    </div>
  );
}
