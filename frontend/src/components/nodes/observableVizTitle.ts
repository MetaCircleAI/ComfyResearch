import { useMemo } from "react";
import { useStore, type Node } from "@xyflow/react";
import { nodeRegistryLabel, nodeRegistryObservableVizTitle } from "../../graph/nodeRegistrySpec";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";

/** Human-readable name for an observable node (matches sidebar / observable headers). */
export function getObservableNodeDisplayName(obs: Node): string {
  const d = (obs.data ?? {}) as { label?: string };
  if (
    obs.type === "observable_embedding_evolution" ||
    obs.type === "observable_embedding_trajectory" ||
    obs.type === "observable_user"
  ) {
    const custom = typeof d.label === "string" && d.label.trim() ? d.label.trim() : undefined;
    if (custom) return custom;
  }
  const type = String(obs.type ?? "");
  return nodeRegistryObservableVizTitle(type) ?? nodeRegistryLabel(type) ?? "Observable";
}

/** Title line: `{observable name} viz` when paired to an observable node; otherwise `Observable viz`. */
export function useObservableVizHeaderTitle(pairedObservableId: string | undefined, vizInstanceTitle?: unknown): string {
  const nodes = useStore((s) => s.nodes);
  return useMemo(() => {
    if (!pairedObservableId) return "Observable viz";
    const obs = nodes.find((n) => n.id === pairedObservableId);
    if (!obs) return "Observable viz";
    if (obs.type === "observable_attention_relation_score") {
      const title = readInstanceTitle({ instanceTitle: vizInstanceTitle }, "");
      if (/^viz\d+$/.test(title)) return title;
    }
    return `${getObservableNodeDisplayName(obs)} viz`;
  }, [pairedObservableId, nodes, vizInstanceTitle]);
}
