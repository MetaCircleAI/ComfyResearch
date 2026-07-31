export type ObservableUserNodeData = {
  userObservableId: string;
  label: string;
  /** Legacy; trainer prefers ``tensorSelectorNodeId`` or the server record. */
  tensorVizNodeId: string;
  /** In-graph tensor selector for live validation (optional if server record has it). */
  tensorSelectorNodeId: string;
  /** Deprecated: definitions are loaded from ``GET /api/user-observables/{id}`` into local state. */
  definitionCode?: string;
};

export function defaultObservableUserData(partial?: Partial<ObservableUserNodeData>): ObservableUserNodeData {
  return {
    userObservableId: partial?.userObservableId ?? "",
    label: partial?.label ?? "User observable",
    tensorVizNodeId: partial?.tensorVizNodeId ?? "",
    tensorSelectorNodeId: partial?.tensorSelectorNodeId ?? "",
    definitionCode: partial?.definitionCode,
  };
}
