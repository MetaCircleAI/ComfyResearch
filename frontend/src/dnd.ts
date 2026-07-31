/** React Flow drag type for node library → canvas. */
export const DND_MIME = "application/reactflow";

/** Fallback: some browsers only expose ``text/plain`` reliably on ``drop`` for cross-widget drags. */
export const DND_TEXT_PLAIN = "text/plain";

/** Extra payload when dragging a user-defined observable (same node type `observable_user`). */
export const USER_OBSERVABLE_DND_MIME = "application/x-comfyresearch-user-observable";

/** Extra payload when dragging a saved linear dataset blueprint (same node type `linear_dataset`). */
export const USER_LINEAR_DATASET_DND_MIME = "application/x-comfyresearch-user-linear-dataset";

/** Extra payload when dragging a saved symbolic func dataset blueprint (same node type `symbolic_func_dataset`). */
export const USER_SYMBOLIC_FUNC_DATASET_DND_MIME = "application/x-comfyresearch-user-symbolic-func-dataset";

/** Dragging a saved combined subgraph template onto the canvas (node type `combined_model`). */
export const COMBINED_MODEL_TEMPLATE_DND_MIME = "application/x-comfyresearch-combined-model-template";

export const USER_OBSERVABLES_CHANGED = "cr-user-observables-changed";

export const USER_LINEAR_DATASETS_CHANGED = "cr-user-linear-datasets-changed";

export const USER_SYMBOLIC_FUNC_DATASETS_CHANGED = "cr-user-symbolic-func-datasets-changed";

/** Templates library on disk changed (save / delete / rename). */
export const GRAPH_TEMPLATE_LIBRARY_CHANGED = "cr-graph-template-library-changed";

/** Combined-model library entries changed (workflow-backed save/delete from Combine or Nodes panel). */
export const GRAPH_COMBINED_MODEL_LIBRARY_CHANGED = "cr-graph-combined-model-library-changed";
