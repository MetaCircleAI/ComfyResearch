import type { GraphDocument } from "./graph";

/** Matches GET/POST /api/workspace (snake_case JSON). */

export type WorkspaceCanvasDTO = {
  id: string;
  title: string;
  document: GraphDocument;
};

/** A project owns exactly one canvas; use a separate project for a controlled variant. */
export type WorkspaceProjectDTO = {
  id: string;
  title: string;
  canvas: WorkspaceCanvasDTO;
};

export type WorkspaceSnapshotDTO = {
  version: 3;
  active_project_id: string;
  projects: WorkspaceProjectDTO[];
};
