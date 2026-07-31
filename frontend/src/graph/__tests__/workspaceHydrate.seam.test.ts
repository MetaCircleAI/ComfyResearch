import { describe, expect, it } from "vitest";
import { mergeWorkspaceHydrateWithLocalProjects } from "../workspaceHydrate";

describe("workspace hydrate", () => {
  it("drops the untouched initial placeholder when the persisted workspace arrives", () => {
    const persisted = { id: "persisted" };
    const initialPlaceholder = { id: "initial-placeholder" };

    expect(
      mergeWorkspaceHydrateWithLocalProjects(
        [persisted],
        [initialPlaceholder],
        initialPlaceholder,
      ),
    ).toEqual([persisted]);
  });

  it("keeps a project opened locally while hydrate is in flight", () => {
    const persisted = { id: "persisted" };
    const initialPlaceholder = { id: "initial-placeholder" };
    const locallyOpened = { id: "locally-opened" };

    expect(
      mergeWorkspaceHydrateWithLocalProjects(
        [persisted],
        [initialPlaceholder, locallyOpened],
        initialPlaceholder,
      ),
    ).toEqual([persisted, locallyOpened]);
  });

  it("drops a structurally unchanged initial placeholder after an immutable update", () => {
    const persisted = { id: "persisted" };
    const initialPlaceholder = { id: "initial-placeholder", title: "Project" };
    const clonedPlaceholder = { ...initialPlaceholder };

    expect(
      mergeWorkspaceHydrateWithLocalProjects(
        [persisted],
        [clonedPlaceholder],
        initialPlaceholder,
      ),
    ).toEqual([persisted]);
  });

  it("keeps the initial project when it was edited before hydrate completed", () => {
    const persisted = { id: "persisted", canvas: { nodes: [] as string[] } };
    const initialPlaceholder = {
      id: "initial-placeholder",
      canvas: { nodes: [] as string[] },
    };
    const editedInitialProject = {
      ...initialPlaceholder,
      canvas: { nodes: ["imported-node"] },
    };

    expect(
      mergeWorkspaceHydrateWithLocalProjects(
        [persisted],
        [editedInitialProject],
        initialPlaceholder,
      ),
    ).toEqual([persisted, editedInitialProject]);
  });
});
