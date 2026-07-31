/**
 * Merge projects opened locally while the persisted workspace request was in flight.
 */
export function mergeWorkspaceHydrateWithLocalProjects<T extends { id: string }>(
  serverProjects: T[],
  currentProjects: T[],
  initialPlaceholderProject: T,
): T[] {
  const serverIds = new Set(serverProjects.map((project) => project.id));
  const initialPlaceholderSnapshot = JSON.stringify(initialPlaceholderProject);
  const extras = currentProjects.filter(
    (project) =>
      !serverIds.has(project.id) &&
      (project.id !== initialPlaceholderProject.id ||
        JSON.stringify(project) !== initialPlaceholderSnapshot),
  );
  return extras.length ? [...serverProjects, ...extras] : serverProjects;
}
