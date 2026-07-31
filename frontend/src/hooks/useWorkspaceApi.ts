import { useCallback, useState } from "react";
import type { WorkspaceSnapshotDTO } from "../types/workspace";

export function useWorkspaceApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadWorkspace = useCallback(async (): Promise<WorkspaceSnapshotDTO> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/workspace");
      if (!res.ok) {
        throw new Error(await res.text());
      }
      return (await res.json()) as WorkspaceSnapshotDTO;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const saveWorkspace = useCallback(async (body: WorkspaceSnapshotDTO): Promise<WorkspaceSnapshotDTO> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/workspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      return (await res.json()) as WorkspaceSnapshotDTO;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loadWorkspace, saveWorkspace, loading, error };
}
