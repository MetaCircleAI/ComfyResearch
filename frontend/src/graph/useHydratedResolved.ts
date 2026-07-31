import { useEffect, useState } from "react";
import { fetchActivationTensorAsOk, hydrateResolved } from "./fetchActivationTensor";
import type { Resolved, ResolvedLazyActivation, ResolvedNone, ResolvedOk } from "./resolveUpstreamTensor";

/**
 * For lazy server-backed tensors, fetches float32 data from the server; otherwise passes through.
 * @param refetchKey Increment (e.g. after a "Refresh" click) to re-fetch lazy tensors without changing the graph.
 */
export function useHydratedResolved(resolved: Resolved, refetchKey = 0): {
  display: ResolvedOk | ResolvedNone;
  loading: boolean;
} {
  const [ok, setOk] = useState<ResolvedOk | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (resolved.kind !== "lazy_activation" && resolved.kind !== "lazy_dataset") {
      setOk(null);
      setErr(null);
      return;
    }
    let cancelled = false;
    setOk(null);
    setErr(null);
    const fetchPromise =
      resolved.kind === "lazy_activation"
        ? fetchActivationTensorAsOk(resolved as ResolvedLazyActivation)
        : hydrateResolved(resolved);
    fetchPromise.then((r) => {
      if (cancelled) return;
      if (r.kind === "ok") {
        setOk(r);
      } else {
        setErr(r.detail);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [resolved, refetchKey]);

  if (resolved.kind === "none") {
    return { display: resolved, loading: false };
  }
  if (resolved.kind === "ok") {
    return { display: resolved, loading: false };
  }
  if (ok) {
    return { display: ok, loading: false };
  }
  if (err) {
    return { display: { kind: "none", detail: err }, loading: false };
  }
  return {
    display: { kind: "none", detail: "Loading tensor from server…" },
    loading: true,
  };
}
