/** Hook points match `_build_mlp` in trainer_run: depth × (Linear → preact, Activation → postact), then output Linear. */

export type ActivationRepresentation = { id: string; label: string };

/** Row-major flattened float values; length equals product of `shape`. */
export type CollectedActivationTensor = {
  shape: number[];
  values: number[];
};

/** Shapes only — values live on the server until a downstream node fetches them. */
export type ActivationManifest = Record<string, { shape: number[] }>;

/** Read-only wire picker: tensor after ``nn.Sequential[afterModuleIndex]`` (non-mutating vs main canvas). */
export type ActivationWirePick = {
  pickId: string;
  /** Stable manifest / tensor-list key (sanitized). */
  tensorKey: string;
  label: string;
  afterModuleIndex: number;
  /** Reserved for loop-count subgraphs; server currently treats all as global hooks. */
  loopScope?: "all" | "instance";
};

export type ActivationNodeData = {
  representationOptions: ActivationRepresentation[];
  /** Checked = collected on “Collect” (one forward pass on the train dataset). */
  selectedRepresentationIds: string[];
  /** When non-empty, Collect uses wire hooks instead of checkbox representations. */
  activationWirePicks: ActivationWirePick[];
  scanMessage: string | null;
  /** Legacy: full tensors in graph state (large). Prefer `activationRunId` + `activationManifest`. */
  collectedActivations: Record<string, CollectedActivationTensor> | null;
  collectSummary: string | null;
  /** Server-side activation cache id after Collect. */
  activationRunId: string | null;
  activationManifest: ActivationManifest | null;
};

export function defaultActivationData(): ActivationNodeData {
  return {
    representationOptions: [],
    selectedRepresentationIds: [],
    activationWirePicks: [],
    scanMessage: null,
    collectedActivations: null,
    collectSummary: null,
    activationRunId: null,
    activationManifest: null,
  };
}

export function buildRepresentationsForMlpDepth(depth: number): ActivationRepresentation[] {
  if (!Number.isFinite(depth) || depth < 1) {
    return [{ id: "input", label: "input" }, { id: "output", label: "output" }];
  }
  const items: ActivationRepresentation[] = [{ id: "input", label: "input" }];
  for (let i = 1; i <= depth; i++) {
    items.push({ id: `h${i}_preact`, label: `layer ${i} preact` });
    items.push({ id: `h${i}_postact`, label: `layer ${i} postact` });
  }
  items.push({ id: "output", label: "output" });
  return items;
}

export function buildRepresentationsForResidualDepth(depth: number): ActivationRepresentation[] {
  if (!Number.isFinite(depth) || depth < 1) {
    return [
      { id: "input", label: "input" },
      { id: "h0", label: "h0 (residual stream)" },
      { id: "output", label: "output" },
    ];
  }
  const items: ActivationRepresentation[] = [
    { id: "input", label: "input" },
    { id: "h0", label: "h0 (residual stream)" },
  ];
  for (let i = 1; i <= depth; i++) {
    items.push({ id: `h${i}`, label: `h${i} (residual stream)` });
  }
  items.push({ id: "output", label: "output" });
  return items;
}
