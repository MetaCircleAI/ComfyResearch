import type { Edge, Node } from "@xyflow/react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type MouseEvent,
} from "react";
import { DND_MIME, DND_TEXT_PLAIN, USER_OBSERVABLE_DND_MIME, USER_OBSERVABLES_CHANGED } from "../dnd";
import { serializeGraphForApi } from "../graph/serializeGraphForApi";
import { appendResearchNode } from "../graph/nodeInstanceTitle";
import { ensureTrainerAutoVizes } from "../graph/trainerAutoVizSpawn";
import { defaultObservableUserData } from "../components/nodes/observableUserDefaults";
import { useResearchGraph } from "../context/ResearchGraphContext";
import {
  autoObservableLabel,
  buildRandomObservableDrafts,
  DEFAULT_RANDOM_GENERATION_PREFERENCES,
  defaultReductionDrafts,
  formatReductionPreview,
  formatTensorShapeBracket,
  flattenOptionsForSource,
  OBSERVABLE_REDUCTION_OPTIONS,
  parseObservableRandomSeed,
  RANDOM_GENERATION_PREFERENCE_ROWS,
  reductionsForFlattenMode,
  isMatrixRepresentation,
  type AlgebraObservableItem,
  type AxisReductionDraft,
  type ObservableFlattenMode,
  type ObservableReductionOp,
  type ObservableSource,
  type ObservableTensorScope,
  type RandomGenerationPreference,
  type RandomGenerationPreferences,
  type RepresentationEntry,
  familyPatternFromTensorName,
  familyPatternFromRepresentationId,
  matchingTensorNames,
  matchingRepresentationIds,
  canUseAllMatchingScope,
  canUseAllMatchingScopeForRepresentation,
  observableScopeOptionsForFlattenMode,
} from "./observableAlgebra";
import { isObservableModelNodeType } from "./modelNodeTypes";

const RANDOM_PREF_LEVELS: RandomGenerationPreference[] = ["none", "some", "all"];

function RandomGenerationPreferenceList({
  prefs,
  onChange,
}: {
  prefs: RandomGenerationPreferences;
  onChange: (next: RandomGenerationPreferences) => void;
}) {
  return (
    <div className="cr-obs-panel__pref-list" role="group" aria-label="Random generation preferences">
      <div className="cr-obs-panel__pref-list-header">
        <span className="cr-obs-panel__pref-list-title">Preference list</span>
        <span className="cr-obs-panel__pref-list-legend">
          <span>None</span>
          <span>Some</span>
          <span>All</span>
        </span>
      </div>
      {RANDOM_GENERATION_PREFERENCE_ROWS.map(({ key, label }) => (
        <div key={key} className="cr-obs-panel__pref-row">
          <span className="cr-obs-panel__pref-row-label">{label}</span>
          <div className="cr-obs-panel__pref-segment" role="radiogroup" aria-label={label}>
            {RANDOM_PREF_LEVELS.map((level) => (
              <button
                key={level}
                type="button"
                role="radio"
                aria-checked={prefs[key] === level}
                className={`cr-obs-panel__pref-segment-btn${prefs[key] === level ? " cr-obs-panel__pref-segment-btn--active" : ""}`}
                onClick={() => onChange({ ...prefs, [key]: level })}
              >
                {level === "none" ? "None" : level === "some" ? "Some" : "All"}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

type ObservablePanelProps = {
  nodes: Node[];
  edges: Edge[];
  selectedNodeId: string | null;
};

type WeightSpec = { shape: number[] };

type AddObservableMode = "hand" | "random";

function defaultAddAllScreenPos(index: number): { x: number; y: number } {
  const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
  return { x: vw * 0.55, y: 220 + index * 56 };
}

function nodeTitle(node: Node | undefined): string {
  if (!node) return "—";
  const d = (node.data ?? {}) as { label?: string; displayName?: string };
  const label = (d.label ?? d.displayName ?? "").trim();
  if (label) return label;
  return String(node.type ?? node.id).replace(/_/g, " ");
}

function AddedObservableRow({
  item,
  graph,
  onDelete,
}: {
  item: AlgebraObservableItem;
  graph: ReturnType<typeof useResearchGraph>;
  onDelete: (e: MouseEvent, id: string) => void;
}) {
  const label = item.label.trim() || item.id.slice(0, 8);

  const onDragStart = useCallback(
    (event: DragEvent) => {
      event.dataTransfer.setData(DND_MIME, "observable_user");
      event.dataTransfer.setData(DND_TEXT_PLAIN, "observable_user");
      event.dataTransfer.setData(
        USER_OBSERVABLE_DND_MIME,
        JSON.stringify({
          userObservableId: item.id,
          label,
          tensorVizNodeId: item.tensor_viz_node_id ?? "",
          tensorSelectorNodeId: item.tensor_selector_node_id ?? "",
        }),
      );
      event.dataTransfer.effectAllowed = "move";
    },
    [item.id, item.tensor_selector_node_id, item.tensor_viz_node_id, label],
  );

  return (
    <div
      className="cr-obs-panel__added-row"
      draggable
      onDragStart={onDragStart}
      onDoubleClick={(e) => {
        if ((e.target as HTMLElement).closest("button")) return;
        graph?.addNode("observable_user", undefined, {
          userObservableId: item.id,
          label,
          tensorVizNodeId: item.tensor_viz_node_id ?? "",
          tensorSelectorNodeId: item.tensor_selector_node_id ?? "",
        });
      }}
      title="Drag onto the canvas or double-click to add"
    >
      <div className="cr-obs-panel__added-main">
        <span className="cr-obs-panel__added-label">{label}</span>
        {item.human_chain ? (
          <span className="cr-obs-panel__added-chain">{item.human_chain}</span>
        ) : null}
      </div>
      <button
        type="button"
        className="cr-nodes-panel__icon-btn cr-nodes-panel__icon-btn--trash"
        title="Delete observable"
        aria-label={`Delete ${label}`}
        onClick={(e) => onDelete(e, item.id)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
          <path
            d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

export function ObservablePanel({ nodes, edges, selectedNodeId }: ObservablePanelProps) {
  const graph = useResearchGraph();
  const selectedNode = useMemo(
    () => (selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : undefined),
    [nodes, selectedNodeId],
  );
  const modelNode = selectedNode && isObservableModelNodeType(selectedNode.type) ? selectedNode : null;

  const [specs, setSpecs] = useState<Record<string, WeightSpec>>({});
  const [repEntries, setRepEntries] = useState<RepresentationEntry[]>([]);
  const [specsSummary, setSpecsSummary] = useState<string | null>(null);
  const [repSummary, setRepSummary] = useState<string | null>(null);
  const [specsLoading, setSpecsLoading] = useState(false);
  const [specsError, setSpecsError] = useState<string | null>(null);
  const [repError, setRepError] = useState<string | null>(null);

  const [observableSource, setObservableSource] = useState<ObservableSource>("weight");
  const [selectedTensor, setSelectedTensor] = useState<string | null>(null);
  const [selectedRepId, setSelectedRepId] = useState<string | null>(null);
  const [flattenMode, setFlattenMode] = useState<ObservableFlattenMode>("none");
  const [tensorScope, setTensorScope] = useState<ObservableTensorScope>("single");
  const [reductions, setReductions] = useState<AxisReductionDraft[]>([]);
  const [nameDraft, setNameDraft] = useState("");
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [addMode, setAddMode] = useState<AddObservableMode>("hand");
  const [randomCountDraft, setRandomCountDraft] = useState("5");
  const [randomSeedDraft, setRandomSeedDraft] = useState("0");
  const [randomPrefs, setRandomPrefs] = useState<RandomGenerationPreferences>(() => ({
    ...DEFAULT_RANDOM_GENERATION_PREFERENCES,
  }));
  const [randomBusy, setRandomBusy] = useState(false);
  const [randomError, setRandomError] = useState<string | null>(null);

  const [savedItems, setSavedItems] = useState<AlgebraObservableItem[]>([]);
  const [savedLoading, setSavedLoading] = useState(false);
  const [deleteAllBusy, setDeleteAllBusy] = useState(false);

  const modelSpecsFetchKey = useMemo(() => {
    if (!modelNode) return "";
    const g = serializeGraphForApi(nodes, edges);
    return JSON.stringify({ modelId: modelNode.id, nodes: g.nodes, edges: g.edges });
  }, [modelNode, nodes, edges]);

  const specsFetchGenRef = useRef(0);
  const tensorNames = useMemo(() => Object.keys(specs).sort(), [specs]);
  const selectedRepEntry = useMemo(
    () => repEntries.find((e) => e.representation_id === selectedRepId) ?? null,
    [repEntries, selectedRepId],
  );
  const selectedSubjectId = observableSource === "weight" ? selectedTensor : selectedRepId;
  const activeShape =
    observableSource === "weight"
      ? selectedTensor
        ? (specs[selectedTensor]?.shape ?? [])
        : []
      : (selectedRepEntry?.shape ?? []);
  const subjectLabel =
    observableSource === "weight" ? (selectedTensor ?? "") : (selectedRepEntry?.label ?? selectedRepId ?? "");
  const flattenOptions = useMemo(() => flattenOptionsForSource(observableSource), [observableSource]);
  const repIds = useMemo(() => repEntries.map((e) => e.representation_id), [repEntries]);
  const scopeMatchedTensors = useMemo(() => {
    if (!selectedTensor) return [] as string[];
    return matchingTensorNames(tensorNames, selectedTensor, tensorScope);
  }, [selectedTensor, tensorNames, tensorScope]);
  const scopeMatchedReps = useMemo(() => {
    if (!selectedRepId) return [] as string[];
    return matchingRepresentationIds(repIds, selectedRepId, tensorScope);
  }, [selectedRepId, repIds, tensorScope]);
  const scopeMatchedRepLabels = useMemo(() => {
    return scopeMatchedReps
      .map((id) => repEntries.find((e) => e.representation_id === id)?.label ?? id)
      .sort();
  }, [repEntries, scopeMatchedReps]);

  const refreshSaved = useCallback(async () => {
    setSavedLoading(true);
    try {
      const res = await fetch("/api/user-observables");
      if (!res.ok) throw new Error(res.statusText);
      const j = (await res.json()) as { items?: AlgebraObservableItem[] };
      setSavedItems(j.items ?? []);
    } catch {
      setSavedItems([]);
    } finally {
      setSavedLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshSaved();
    const onChanged = () => void refreshSaved();
    window.addEventListener(USER_OBSERVABLES_CHANGED, onChanged);
    return () => window.removeEventListener(USER_OBSERVABLES_CHANGED, onChanged);
  }, [refreshSaved]);

  useEffect(() => {
    if (!modelSpecsFetchKey) {
      setSpecs({});
      setRepEntries([]);
      setSpecsSummary(null);
      setRepSummary(null);
      setSpecsError(null);
      setRepError(null);
      setSelectedTensor(null);
      setSelectedRepId(null);
      setSpecsLoading(false);
      return;
    }
    const { modelId, nodes: apiNodes, edges: apiEdges } = JSON.parse(modelSpecsFetchKey) as {
      modelId: string;
      nodes: unknown[];
      edges: unknown[];
    };
    const gen = ++specsFetchGenRef.current;
    setSpecsLoading(true);
    setSpecsError(null);
    setRepError(null);
    (async () => {
      try {
        const body = {
          nodes: apiNodes,
          edges: apiEdges,
          model_node_id: modelId,
        };
        const [weightRes, repRes] = await Promise.all([
          fetch("/api/model_weight_specs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
          fetch("/api/model_representation_specs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
        ]);
        if (!weightRes.ok) {
          let msg = weightRes.statusText;
          try {
            const j = (await weightRes.json()) as { detail?: unknown };
            if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {
            /* ignore */
          }
          throw new Error(msg);
        }
        const wj = (await weightRes.json()) as {
          specs?: Record<string, WeightSpec>;
          summary?: string;
        };
        let repList: RepresentationEntry[] = [];
        let repSum: string | null = null;
        if (repRes.ok) {
          const rj = (await repRes.json()) as { entries?: RepresentationEntry[]; summary?: string };
          repList = rj.entries ?? [];
          repSum = typeof rj.summary === "string" ? rj.summary : null;
        } else {
          let msg = repRes.statusText;
          try {
            const j = (await repRes.json()) as { detail?: unknown };
            if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {
            /* ignore */
          }
          if (gen === specsFetchGenRef.current) setRepError(msg);
        }
        if (gen !== specsFetchGenRef.current) return;
        setSpecs(wj.specs ?? {});
        setRepEntries(repList);
        setSpecsSummary(typeof wj.summary === "string" ? wj.summary : null);
        setRepSummary(repSum);
        setSelectedTensor(null);
        setSelectedRepId(null);
      } catch (e) {
        if (gen !== specsFetchGenRef.current) return;
        setSpecs({});
        setRepEntries([]);
        setSpecsSummary(null);
        setRepSummary(null);
        setSpecsError(e instanceof Error ? e.message : String(e));
      } finally {
        if (gen === specsFetchGenRef.current) setSpecsLoading(false);
      }
    })();
  }, [modelSpecsFetchKey]);

  useEffect(() => {
    if (!selectedSubjectId) {
      setReductions([]);
      setNameDraft("");
      setTensorScope("single");
      setFlattenMode("none");
      return;
    }
    const shape = activeShape;
    const drafts = defaultReductionDrafts(shape, selectedSubjectId);
    setFlattenMode("none");
    setReductions(drafts);
    setTensorScope("single");
    setNameDraft(
      autoObservableLabel(selectedSubjectId, drafts, "single", "none", observableSource, selectedRepEntry),
    );
  }, [selectedSubjectId, observableSource]);

  useEffect(() => {
    if (!selectedSubjectId || reductions.length === 0) return;
    setNameDraft(
      autoObservableLabel(
        selectedSubjectId,
        reductions,
        tensorScope,
        flattenMode,
        observableSource,
        selectedRepEntry,
      ),
    );
  }, [tensorScope, selectedSubjectId, reductions, flattenMode, observableSource, selectedRepEntry]);

  const onFlattenModeChange = useCallback(
    (mode: ObservableFlattenMode) => {
      if (!selectedSubjectId) return;
      if (mode === "sv_entropy" && !isMatrixRepresentation(activeShape)) {
        setAddError("Singular value entropy requires a matrix-shaped representation (batch × features at training time).");
        return;
      }
      setFlattenMode(mode);
      setReductions((prev) => reductionsForFlattenMode(mode, activeShape, selectedSubjectId, prev));
      if (observableSource === "weight" && mode === "global" && selectedTensor && canUseAllMatchingScope(selectedTensor, tensorNames)) {
        setTensorScope("all_matching");
      }
    },
    [activeShape, observableSource, selectedSubjectId, selectedTensor, tensorNames],
  );

  const preview = useMemo(() => {
    if (!selectedSubjectId || reductions.length === 0) return "";
    return formatReductionPreview(subjectLabel, activeShape, reductions, flattenMode);
  }, [selectedSubjectId, subjectLabel, reductions, activeShape, flattenMode]);

  const flattenHint = flattenOptions.find((o) => o.id === flattenMode)?.hint ?? "";
  const scopeOptions = useMemo(
    () => observableScopeOptionsForFlattenMode(flattenMode),
    [flattenMode],
  );

  const updateReductionOp = useCallback(
    (axisIndex: number, op: ObservableReductionOp) => {
      setReductions((prev) => {
        const next = prev.map((r) => (r.axisIndex === axisIndex ? { ...r, op } : r));
        setNameDraft(
          autoObservableLabel(
            selectedSubjectId ?? "",
            next,
            tensorScope,
            flattenMode,
            observableSource,
            selectedRepEntry,
          ),
        );
        return next;
      });
    },
    [flattenMode, observableSource, selectedRepEntry, selectedSubjectId, tensorScope],
  );

  const updateFlatOp = useCallback(
    (op: ObservableReductionOp) => {
      const next = [{ axisIndex: 0, axisLabel: "flat", op }];
      setReductions(next);
      setNameDraft(
        autoObservableLabel(
          selectedSubjectId ?? "",
          next,
          tensorScope,
          flattenMode,
          observableSource,
          selectedRepEntry,
        ),
      );
    },
    [flattenMode, observableSource, selectedRepEntry, selectedSubjectId, tensorScope],
  );

  const postAlgebraObservable = useCallback(
    async (payload: {
      label?: string;
      tensor_name: string;
      tensor_shape: number[];
      tensor_scope: ObservableTensorScope;
      flatten_mode: ObservableFlattenMode;
      observable_source: ObservableSource;
      representation_id?: string;
      layer_index?: number;
      layer_io?: string;
      reductions: AxisReductionDraft[];
    }) => {
      if (!modelNode) throw new Error("No model selected.");
      const res = await fetch("/api/user-observables/algebra", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: payload.label?.trim() || undefined,
          source_model_node_id: modelNode.id,
          tensor_name: payload.tensor_name,
          tensor_shape: payload.tensor_shape,
          tensor_scope: payload.tensor_scope,
          flatten_mode: payload.flatten_mode,
          observable_source: payload.observable_source,
          representation_id: payload.representation_id ?? "",
          layer_index: payload.layer_index ?? 0,
          layer_io: payload.layer_io ?? "",
          reductions: payload.reductions.map((r) => ({
            axis_index: r.axisIndex,
            axis_label: r.axisLabel,
            op: r.op,
          })),
        }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
    },
    [modelNode],
  );

  const onAddObservable = useCallback(async () => {
    if (!modelNode || !selectedSubjectId || reductions.length === 0) return;
    const rank = activeShape.length;
    if (flattenMode === "none" && rank > 0 && reductions.length !== rank) {
      setAddError(`Rank-${rank} tensor needs ${rank} per-axis reductions (one per dimension).`);
      return;
    }
    if (flattenMode === "local" && reductions.length !== 1) {
      setAddError("Local flatten requires exactly one 1D reduction.");
      return;
    }
    if (flattenMode === "global" && reductions.length !== 1) {
      setAddError("Global flatten requires exactly one 1D reduction.");
      return;
    }
    if (flattenMode === "sv_entropy" && !isMatrixRepresentation(activeShape)) {
      setAddError("Singular value entropy requires a matrix-shaped representation.");
      return;
    }
    setAddBusy(true);
    setAddError(null);
    try {
      await postAlgebraObservable({
        label: nameDraft,
        tensor_name: selectedSubjectId,
        tensor_shape: activeShape,
        tensor_scope: tensorScope,
        flatten_mode: flattenMode,
        observable_source: observableSource,
        representation_id: observableSource === "representation" ? selectedSubjectId : "",
        layer_index: selectedRepEntry?.layer_index,
        layer_io: selectedRepEntry?.io,
        reductions,
      });
      window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
      setAddError(null);
    } catch (e) {
      setAddError(e instanceof Error ? e.message : String(e));
    } finally {
      setAddBusy(false);
    }
  }, [
    activeShape,
    flattenMode,
    modelNode,
    nameDraft,
    observableSource,
    postAlgebraObservable,
    reductions,
    selectedRepEntry,
    selectedSubjectId,
    tensorScope,
  ]);

  const onGenerateRandomObservables = useCallback(async () => {
    if (!modelNode || (tensorNames.length === 0 && repEntries.length === 0)) return;
    const count = Math.floor(Number.parseFloat(randomCountDraft.trim()));
    if (!Number.isFinite(count) || count < 1) {
      setRandomError("Enter a positive integer count.");
      return;
    }
    if (count > 200) {
      setRandomError("Maximum 200 random observables per batch.");
      return;
    }
    const seedParsed = parseObservableRandomSeed(randomSeedDraft);
    if (!seedParsed.ok) {
      setRandomError(seedParsed.error);
      return;
    }
    setRandomBusy(true);
    setRandomError(null);
    try {
      const drafts = buildRandomObservableDrafts(
        count,
        seedParsed.seed,
        tensorNames,
        specs,
        repEntries,
        { preferences: randomPrefs },
      );
      for (const draft of drafts) {
        await postAlgebraObservable({
          label: draft.label,
          tensor_name: draft.tensorName,
          tensor_shape: draft.tensorShape,
          tensor_scope: draft.tensorScope,
          flatten_mode: draft.flattenMode,
          observable_source: draft.observableSource,
          representation_id: draft.representationId ?? "",
          layer_index: draft.layerIndex,
          layer_io: draft.layerIo,
          reductions: draft.reductions,
        });
      }
      window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
    } catch (e) {
      setRandomError(e instanceof Error ? e.message : String(e));
    } finally {
      setRandomBusy(false);
    }
  }, [modelNode, postAlgebraObservable, randomCountDraft, randomPrefs, randomSeedDraft, repEntries, specs, tensorNames]);

  const onAddAllToExperiment = useCallback(() => {
    if (!graph || savedItems.length === 0) return;
    const trainer = nodes.find((n) => n.type === "trainer" || n.type === "crl_trainer");
    if (!graph.setFlowNodes || !graph.setFlowEdges || !trainer) {
      savedItems.forEach((item, index) => {
        const label = item.label.trim() || item.id.slice(0, 8);
        graph.addNode("observable_user", defaultAddAllScreenPos(index), {
          userObservableId: item.id,
          label,
          tensorVizNodeId: item.tensor_viz_node_id ?? "",
          tensorSelectorNodeId: item.tensor_selector_node_id ?? "",
        });
      });
      return;
    }

    let outNodes = [...nodes];
    let outEdges = [...edges];
    const trainerId = trainer.id;

    savedItems.forEach((item, index) => {
      const label = item.label.trim() || item.id.slice(0, 8);
      let obsNode = outNodes.find((n) => {
        if (n.type !== "observable_user") return false;
        const d = (n.data ?? {}) as { userObservableId?: string };
        return (d.userObservableId ?? "").trim() === item.id;
      });
      if (!obsNode) {
        obsNode = appendResearchNode(
          outNodes,
          "observable_user",
          defaultAddAllScreenPos(index),
          defaultObservableUserData({
            userObservableId: item.id,
            label,
            tensorVizNodeId: item.tensor_viz_node_id ?? "",
            tensorSelectorNodeId: item.tensor_selector_node_id ?? "",
          }) as Record<string, unknown>,
        );
        outNodes = [...outNodes, obsNode];
      }
      const alreadyWired = outEdges.some(
        (e) =>
          e.source === obsNode!.id &&
          e.target === trainerId &&
          (e.targetHandle ?? "") === "observables",
      );
      if (!alreadyWired) {
        outEdges = [
          ...outEdges,
          {
            id: `e-obs-all-${obsNode.id}-${trainerId}`,
            source: obsNode.id,
            target: trainerId,
            sourceHandle: "observables",
            targetHandle: "observables",
            type: "research_default",
          },
        ];
      }
    });

    const fin = ensureTrainerAutoVizes(outNodes, outEdges, trainerId);
    graph.setFlowNodes(fin.nodes);
    graph.setFlowEdges(fin.edges);
  }, [edges, graph, nodes, savedItems]);

  const onDeleteAllSaved = useCallback(async () => {
    if (savedItems.length === 0) return;
    if (
      !window.confirm(
        `Delete all ${savedItems.length} saved observable${savedItems.length === 1 ? "" : "s"}? This cannot be undone.`,
      )
    ) {
      return;
    }
    setDeleteAllBusy(true);
    try {
      for (const item of savedItems) {
        await fetch(`/api/user-observables/${encodeURIComponent(item.id)}`, { method: "DELETE" });
      }
      window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
    } catch {
      /* ignore */
    } finally {
      setDeleteAllBusy(false);
    }
  }, [savedItems]);

  const onDeleteSaved = useCallback(async (e: MouseEvent, itemId: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      const r = await fetch(`/api/user-observables/${encodeURIComponent(itemId)}`, { method: "DELETE" });
      if (r.ok) {
        window.dispatchEvent(
          new CustomEvent(USER_OBSERVABLES_CHANGED, { detail: { deletedUserObservableId: itemId } }),
        );
      }
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <div className="cr-nodes-panel cr-obs-panel">
      <header className="cr-nodes-panel__header">
        <h2 className="cr-nodes-panel__title">Observables</h2>
        <p className="cr-obs-panel__subtitle">
          Observable algebra on model weights or layer representations — pick a tensor, flatten mode, and reductions.
        </p>
      </header>

      <div className="cr-nodes-panel__scroll cr-obs-panel__scroll">
        <section className="cr-obs-panel__section">
          <h3 className="cr-obs-panel__section-title">Selected model</h3>
          {modelNode ? (
            <>
              <p className="cr-obs-panel__model-line">
                <span className="cr-obs-panel__model-name">{nodeTitle(modelNode)}</span>
                <span className="cr-obs-panel__model-type">{String(modelNode.type ?? "").replace(/_/g, " ")}</span>
              </p>
              {specsLoading ? <p className="cr-obs-panel__hint">Loading tensors and representations…</p> : null}
              {specsError ? <p className="cr-obs-panel__error">{specsError}</p> : null}
              {repError ? <p className="cr-obs-panel__error">{repError}</p> : null}
              {specsSummary ? <p className="cr-obs-panel__hint">{specsSummary}</p> : null}
              {repSummary ? <p className="cr-obs-panel__hint">{repSummary}</p> : null}
            </>
          ) : (
            <p className="cr-obs-panel__hint">Select a model node on the canvas to list its weight tensors.</p>
          )}
        </section>

        {modelNode && (tensorNames.length > 0 || repEntries.length > 0) ? (
          <section className="cr-obs-panel__section">
            <div className="cr-obs-panel__mode-toggle" role="group" aria-label="Observable source">
              <button
                type="button"
                className={`cr-obs-panel__mode-btn${observableSource === "weight" ? " cr-obs-panel__mode-btn--active" : ""}`}
                aria-pressed={observableSource === "weight"}
                disabled={tensorNames.length === 0}
                onClick={() => {
                  setObservableSource("weight");
                  setSelectedRepId(null);
                }}
              >
                Weights
              </button>
              <button
                type="button"
                className={`cr-obs-panel__mode-btn${observableSource === "representation" ? " cr-obs-panel__mode-btn--active" : ""}`}
                aria-pressed={observableSource === "representation"}
                disabled={repEntries.length === 0}
                onClick={() => {
                  setObservableSource("representation");
                  setSelectedTensor(null);
                }}
              >
                Representations
              </button>
            </div>
            <h3 className="cr-obs-panel__section-title">
              {observableSource === "weight" ? "Weight tensors" : "Layer representations"}
              <span className="cr-obs-panel__section-count">
                {observableSource === "weight" ? tensorNames.length : repEntries.length}
              </span>
            </h3>
            <div className="cr-obs-panel__tensor-scroll-host" tabIndex={0} aria-label="Scrollable tensor list">
              <ul className="cr-obs-panel__tensor-list">
                {observableSource === "weight"
                  ? tensorNames.map((name) => {
                      const shape = specs[name]?.shape ?? [];
                      const active = selectedTensor === name;
                      return (
                        <li key={name}>
                          <button
                            type="button"
                            className={`cr-obs-panel__tensor-btn${active ? " cr-obs-panel__tensor-btn--active" : ""}`}
                            onClick={() => setSelectedTensor(name)}
                          >
                            <span className="cr-obs-panel__tensor-name">{name}</span>
                            <span className="cr-obs-panel__tensor-shape">[{shape.join(", ") || "?"}]</span>
                          </button>
                        </li>
                      );
                    })
                  : repEntries.map((entry) => {
                      const active = selectedRepId === entry.representation_id;
                      return (
                        <li key={entry.representation_id}>
                          <button
                            type="button"
                            className={`cr-obs-panel__tensor-btn${active ? " cr-obs-panel__tensor-btn--active" : ""}`}
                            onClick={() => setSelectedRepId(entry.representation_id)}
                          >
                            <span className="cr-obs-panel__tensor-name">{entry.label}</span>
                            <span className="cr-obs-panel__tensor-shape">{formatTensorShapeBracket(entry.shape)}</span>
                          </button>
                        </li>
                      );
                    })}
              </ul>
            </div>
            {(observableSource === "weight" ? tensorNames.length : repEntries.length) > 3 ? (
              <p className="cr-obs-panel__scroll-hint">Scroll inside the box to see all entries.</p>
            ) : null}
            <div className="cr-obs-panel__add-mode">
              <div className="cr-obs-panel__mode-toggle" role="group" aria-label="Add observable mode">
                <button
                  type="button"
                  className={`cr-obs-panel__mode-btn${addMode === "hand" ? " cr-obs-panel__mode-btn--active" : ""}`}
                  aria-pressed={addMode === "hand"}
                  onClick={() => setAddMode("hand")}
                >
                  Add variables by hand
                </button>
                <button
                  type="button"
                  className={`cr-obs-panel__mode-btn${addMode === "random" ? " cr-obs-panel__mode-btn--active" : ""}`}
                  aria-pressed={addMode === "random"}
                  onClick={() => setAddMode("random")}
                >
                  Random generation
                </button>
              </div>
              {addMode === "random" ? (
                <div className="cr-obs-panel__random-block">
                  <div className="cr-obs-panel__random-fields-row">
                    <label className="cr-obs-panel__random-count-field">
                      <span>Count</span>
                      <input
                        type="number"
                        min={1}
                        max={200}
                        step={1}
                        className="cr-obs-panel__name-input cr-obs-panel__random-count-input"
                        value={randomCountDraft}
                        onChange={(e) => setRandomCountDraft(e.target.value)}
                        aria-label="Number of random observables to generate"
                      />
                    </label>
                    <label className="cr-obs-panel__random-count-field">
                      <span>Seed</span>
                      <input
                        type="number"
                        step={1}
                        className="cr-obs-panel__name-input cr-obs-panel__random-count-input"
                        value={randomSeedDraft}
                        onChange={(e) => setRandomSeedDraft(e.target.value)}
                        aria-label="Random seed for reproducible observable generation"
                        title="Same model, count, and seed produce the same observable batch."
                      />
                    </label>
                  </div>
                  <RandomGenerationPreferenceList prefs={randomPrefs} onChange={setRandomPrefs} />
                  <p className="cr-obs-panel__hint">
                    Each draw randomly picks weights or layer representations and flatten mode. Per row:{" "}
                    <strong>None</strong> never wins when other options compete, <strong>Some</strong> may be chosen,
                    <strong> All</strong> always wins when competing. Fixed <strong>count + seed</strong> reproduces
                    the batch.
                  </p>
                  <button
                    type="button"
                    className="cr-obs-panel__add-btn"
                    disabled={randomBusy}
                    onClick={() => void onGenerateRandomObservables()}
                  >
                    {randomBusy ? "Generating…" : "Generate random observables"}
                  </button>
                  {randomError ? <p className="cr-obs-panel__error">{randomError}</p> : null}
                </div>
              ) : (
                <p className="cr-obs-panel__hint">
                  Select an entry above, choose flatten mode, then configure reductions below.
                </p>
              )}
            </div>
          </section>
        ) : null}

        {addMode === "hand" && selectedSubjectId && reductions.length > 0 ? (
          <section className="cr-obs-panel__section">
            <label className="cr-obs-panel__scope-field">
              <span>Flatten mode</span>
              <select
                className="cr-obs-panel__select"
                value={flattenMode}
                onChange={(e) => onFlattenModeChange(e.target.value as ObservableFlattenMode)}
                aria-label="Observable flatten mode"
              >
                {flattenOptions.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
            {flattenHint ? <p className="cr-obs-panel__hint">{flattenHint}</p> : null}
            {flattenMode === "none" ? (
              <>
                <h3 className="cr-obs-panel__section-title">Per-axis reductions</h3>
                <div className="cr-obs-panel__reduction-grid">
                  {reductions.map((r) => (
                    <label key={r.axisIndex} className="cr-obs-panel__reduction-row">
                      <span className="cr-obs-panel__axis-label">{r.axisLabel}</span>
                      <select
                        className="cr-obs-panel__select"
                        value={r.op}
                        onChange={(e) => updateReductionOp(r.axisIndex, e.target.value as ObservableReductionOp)}
                        aria-label={`Reduction for ${r.axisLabel}`}
                      >
                        {OBSERVABLE_REDUCTION_OPTIONS.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
              </>
            ) : flattenMode === "sv_entropy" ? (
              <p className="cr-obs-panel__hint">
                Applies SVD on the activation matrix (batch × features) and returns entropy of normalized singular values.
              </p>
            ) : (
              <>
                <h3 className="cr-obs-panel__section-title">1D reduction</h3>
                <label className="cr-obs-panel__reduction-row">
                  <span className="cr-obs-panel__axis-label">flat</span>
                  <select
                    className="cr-obs-panel__select"
                    value={reductions[0]?.op ?? "l2_norm"}
                    onChange={(e) => updateFlatOp(e.target.value as ObservableReductionOp)}
                    aria-label="1D reduction after flatten"
                  >
                    {OBSERVABLE_REDUCTION_OPTIONS.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}
            {preview ? <p className="cr-obs-panel__preview">{preview}</p> : null}
            <label className="cr-obs-panel__scope-field">
              <span>Scope</span>
              <select
                className="cr-obs-panel__select"
                value={tensorScope}
                onChange={(e) => setTensorScope(e.target.value as ObservableTensorScope)}
                aria-label="Observable tensor scope"
              >
                {scopeOptions.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
            {tensorScope === "all_matching" && observableSource === "weight" && selectedTensor ? (
              flattenMode === "global" ? (
                <p className="cr-obs-panel__hint">
                  One global scalar over all matching weights
                  {scopeMatchedTensors.length > 0 ? ` (${scopeMatchedTensors.length} tensors concatenated)` : ""}
                </p>
              ) : (
                <p className="cr-obs-panel__hint">
                  Pattern{" "}
                  <code className="cr-obs-panel__inline-code">{familyPatternFromTensorName(selectedTensor)}</code>
                  {scopeMatchedTensors.length > 0
                    ? ` · ${scopeMatchedTensors.length} tensors: ${scopeMatchedTensors.join(", ")}`
                    : " · no matching tensors found"}
                </p>
              )
            ) : null}
            {tensorScope === "all_matching" && observableSource === "representation" && selectedRepId ? (
              flattenMode === "global" ? (
                <p className="cr-obs-panel__hint">
                  One forward pass · one global scalar over all matching representations
                  {scopeMatchedReps.length > 0 ? ` (${scopeMatchedReps.length} tensors concatenated)` : ""}
                </p>
              ) : (
                <p className="cr-obs-panel__hint">
                  Pattern{" "}
                  <code className="cr-obs-panel__inline-code">{familyPatternFromRepresentationId(selectedRepId)}</code>
                  {scopeMatchedReps.length > 0
                    ? ` · ${scopeMatchedReps.length} representations: ${scopeMatchedRepLabels.join(", ")}`
                    : " · no matching representations found"}
                </p>
              )
            ) : null}
            <label className="cr-obs-panel__name-field">
              <span>Name</span>
              <input
                type="text"
                className="cr-obs-panel__name-input"
                value={nameDraft}
                onChange={(e) => setNameDraft(e.target.value)}
                placeholder="Observable name"
              />
            </label>
            <button
              type="button"
              className="cr-obs-panel__add-btn"
              disabled={addBusy || !selectedSubjectId}
              onClick={() => void onAddObservable()}
            >
              {addBusy ? "Adding…" : "Add observable"}
            </button>
            {addError ? <p className="cr-obs-panel__error">{addError}</p> : null}
          </section>
        ) : null}

        <section className="cr-obs-panel__section cr-obs-panel__section--added-list">
          <h3 className="cr-obs-panel__section-title">Added observables</h3>
          {savedLoading ? <p className="cr-obs-panel__hint">Loading…</p> : null}
          {!savedLoading && savedItems.length === 0 ? (
            <p className="cr-obs-panel__hint">No saved observables yet.</p>
          ) : null}
          {savedItems.length > 0 ? (
            <div className="cr-obs-panel__added-actions">
              <button
                type="button"
                className="cr-obs-panel__add-btn cr-obs-panel__added-action-btn"
                onClick={onAddAllToExperiment}
              >
                Add all observables to experiment
              </button>
              <button
                type="button"
                className="cr-obs-panel__delete-all-btn cr-obs-panel__added-action-btn"
                disabled={deleteAllBusy}
                onClick={() => void onDeleteAllSaved()}
              >
                {deleteAllBusy ? "Deleting…" : "Delete all observables"}
              </button>
            </div>
          ) : null}
          <div className="cr-obs-panel__added-list">
            {savedItems.map((item) => (
              <AddedObservableRow key={item.id} item={item} graph={graph} onDelete={onDeleteSaved} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
