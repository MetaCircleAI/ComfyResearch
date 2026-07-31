import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { useReactFlow } from "@xyflow/react";
import { createPortal } from "react-dom";
import { NodeSpecParameterModal } from "./NodeSpecParameterModal";
import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import { renderNodeInfoTextWithLinks } from "./nodeInfoInlineLinks";
import { openNodeInformation } from "../nodeInformationEvents";

const ATOMIC_MODEL_NODE_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
]);

type NodeSpecCodeFooterProps = {
  nodeId?: string;
  graphNodeType?: string;
  generatedCode: string;
  showTopIcons?: boolean;
  infoTitle?: string;
  infoText?: string;
};

function ModelNodeInfoModal({
  open,
  title,
  text,
  storageKey,
  onClose,
}: {
  open: boolean;
  title: string;
  text: string;
  storageKey: string;
  onClose: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"write" | "preview">("preview");
  const [draftText, setDraftText] = useState(text);
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    setActiveTab("preview");
    if (typeof window === "undefined") {
      setDraftText(text);
      return;
    }
    const stored = window.localStorage.getItem(storageKey);
    setDraftText(stored ?? text);
  }, [open, storageKey, text]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return createPortal(
    <div
      className="cr-modal-backdrop cr-dataset-info-modal-backdrop"
      style={{ zIndex: 10040 }}
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="cr-modal cr-dataset-info-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="cr-modal__title">
          {title}
        </h2>
        <div className="cr-dataset-info-modal__tabs" role="tablist" aria-label="Model description editor mode">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "write"}
            className={`cr-dataset-info-modal__tab${activeTab === "write" ? " is-active" : ""}`}
            onClick={() => setActiveTab("write")}
          >
            Write
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "preview"}
            className={`cr-dataset-info-modal__tab${activeTab === "preview" ? " is-active" : ""}`}
            onClick={() => setActiveTab("preview")}
          >
            Preview
          </button>
        </div>
        {activeTab === "write" ? (
          <div className="cr-dataset-info-modal__editor-wrap">
            <textarea
              className="cr-dataset-info-modal__editor"
              value={draftText}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setDraftText(e.target.value)}
              spellCheck={false}
              aria-label={`Edit ${title} description`}
            />
            <div className="cr-dataset-info-modal__editor-hint">
              Preview renders markdown links as <code>[label](https://…)</code>; line breaks preserved.
            </div>
          </div>
        ) : (
          <div className="cr-dataset-info-modal__body">
            {draftText
              .trim()
              .split(/\n\n+/)
              .filter(Boolean)
              .map((block, idx) => (
                <p key={idx} className="cr-dataset-info-modal__p">
                  {renderNodeInfoTextWithLinks(block, `model-info-${idx}`)}
                </p>
              ))}
          </div>
        )}
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="cr-modal__btn cr-modal__btn--primary"
            onClick={() => {
              if (typeof window !== "undefined") {
                window.localStorage.setItem(storageKey, draftText);
              }
              onClose();
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export function NodeSpecHeaderActions({
  nodeId,
  graphNodeType,
  generatedCode,
  infoTitle,
  infoText,
  codeKind = "model",
}: Pick<NodeSpecCodeFooterProps, "nodeId" | "graphNodeType" | "generatedCode" | "infoTitle" | "infoText"> & {
  codeKind?: "model" | "optimizer" | "observable" | "dataset";
}) {
  const { getNodes } = useReactFlow();
  const resolvedGraphNodeType =
    graphNodeType ??
    (nodeId ? (getNodes().find((n) => n.id === nodeId)?.type as string | undefined) : undefined);
  const canOpenInfo = Boolean(infoText?.trim());
  const infoStorageKey =
    codeKind === "optimizer"
      ? `cr.optimizerInfoMarkdown.v2.${resolvedGraphNodeType ?? "unknown"}`
      : codeKind === "observable"
        ? `cr.observableInfoMarkdown.v1.${resolvedGraphNodeType ?? "unknown"}`
        : codeKind === "dataset"
          ? `cr.datasetInfoMarkdown.v1.${resolvedGraphNodeType ?? "unknown"}`
          : `cr.modelInfoMarkdown.v2.${resolvedGraphNodeType ?? "unknown"}`;
  const infoLabel =
    codeKind === "optimizer"
      ? "Optimizer information"
      : codeKind === "observable"
        ? "Observable information"
        : "Model information";

  return (
    <>
      <div className="cr-node-spec-code__top-icons">
        {canOpenInfo ? (
          <button
            type="button"
            className="cr-node-spec-code__icon-btn nodrag nopan"
            aria-label={infoTitle ? `About ${infoTitle}` : infoLabel}
            title={infoLabel}
            onClick={(e: ReactMouseEvent) => {
              e.stopPropagation();
              e.preventDefault();
              if (!nodeId) return;
              openNodeInformation({
                nodeId,
                title: infoTitle ?? infoLabel,
                text: infoText ?? "",
                code: generatedCode,
                mode: "parameters",
              });
            }}
            onPointerDown={(e: ReactPointerEvent<HTMLButtonElement>) => e.stopPropagation()}
          >
            <svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="M10 18.5a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17Z" stroke="currentColor" strokeWidth="1.4" />
              <path d="M10 9.2V14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              <circle cx="10" cy="6.3" r="0.9" fill="currentColor" />
            </svg>
          </button>
        ) : null}
      </div>
      <ModelNodeInfoModal
        open={false}
        title={infoTitle ?? infoLabel}
        text={infoText ?? ""}
        storageKey={infoStorageKey}
        onClose={() => {}}
      />
    </>
  );
}

export function NodeSpecCodeFooter({
  nodeId,
  generatedCode,
}: NodeSpecCodeFooterProps) {
  const { getNodes, getEdges, setNodes } = useReactFlow();
  const [parametersOpen, setParametersOpen] = useState(false);
  const [parameterTensors, setParameterTensors] = useState<Record<string, WeightTensorPayload>>({});
  const [loadingParameters, setLoadingParameters] = useState(false);
  const [parameterErr, setParameterErr] = useState<string | null>(null);
  const [loadedSignature, setLoadedSignature] = useState<string | null>(null);
  const loadingInFlightRef = useRef(false);
  const canOpenParameters = !!nodeId;

  const readSavedParameterState = useCallback((): {
    tensors: Record<string, WeightTensorPayload>;
    signature: string | null;
  } => {
    if (!nodeId) return { tensors: {}, signature: null };
    const me = getNodes().find((n) => n.id === nodeId);
    const data = (me?.data as Record<string, unknown> | undefined) ?? {};
    return {
      tensors: (data.parameterTensorPayloads ?? {}) as Record<string, WeightTensorPayload>,
      signature:
        typeof data.parameterTensorSignature === "string" && data.parameterTensorSignature.length > 0
          ? data.parameterTensorSignature
          : null,
    };
  }, [getNodes, nodeId]);

  const patchNodeParameters = useCallback(
    (next: Record<string, WeightTensorPayload>, signature: string | null = null) => {
      if (!nodeId) return;
      setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== nodeId) return n;
          const prevData = (n.data ?? {}) as Record<string, unknown>;
          return {
            ...n,
            data: {
              ...prevData,
              parameterTensorPayloads: next,
              parameterTensorSignature: signature ?? generatedCode,
            },
          };
        }),
      );
    },
    [generatedCode, nodeId, setNodes],
  );

  const loadParameters = useCallback(async () => {
    if (!nodeId) return;
    if (loadingInFlightRef.current) return;
    loadingInFlightRef.current = true;
    setLoadingParameters(true);
    setParameterErr(null);
    try {
      const graph = serializeGraphForApi(getNodes(), getEdges());
      const apiNodes = graph.nodes as { id: string; type?: string }[];
      const apiEdges = graph.edges as { id: string; source: string; target: string }[];
      const sourceNode = apiNodes.find((n) => n.id === nodeId);
      const isAtomicNode = sourceNode?.type ? ATOMIC_MODEL_NODE_TYPES.has(sourceNode.type) : false;
      const graphEdges = isAtomicNode ? apiEdges.filter((e) => e.target !== nodeId) : apiEdges;
      const syntheticWeightNodeId = `__params_weight__${nodeId}`;
      const res = await fetch("/api/collect_model_weights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes: [...apiNodes, { id: syntheticWeightNodeId, type: "model_weight_tensors", data: {} }],
          edges: [
            ...graphEdges,
            {
              id: `${nodeId}__to__${syntheticWeightNodeId}`,
              source: nodeId,
              target: syntheticWeightNodeId,
              targetHandle: "model",
            },
          ],
          model_weight_tensors_node_id: syntheticWeightNodeId,
          include_upstream_chain: false,
        }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const j = (await res.json()) as { weights?: Record<string, WeightTensorPayload> };
      const weights = j.weights ?? {};
      const saved = readSavedParameterState();
      const merged: Record<string, WeightTensorPayload> = { ...weights };
      const canApplySaved = saved.signature != null && saved.signature === generatedCode;
      for (const [name, savedTensor] of Object.entries(canApplySaved ? saved.tensors : {})) {
        const fresh = weights[name];
        if (!fresh) continue;
        const sameShape =
          fresh.shape.length === savedTensor.shape.length &&
          fresh.shape.every((d, i) => d === savedTensor.shape[i]);
        const expected = fresh.shape.length ? fresh.shape.reduce((acc, n) => acc * n, 1) : 1;
        if (!sameShape || savedTensor.values.length !== expected) continue;
        merged[name] = { shape: [...fresh.shape], values: [...savedTensor.values] };
      }
      setParameterTensors(merged);
      patchNodeParameters(merged, generatedCode);
    } catch (e) {
      setParameterErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingParameters(false);
      loadingInFlightRef.current = false;
    }
  }, [generatedCode, getEdges, getNodes, nodeId, patchNodeParameters, readSavedParameterState]);

  useEffect(() => {
    if (!nodeId) return;
    const me = getNodes().find((n) => n.id === nodeId);
    const data = (me?.data as Record<string, unknown> | undefined) ?? {};
    const saved = (data.parameterTensorPayloads ?? {}) as Record<string, WeightTensorPayload>;
    const sig = typeof data.parameterTensorSignature === "string" ? data.parameterTensorSignature : null;
    if (saved && Object.keys(saved).length > 0 && sig === generatedCode) {
      setParameterTensors(saved);
      return;
    }
    setParameterTensors({});
  }, [generatedCode, getNodes, nodeId]);

  useEffect(() => {
    if (!parametersOpen) return;
    if (loadedSignature === generatedCode) return;
    setLoadedSignature(generatedCode);
    void loadParameters();
  }, [generatedCode, loadParameters, loadedSignature, parametersOpen]);

  useEffect(() => {
    if (parametersOpen) return;
    setLoadedSignature(null);
  }, [parametersOpen]);

  return (
    <div className="cr-node-spec-code nodrag nopan cr-node-spec-code--model">
      <div className="cr-node-spec-code__actions">
        <button
          type="button"
          className="cr-node-spec-code__btn"
          onClick={() => setParametersOpen(true)}
          disabled={!canOpenParameters}
          title={!canOpenParameters ? "Missing node id for parameter view." : undefined}
        >
          View/edit parameters
        </button>
      </div>
      <NodeSpecParameterModal
        open={parametersOpen}
        tensors={parameterTensors}
        loading={loadingParameters}
        error={parameterErr}
        onRefresh={() => void loadParameters()}
        onSaveTensor={(name, payload) => {
          const next = { ...parameterTensors, [name]: payload };
          setParameterTensors(next);
          patchNodeParameters(next);
          setParametersOpen(false);
        }}
        onClose={() => setParametersOpen(false)}
      />
    </div>
  );
}
