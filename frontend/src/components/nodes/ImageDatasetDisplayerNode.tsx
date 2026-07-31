import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { VISION_DATASET_KINDS } from "./visionDatasetDefaults";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultImageDatasetDisplayerData,
  parseInclusiveIndexRange,
  type ImageDatasetDisplayerNodeData,
  type VisionSplit,
} from "./imageDatasetDisplayerDefaults";

function patchData(
  id: string,
  prev: ImageDatasetDisplayerNodeData,
  patch: Partial<ImageDatasetDisplayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

type GalleryResponse = {
  images: string[];
  labels: number[];
  warnings?: string[];
};

export function ImageDatasetDisplayerNode({ id, data, selected }: NodeProps) {
  const d = defaultImageDatasetDisplayerData((data ?? {}) as Partial<ImageDatasetDisplayerNodeData>);
  const { setNodes, getEdges, getNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<ImageDatasetDisplayerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const [images, setImages] = useState<string[]>([]);
  const [labels, setLabels] = useState<number[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const incoming = useStore(
    useCallback(
      (state) => {
        const edges = state.edges as Edge[];
        return edges.find((e) => e.target === id && (e.targetHandle ?? "").trim() === "dataset");
      },
      [id],
    ),
  );

  const sourceNode = useStore(
    useCallback(
      (state) => {
        if (!incoming?.source) return undefined;
        return (state.nodes as Node[]).find((n) => n.id === incoming.source);
      },
      [incoming],
    ),
  );

  const canRun = useMemo(() => {
    if (!incoming || !sourceNode) return false;
    const t = String(sourceNode.type ?? "");
    return (VISION_DATASET_KINDS as readonly string[]).includes(t);
  }, [incoming, sourceNode]);

  const rangeStart = useMemo(() => parseInclusiveIndexRange(d.indexRange)?.start ?? 0, [d.indexRange]);

  const loadPreview = useCallback(async () => {
    setStatus(null);
    const rng = parseInclusiveIndexRange(d.indexRange);
    if (!rng) {
      setStatus('Invalid index range. Use e.g. "10-19" or a single index "5".');
      setImages([]);
      setLabels([]);
      return;
    }
    const edges = getEdges();
    const nodes = getNodes();
    const inc = edges.find((e) => e.target === id && (e.targetHandle ?? "").trim() === "dataset");
    const src = inc ? nodes.find((n) => n.id === inc.source) : undefined;
    if (!inc || !src) {
      setStatus("Wire a vision dataset (MNIST, Gaussian blob, shape world, or hole counting) into the dataset input.");
      setImages([]);
      setLabels([]);
      return;
    }
    const dsType = String(src.type ?? "");
    if (!(VISION_DATASET_KINDS as readonly string[]).includes(dsType)) {
      setStatus("Source must be a vision dataset (MNIST, Gaussian blob, shape world, or hole counting).");
      setImages([]);
      setLabels([]);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/vision_dataset_gallery", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_node_type: dsType,
          dataset_data: (src.data ?? {}) as Record<string, unknown>,
          split: d.split,
          start_index: rng.start,
          end_index: rng.end,
        }),
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          const t = await res.text().catch(() => "");
          if (t) detail = t.slice(0, 400);
        }
        setStatus(detail);
        setImages([]);
        setLabels([]);
        return;
      }
      const payload = (await res.json()) as GalleryResponse;
      setImages(payload.images ?? []);
      setLabels(payload.labels ?? []);
      const warn = (payload.warnings ?? []).join(" ");
      setStatus(warn || `Showing ${payload.images?.length ?? 0} image(s).`);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setImages([]);
      setLabels([]);
    } finally {
      setLoading(false);
    }
  }, [d.indexRange, d.split, getEdges, getNodes, id]);

  const cols = Math.max(1, Math.min(12, Math.floor(Number(d.columnsPerRow)) || 5));

  return (
    <div
      className={`cr-node cr-node--image-dataset-displayer${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">
            {readInstanceTitle(data as Record<string, unknown>, "Image dataset displayer")}
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Dataset input">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset</span>
            </div>
          </div>
        </div>

        <DiscreteMultiSelect<VisionSplit>
          label="split"
          options={[
            { id: "train", label: "train" },
            { id: "test", label: "test" },
          ]}
          value={d.split}
          onCommit={(v) => {
            const one = (Array.isArray(v) ? v[0] : v) ?? "train";
            update({ split: one as VisionSplit });
          }}
          ariaLabel="Train or test split"
          singleSelect
        />

        <div className="cr-comfy-field">
          <label className="cr-comfy-widget cr-comfy-widget--flush">
            <span className="cr-comfy-widget__label" title="Inclusive indices, e.g. 10-19 for ten images.">
              index range
            </span>
            <input
              className="cr-input cr-comfy-widget__control"
              type="text"
              value={d.indexRange}
              onChange={(e) => update({ indexRange: e.target.value })}
              aria-label="Inclusive index range (e.g. 10-19)"
            />
          </label>
        </div>

        <ComfyIntField
          label="images per row"
          value={cols}
          min={1}
          max={12}
          title="Grid columns (max 12)."
          ariaLabel="Images per row"
          onCommit={(columnsPerRow) => update({ columnsPerRow })}
        />

        <div className="cr-tensor-constant-footer nodrag nopan">
          <button
            type="button"
            className="cr-trainer-train-btn nodrag nopan"
            disabled={!canRun || loading}
            onClick={() => void loadPreview()}
          >
            {loading ? "Loading…" : "Show images"}
          </button>
        </div>

        {status ? <p className="cr-activation-collect-summary">{status}</p> : null}

        {images.length > 0 ? (
          <div
            className="cr-image-dataset-displayer__grid nodrag nopan"
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
            }}
          >
            {images.map((src, i) => (
              <figure key={`${i}-${src.slice(0, 32)}`} className="cr-image-dataset-displayer__cell">
                <img src={src} alt={`sample ${i}`} className="cr-image-dataset-displayer__img" />
                <figcaption className="cr-image-dataset-displayer__cap">
                  #{rangeStart + i}
                  {labels[i] != null ? ` · y=${labels[i]}` : ""}
                </figcaption>
              </figure>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
