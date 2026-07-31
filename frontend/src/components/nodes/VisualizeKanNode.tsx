import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultVisualizeKanData,
  KAN_PLOT_METRIC_OPTIONS,
  type DatasetSampleSplitId,
  type KanPlotMetricId,
  type VisualizeKanNodeData,
} from "./visualizeKanDefaults";

const DATASET_SAMPLE_SPLIT_OPTIONS: { id: DatasetSampleSplitId; label: string }[] = [
  { id: "train", label: "train split" },
  { id: "test", label: "test split" },
];

function patchData(
  id: string,
  patch: Partial<VisualizeKanNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultVisualizeKanData();
      const cur = (n.data ?? {}) as Partial<VisualizeKanNodeData>;
      const prev: VisualizeKanNodeData = {
        plotPngBase64: cur.plotPngBase64 ?? def.plotPngBase64,
        lastPlotError: cur.lastPlotError ?? def.lastPlotError,
        datasetSampleSplit: cur.datasetSampleSplit ?? def.datasetSampleSplit,
        sampleCount: cur.sampleCount ?? def.sampleCount,
        plotScale: cur.plotScale ?? def.plotScale,
        plotMetric: cur.plotMetric ?? def.plotMetric,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function VisualizeKanNode({ id, data, selected }: NodeProps) {
  const def = defaultVisualizeKanData();
  const raw = (data ?? {}) as Partial<VisualizeKanNodeData>;
  const d: VisualizeKanNodeData = {
    plotPngBase64: raw.plotPngBase64 ?? def.plotPngBase64,
    lastPlotError: raw.lastPlotError ?? def.lastPlotError,
    datasetSampleSplit: raw.datasetSampleSplit ?? def.datasetSampleSplit,
    sampleCount: raw.sampleCount ?? def.sampleCount,
    plotScale: raw.plotScale ?? def.plotScale,
    plotMetric: raw.plotMetric ?? def.plotMetric,
  };
  const { setNodes, getNodes, getEdges } = useReactFlow();
  const [loading, setLoading] = useState(false);

  const datasetWire = useStore(
    useCallback(
      (state) => {
        const e = state.edges.find((x) => x.target === id && (x.targetHandle ?? "") === "dataset");
        if (!e) return null;
        const ds = state.nodes.find((n) => n.id === e.source);
        if (!ds) return null;
        const sh = (e.sourceHandle ?? "").trim();
        const fromTest =
          sh === "test_dataset" || (sh === "dataset" && (d.datasetSampleSplit ?? "train") === "test");
        const dd = (ds.data ?? {}) as Record<string, unknown>;
        const raw = fromTest
          ? (Array.isArray(dd.testSize) ? dd.testSize[0] : dd.testSize)
          : (Array.isArray(dd.trainSize) ? dd.trainSize[0] : dd.trainSize);
        const n = typeof raw === "number" ? raw : Number(raw);
        const configured = Number.isFinite(n) ? Math.floor(n) : 0;
        const effective = configured < 1 ? 0 : Math.min(4096, Math.max(8, configured));
        return { fromTest, configured, effective, kind: String(ds.type) };
      },
      [id, d.datasetSampleSplit],
    ),
  );

  const update = useCallback(
    (patch: Partial<VisualizeKanNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  const refresh = useCallback(async () => {
    update({ lastPlotError: null });
    setLoading(true);
    try {
      const g = serializeGraphForApi(getNodes(), getEdges());
      const sc = intChoices(d.sampleCount, 256)[0] ?? 256;
      const ps = floatChoices(d.plotScale, 0.35)[0] ?? 0.35;
      const res = await fetch("/api/kan_plot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes: g.nodes,
          edges: g.edges,
          visualize_kan_node_id: id,
          sample_count: sc,
          plot_scale: ps,
          plot_metric: d.plotMetric,
          dpi: 120,
        }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = (await res.json()) as { detail?: unknown };
          if (j?.detail != null) {
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const j = (await res.json()) as { plot_png_base64?: string };
      const b64 = typeof j.plot_png_base64 === "string" ? j.plot_png_base64 : "";
      if (!b64) throw new Error("Server returned an empty plot.");
      update({ plotPngBase64: b64, lastPlotError: null });
    } catch (e) {
      update({
        plotPngBase64: "",
        lastPlotError: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setLoading(false);
    }
  }, [d.plotMetric, d.plotScale, d.sampleCount, datasetWire, getEdges, getNodes, id, update]);

  const sampleCountHint = useMemo(() => {
    if (!datasetWire) return null;
    const split = datasetWire.fromTest ? "test" : "train";
    if (datasetWire.configured < 1) {
      return `Dataset ${split}: 0 samples — increase ${datasetWire.fromTest ? "testSize" : "trainSize"}.`;
    }
    return `Dataset ${split}: ${datasetWire.configured} sample${datasetWire.configured === 1 ? "" : "s"} (uses ${datasetWire.effective} after 8…4096 clamp).`;
  }, [datasetWire]);

  const src = d.plotPngBase64.trim() ? `data:image/png;base64,${d.plotPngBase64.trim()}` : "";

  return (
    <div
      className={`cr-node cr-node--visualize-kan${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-model)" }}
    >
      <div className="cr-node__header cr-node__header--activation">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">{readInstanceTitle(d, "Visualize KAN")}</span>
          <button type="button" className="cr-activation-collect-btn nodrag nopan" disabled={loading} onClick={() => void refresh()}>
            {loading ? "…" : "Refresh plot"}
          </button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Visualize KAN inputs">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model"
              />
              <span className="cr-trainer-socket-label">model</span>
            </div>
          </div>
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

        {datasetWire ? (
          <>
            <DiscreteMultiSelect
              label="dataset split for plot"
              options={DATASET_SAMPLE_SPLIT_OPTIONS}
              value={d.datasetSampleSplit ?? "train"}
              onCommit={(next) =>
                update({
                  datasetSampleSplit: (Array.isArray(next) ? next[0] : next) as DatasetSampleSplitId,
                })
              }
              singleSelect
              ariaLabel="Which dataset split to sample for KAN plot"
            />
            <p className="cr-node__hint cr-visualize-kan__sample-hint" title="Batch size and x are taken from this dataset when you Refresh.">
              {sampleCountHint}
            </p>
          </>
        ) : (
          <ComfyIntListField
            label="sample count"
            values={intChoices(d.sampleCount, 256)}
            min={8}
            max={4096}
            title="Random batch size for the forward pass before plotting (no dataset wired)"
            onCommit={(vals) => update({ sampleCount: packIntList(vals) })}
            ariaLabel="Sample count for KAN plot"
          />
        )}
        <ComfyFloatListField
          label="plot scale"
          values={floatChoices(d.plotScale, 0.35)}
          min={0.05}
          max={2}
          onCommit={(vals) => update({ plotScale: packFloatList(vals) })}
          ariaLabel="KAN plot scale"
        />
        <DiscreteMultiSelect
          label="plot metric"
          options={KAN_PLOT_METRIC_OPTIONS}
          value={d.plotMetric}
          onCommit={(next) =>
            update({
              plotMetric: (Array.isArray(next) ? next[0] : next) as KanPlotMetricId,
            })
          }
          singleSelect
          ariaLabel="pykan plot metric"
        />

        <p className="cr-node__hint">
          Wire a <strong>kan_model</strong> or a <strong>model checkpoint</strong> from a trained KAN. Optionally wire{" "}
          <strong>dataset</strong> so the plot uses the same input law as training; pick train vs test split above when
          both are configured. Otherwise set sample count for random Gaussian inputs. Then Refresh to run pykan&apos;s{" "}
          <code>plot()</code>.
        </p>

        {d.lastPlotError ? <p className="cr-activation-scan-msg">{d.lastPlotError}</p> : null}

        {src ? (
          <div className="cr-visualize-kan__imgwrap nodrag nopan">
            <img className="cr-visualize-kan__img" src={src} alt="KAN structure plot from pykan" />
          </div>
        ) : (
          <p className="cr-activation-scan-msg">No plot yet — connect a KAN and press Refresh plot.</p>
        )}
      </div>
    </div>
  );
}
