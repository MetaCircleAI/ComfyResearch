import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo } from "react";
import { CurveSeriesPlotBody } from "./CurveSeriesPlotBody";
import { SweepVizLinePlot } from "./SweepVizLinePlot";
import { resolveMetricCompareSource } from "../../graph/curveSeriesVizResolution";
import { buildMetricCompareOverlay } from "../../graph/metricComparePlot";
import { dualAxisWarranted } from "../../graph/sweepVizPlot";

type MetricCompareNodeData = { layout?: "horizontal" | "vertical" | "overlay"; logScaleX?: boolean; logScaleY?: boolean };
const defaults = (): Required<MetricCompareNodeData> => ({ layout: "horizontal", logScaleX: false, logScaleY: false });
const isCurveSource = (source: NonNullable<ReturnType<typeof resolveMetricCompareSource>>) => "settings" in source;

export function MetricCompareNode({ id, selected }: NodeProps) {
  const { setNodes } = useReactFlow();
  const sources = useStore(useCallback((state) => ({
    left: resolveMetricCompareSource(state.nodes as Node[], state.edges as Edge[], id, "left"),
    right: resolveMetricCompareSource(state.nodes as Node[], state.edges as Edge[], id, "right"),
  }), [id]));
  const data = useStore(useCallback((state) => ({ ...defaults(), ...((state.nodes.find((node) => node.id === id)?.data ?? {}) as MetricCompareNodeData) }), [id]));
  const patch = useCallback((next: Partial<MetricCompareNodeData>) => setNodes((nodes) => nodes.map((node) =>
    node.id === id ? { ...node, data: { ...defaults(), ...(node.data as MetricCompareNodeData), ...next } } : node,
  )), [id, setNodes]);

  const overlay = useMemo(() => buildMetricCompareOverlay(sources.left, sources.right), [sources]);
  const missing = [sources.left ? null : "A", sources.right ? null : "B"].filter(Boolean).join(" and ");
  const empty = [sources.left?.plotSeries.length === 0 ? "A" : null, sources.right?.plotSeries.length === 0 ? "B" : null].filter(Boolean).join(" and ");

  return <div className={`cr-node cr-node--metric-compare${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}>
    <div className="cr-node__header">Metric compare</div>
    <div className="cr-node__body cr-node__body--compact">
      <div className="cr-tensor-viz__io nodrag nopan" aria-label="Metric compare sockets">
        {(["left", "right"] as const).map((slot, index) => <div className="cr-tviz-socket-row" key={slot}>
          <Handle type="target" position={Position.Left} id={slot} className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket" />
          <span className="cr-tviz-socket-label">source {index === 0 ? "A" : "B"}</span>
        </div>)}
      </div>
      <p className="cr-sweep-viz__hint cr-tensor-viz__hint">Connect a 1D visualization <code>compare</code> output to each source.</p>
      <div className="cr-tviz-chart-controls nodrag nopan">
        {(["horizontal", "vertical", "overlay"] as const).map((layout) => <label className="cr-tviz-check" key={layout}>
          <input type="radio" name={`${id}-layout`} checked={data.layout === layout} onChange={() => patch({ layout })} />{layout === "overlay" ? "same plot" : layout}
        </label>)}
        {data.layout === "overlay" ? <>
          <label className="cr-tviz-check cr-tviz-check--log-x"><input type="checkbox" checked={data.logScaleX} disabled={!overlay.canLogX} onChange={(event) => patch({ logScaleX: event.target.checked })} />log x</label>
          <label className="cr-tviz-check cr-tviz-check--log-y"><input type="checkbox" checked={data.logScaleY} disabled={!overlay.canLogY} onChange={(event) => patch({ logScaleY: event.target.checked })} />log y</label>
        </> : null}
      </div>
      {data.layout === "overlay" ? <div className="cr-sweep-viz__plot-wrap nodrag nopan">
        {overlay.series.length ? <SweepVizLinePlot chartId={`${id}-overlay`} series={overlay.series} xKey={overlay.xKey} xIsNumeric yAxisLabel={dualAxisWarranted(overlay.series) ? "loss" : "value"} yAxisLabelRight="accuracy" legendSummary={`${overlay.series.length} series`} logScaleX={data.logScaleX && overlay.canLogX} logScaleY={data.logScaleY && overlay.canLogY} dualAxis={dualAxisWarranted(overlay.series)} showMarkers={false} /> : <p className="cr-sweep-viz__plot-muted">No source series to compare.</p>}
        {missing ? <p className="cr-tensor-viz__hint">Source {missing} is not connected.</p> : null}
        {empty ? <p className="cr-tensor-viz__hint">Source {empty} has no selected series.</p> : null}
      </div> : <div className={`cr-metric-compare__panels cr-metric-compare__panels--${data.layout}`}>
        {([sources.left, sources.right] as const).map((source, index) => <section className="cr-metric-compare__panel" key={index}>
          <strong>Source {index === 0 ? "A" : "B"}</strong>
          {source ? isCurveSource(source) ? (
            <CurveSeriesPlotBody resolved={source} chartId={`${id}-${index === 0 ? "left" : "right"}`} />
          ) : source.plotSeries.length ? (
            <SweepVizLinePlot
              chartId={`${id}-${index === 0 ? "left" : "right"}`}
              series={source.plotSeries}
              xKey={source.xKey}
              xIsNumeric
              yAxisLabel={source.yAxisLabel}
              legendSummary={`${source.plotSeries.length} series`}
              logScaleX={source.logScaleX && source.canLogX}
              logScaleY={source.logScaleY && source.canLogY}
              showMarkers={false}
            />
          ) : <p className="cr-sweep-viz__plot-muted">No visible series to compare.</p> : <p className="cr-tensor-viz__hint">Connect source {index === 0 ? "A" : "B"}.</p>}
        </section>)}
      </div>}
    </div>
  </div>;
}
