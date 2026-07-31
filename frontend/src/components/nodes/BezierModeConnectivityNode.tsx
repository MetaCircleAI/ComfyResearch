import { Handle, Position, useReactFlow, type NodeProps } from "@xyflow/react";
import { useState } from "react";
import { interpolationAlphaCount, linearInterpolationPlotSeries } from "../../graph/linearInterpolationPlotSeries";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { serializeExecutionGraphForTarget } from "../../graph/trainSeriesPlan";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { SweepVizLinePlot } from "./SweepVizLinePlot";

type BezierData = Record<string, unknown>;

function numberText(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

export function BezierModeConnectivityNode({ id, data, selected }: NodeProps) {
  const { getNodes, getEdges, setNodes } = useReactFlow();
  const d = data as BezierData;
  const [running, setRunning] = useState(false);
  const update = (patch: BezierData) => setNodes((nodes) => nodes.map((node) =>
    node.id === id ? { ...node, data: { ...node.data, ...patch } } : node,
  ));
  const run = async () => {
    if (running) return;
    setRunning(true);
    update({ lastError: "" });
    try {
      const graph = serializeExecutionGraphForTarget(getNodes(), getEdges(), id);
      const response = await fetch("/api/parametric_path_sampler", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sampler_node_id: id, nodes: graph.nodes, edges: graph.edges }),
      });
      const payload = await response.json() as BezierData;
      if (!response.ok) throw new Error(String(payload.detail ?? "Bezier connectivity evaluation failed."));
      update({
        alphaSeries: payload.alphaSeries,
        linearTrainLoss: payload.linearTrainLoss,
        linearTestLoss: payload.linearTestLoss,
        linearTrainAcc: payload.linearTrainAcc,
        linearTestAcc: payload.linearTestAcc,
        bezierTrainLoss: payload.bezierTrainLoss,
        bezierTestLoss: payload.bezierTestLoss,
        bezierTrainAcc: payload.bezierTrainAcc,
        bezierTestAcc: payload.bezierTestAcc,
        linearLossBarrier: payload.linearLossBarrier,
        bezierLossBarrier: payload.bezierLossBarrier,
        runSummary: payload.summary,
      });
    } catch (error) {
      update({ lastError: error instanceof Error ? error.message : String(error) });
    } finally {
      setRunning(false);
    }
  };

  const showTrain = d.showTrainCurve !== false;
  const showTest = d.showTestCurve !== false;
  const lossSeries = linearInterpolationPlotSeries(
    d.alphaSeries, d.bezierTrainLoss, d.bezierTestLoss, showTrain, showTest, "loss",
  );
  const accuracySeries = linearInterpolationPlotSeries(
    d.alphaSeries, d.bezierTrainAcc, d.bezierTestAcc, showTrain, showTest, "accuracy",
  );
  const alphaCount = interpolationAlphaCount(d.alphaSeries);
  const activeSeriesKey = `${showTrain ? "train" : ""}-${showTest ? "test" : ""}`;

  return (
    <div className={`cr-node cr-node--parametric-path-sampler${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
      <div className="cr-node__header cr-node__header--activation">
        <div className="cr-node__header--row cr-node__header--activation-main">
          <span className="cr-node__header-title">{readInstanceTitle(data, "Bezier mode connectivity")}</span>
          <button type="button" className="cr-activation-collect-btn nodrag nopan" onClick={() => void run()} disabled={running}>{running ? "…" : "Run"}</button>
        </div>
      </div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Bezier mode connectivity sockets">
          {[["checkpoint A", "checkpoint_a", "18%"], ["checkpoint B", "checkpoint_b", "32%"], ["model", "model", "46%"], ["dataset", "dataset", "60%"], ["loss", "loss", "74%"]].map(([label, handle, top]) => (
            <div className="cr-trainer-io-row" key={handle}><div className="cr-trainer-io-row__leftwrap">
              <Handle type="target" position={Position.Left} id={handle} className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle" style={{ top }} />
              <span className="cr-trainer-socket-label">{label}</span>
            </div></div>
          ))}
        </div>
        <ComfyIntField label="alpha steps" value={Number(d.alphaSteps ?? 21)} min={2} max={201} onCommit={(alphaSteps) => update({ alphaSteps })} ariaLabel="Bezier alpha steps" />
        <ComfyIntField label="curve optimization steps" value={Number(d.curveOptimizationSteps ?? 500)} min={1} max={10000} onCommit={(curveOptimizationSteps) => update({ curveOptimizationSteps })} ariaLabel="Bezier optimization steps" />
        <ComfyIntField label="curve samples per step" value={Number(d.curveSamplesPerStep ?? 4)} min={1} max={32} onCommit={(curveSamplesPerStep) => update({ curveSamplesPerStep })} ariaLabel="Bezier samples per step" />
        <ComfyIntField label="curve batch size" value={Number(d.curveBatchSize ?? 256)} min={1} max={4096} onCommit={(curveBatchSize) => update({ curveBatchSize })} ariaLabel="Bezier curve batch size" />
        <ComfyFloatField label="curve learning rate" value={Number(d.curveLearningRate ?? 0.01)} min={0.000001} onCommit={(curveLearningRate) => update({ curveLearningRate })} ariaLabel="Bezier curve learning rate" />
        <label className="cr-node__field cr-node__field--checkbox nodrag nopan"><input type="checkbox" checked={Boolean(d.recomputeBnStats ?? true)} onChange={(event) => update({ recomputeBnStats: event.target.checked })} />Recompute BN stats</label>
        {Boolean(d.recomputeBnStats ?? true) ? <ComfyIntField label="BN calibration batches" value={Number(d.bnCalibrationBatches ?? 100)} min={1} max={100} onCommit={(bnCalibrationBatches) => update({ bnCalibrationBatches })} ariaLabel="BN calibration batches" /> : null}
        <p className="cr-observable-hint">linear loss barrier: {numberText(d.linearLossBarrier)} → Bezier: {numberText(d.bezierLossBarrier)}</p>
        {lossSeries.length > 0 ? <div className="cr-sweep-viz__plot-wrap nodrag nopan">
          <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked"><div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--series">
            <label className="cr-tviz-check cr-tviz-check--train"><input type="checkbox" checked={showTrain} onChange={(event) => update({ showTrainCurve: event.target.checked })} />train</label>
            <label className="cr-tviz-check cr-tviz-check--test"><input type="checkbox" checked={showTest} onChange={(event) => update({ showTestCurve: event.target.checked })} />test</label>
          </div></div>
          <p className="cr-sweep-viz__lineplot-chart-title">Bezier interpolation loss</p>
          <SweepVizLinePlot chartId={`${id}-bezier-loss-${activeSeriesKey}`} series={lossSeries} xKey="alpha" xIsNumeric yAxisLabel="cross-entropy" legendSummary="" logScaleX={false} logScaleY={false} showMarkers={alphaCount <= 41} showLegendList={false} />
          {accuracySeries.length > 0 ? <><p className="cr-sweep-viz__lineplot-chart-title">Bezier interpolation accuracy</p><SweepVizLinePlot chartId={`${id}-bezier-accuracy-${activeSeriesKey}`} series={accuracySeries} xKey="alpha" xIsNumeric yAxisLabel="accuracy" legendSummary="" logScaleX={false} logScaleY={false} showMarkers={alphaCount <= 41} showLegendList={false} /></> : null}
        </div> : null}
        {typeof d.runSummary === "string" ? <p className="cr-observable-hint">{d.runSummary}</p> : null}
        {typeof d.lastError === "string" && d.lastError ? <p className="cr-node-error">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
