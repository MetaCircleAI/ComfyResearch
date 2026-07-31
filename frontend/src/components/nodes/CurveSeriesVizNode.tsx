import {
  Handle,
  Position,
  useReactFlow,
  useStore,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { useCallback } from "react";
import { CurveSeriesPlotBody } from "./CurveSeriesPlotBody";
import {
  defaultCurveSeriesVizSettings,
  resolveCurveSeriesViz,
  type CurveSeriesVizSettings,
} from "../../graph/curveSeriesVizResolution";
import type { CurveSeriesPlotXMode } from "../../graph/curveSeriesPlot";

export type CurveSeriesVizNodeData = Partial<CurveSeriesVizSettings> & {
  /** X axis: raw training step, or 0–100% progress per series (default, for SB/LB overlay). */
  plotXMode?: CurveSeriesPlotXMode;
  plotXKey?: string;
};

export function defaultCurveSeriesVizData(): CurveSeriesVizNodeData {
  return defaultCurveSeriesVizSettings();
}

export function CurveSeriesVizNode({ id, selected }: NodeProps) {
  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) => resolveCurveSeriesViz(state.nodes as Node[], state.edges as Edge[], id),
      [id],
    ),
  );

  const nodeData = useStore(
    useCallback((state) => {
      const n = state.nodes.find((x) => x.id === id);
      const raw = (n?.data ?? {}) as Partial<CurveSeriesVizNodeData>;
      const def = defaultCurveSeriesVizData();
      return {
        logScaleX: raw.logScaleX ?? def.logScaleX,
        logScaleY: raw.logScaleY ?? def.logScaleY,
        dualAxis: raw.dualAxis ?? def.dualAxis,
        plotXMode: raw.plotXMode ?? def.plotXMode,
        plotXKey: raw.plotXKey ?? def.plotXKey,
      };
    }, [id]),
  );

  const updatePlot = useCallback(
    (patch: Partial<CurveSeriesVizNodeData>) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = { ...defaultCurveSeriesVizData(), ...(n.data as Partial<CurveSeriesVizNodeData>) };
          return { ...n, data: { ...prev, ...patch } };
        }),
      );
    },
    [id, setNodes],
  );

  const plotXMode = nodeData.plotXMode ?? "progress";

  return (
    <div
      className={`cr-node cr-node--curve-series-viz${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}
    >
      <div className="cr-node__header">Curve series viz</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-tensor-viz__io nodrag nopan" aria-label="Curve series viz sockets">
          <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
            <div className="cr-tviz-socket-row__left">
              <Handle
                type="target"
                position={Position.Left}
                id="curves"
                className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
              />
              <span className="cr-tviz-socket-label">curves</span>
            </div>
            <div className="cr-tviz-socket-row__right">
              <span className="cr-tviz-socket-label">compare</span>
              <Handle type="source" position={Position.Right} id="compare" className="cr-handle-source" />
            </div>
          </div>
        </div>
        <p className="cr-sweep-viz__hint cr-tensor-viz__hint">
          Connect <strong>Curve series table</strong> <code>series</code> → <code>curves</code>. The resolved chart can
          feed <strong>Metric compare</strong> through <code>compare</code>.
        </p>

        {resolved ? (
          <div className="cr-sweep-viz__plot-wrap nodrag nopan">
            <div className="cr-tviz-chart-controls nodrag nopan">
              <label className="cr-tviz-check">
                <input
                  type="radio"
                  name={`${id}-plot-x`}
                  checked={plotXMode === "progress"}
                  onChange={() => updatePlot({ plotXMode: "progress" })}
                />
                progress %
              </label>
              <label className="cr-tviz-check">
                <input
                  type="radio"
                  name={`${id}-plot-x`}
                  checked={plotXMode === "step"}
                  onChange={() => updatePlot({ plotXMode: "step" })}
                />
                step
              </label>
              <label className="cr-tviz-check cr-tviz-check--log-x">
                <input
                  type="checkbox"
                  checked={!!nodeData.logScaleX}
                  disabled={!resolved.canLogX}
                  onChange={(e) => updatePlot({ logScaleX: e.target.checked })}
                />
                log x
              </label>
              <label className="cr-tviz-check cr-tviz-check--log-y">
                <input
                  type="checkbox"
                  checked={!!nodeData.logScaleY}
                  disabled={!resolved.canLogY}
                  onChange={(e) => updatePlot({ logScaleY: e.target.checked })}
                />
                log y
              </label>
            </div>
            <CurveSeriesPlotBody
              resolved={{ ...resolved, settings: { ...resolved.settings, ...nodeData } }}
              chartId={id}
            />
          </div>
        ) : (
          <p className="cr-tensor-viz__hint">Curve visualization is unavailable.</p>
        )}
      </div>
    </div>
  );
}
