import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  centeredRmsd,
  defaultProteinStructureComparisonVizData,
  extractStructureCoordsFromTensor,
  parseCoordsFlat,
  projectCoords2d,
  type ProteinStructureComparisonVizNodeData,
} from "./proteinStructureVizDefaults";
import { ComfyIntField } from "./comfyNumberFields";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
} from "../../graph/resolveUpstreamTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";

const W = 220;
const H = 140;

function patchData(
  id: string,
  patch: Partial<ProteinStructureComparisonVizNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const prev = defaultProteinStructureComparisonVizData((n.data ?? {}) as Partial<ProteinStructureComparisonVizNodeData>);
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function ProteinStructureComparisonVizNode({ id, data, selected }: NodeProps) {
  const d = defaultProteinStructureComparisonVizData((data ?? {}) as Partial<ProteinStructureComparisonVizNodeData>);
  const { setNodes } = useReactFlow();
  const predResolved = useStore(
    useCallback((state) => resolveUpstreamTensor(state.nodes as Node[], state.edges as FlowEdge[], id, "pred_coords"), [id]),
    resolvedTensorEqual,
  );
  const trueResolved = useStore(
    useCallback((state) => resolveUpstreamTensor(state.nodes as Node[], state.edges as FlowEdge[], id, "true_coords"), [id]),
    resolvedTensorEqual,
  );
  const { display: predDisplay, loading: predLoading } = useHydratedResolved(predResolved);
  const { display: trueDisplay, loading: trueLoading } = useHydratedResolved(trueResolved);

  const predFromTensor = useMemo(() => {
    if (predDisplay.kind !== "ok") return null;
    return extractStructureCoordsFromTensor(predDisplay.shape, predDisplay.values, d.sampleIndex ?? 0);
  }, [predDisplay, d.sampleIndex]);
  const trueFromTensor = useMemo(() => {
    if (trueDisplay.kind !== "ok") return null;
    return extractStructureCoordsFromTensor(trueDisplay.shape, trueDisplay.values, d.sampleIndex ?? 0);
  }, [trueDisplay, d.sampleIndex]);

  const predCoords = useMemo(() => {
    if (predFromTensor?.coords.length) return predFromTensor.coords;
    return parseCoordsFlat(d.predCoordsFlat);
  }, [d.predCoordsFlat, predFromTensor]);
  const trueCoords = useMemo(() => {
    if (trueFromTensor?.coords.length) return trueFromTensor.coords;
    return parseCoordsFlat(d.trueCoordsFlat);
  }, [d.trueCoordsFlat, trueFromTensor]);
  const predPts = useMemo(() => projectCoords2d(predCoords, W, H), [predCoords]);
  const truePts = useMemo(() => projectCoords2d(trueCoords, W, H), [trueCoords]);
  const predLine = useMemo(() => predPts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "), [predPts]);
  const trueLine = useMemo(() => truePts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "), [truePts]);
  const rmsd = useMemo(() => centeredRmsd(predCoords, trueCoords), [predCoords, trueCoords]);
  const predStatus = useMemo(() => {
    if (predFromTensor?.coords.length) {
      const samplePart =
        predFromTensor.sampleCount > 1
          ? `sample ${predFromTensor.sampleIndexUsed + 1}/${predFromTensor.sampleCount}`
          : "single sample";
      return `pred: upstream tensor (${samplePart})`;
    }
    if (predLoading) return "pred: loading…";
    if (predDisplay.kind === "none" && predResolved.kind !== "none") return `pred: fallback (${predDisplay.detail})`;
    return "pred: fallback/manual";
  }, [predFromTensor, predLoading, predDisplay, predResolved.kind]);
  const trueStatus = useMemo(() => {
    if (trueFromTensor?.coords.length) {
      const samplePart =
        trueFromTensor.sampleCount > 1
          ? `sample ${trueFromTensor.sampleIndexUsed + 1}/${trueFromTensor.sampleCount}`
          : "single sample";
      return `true: upstream tensor (${samplePart})`;
    }
    if (trueLoading) return "true: loading…";
    if (trueDisplay.kind === "none" && trueResolved.kind !== "none") return `true: fallback (${trueDisplay.detail})`;
    return "true: fallback/manual";
  }, [trueFromTensor, trueLoading, trueDisplay, trueResolved.kind]);
  const update = useCallback(
    (patch: Partial<ProteinStructureComparisonVizNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  return (
    <div className={`cr-node cr-node--training-viz${selected ? " cr-node--selected" : ""}`}>
      <div className="cr-node__header">
        <div className="cr-node__header-title">
          {readInstanceTitle(data as Record<string, unknown>, "Protein structure comparison")}
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Structure comparison I/O">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="pred_coords"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">pred</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="true_coords"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">true</span>
            </div>
          </div>
        </div>
        <svg width={W} height={H} className="cr-training-viz__chart nodrag nopan" role="img" aria-label="Structure comparison preview">
          <rect x={0} y={0} width={W} height={H} fill="rgba(255,255,255,0.02)" />
          {truePts.length > 1 ? (
            <polyline points={trueLine} fill="none" stroke="var(--cr-accent-dataset)" strokeWidth={1.4} opacity={0.75} />
          ) : null}
          {predPts.length > 1 ? (
            <polyline points={predLine} fill="none" stroke="var(--cr-accent-model)" strokeWidth={1.4} opacity={0.8} />
          ) : null}
        </svg>
        <ComfyIntField
          label="sample index"
          value={Math.max(0, Math.floor(Number(d.sampleIndex ?? 0)))}
          min={0}
          onCommit={(n) => update({ sampleIndex: Math.max(0, Math.floor(Number(n))) })}
          ariaLabel="Protein structure comparison sample index"
          title="If pred/true tensors are batched, this sample index is applied to both."
        />
        <p className="cr-activation-collect-summary">
          {rmsd == null
            ? "Need both predicted and true coordinates."
            : `RMSD (centered): ${rmsd.toFixed(4)} · pred n=${predCoords.length}, true n=${trueCoords.length}`}
        </p>
        <p className="cr-activation-collect-summary">{predStatus} · {trueStatus}</p>
      </div>
    </div>
  );
}

