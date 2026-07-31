import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  resolveUpstreamTensor,
  resolvedTensorEqual,
  type FlowEdge,
} from "../../graph/resolveUpstreamTensor";
import { useHydratedResolved } from "../../graph/useHydratedResolved";
import {
  defaultProteinStructureDisplayerData,
  extractStructureCoordsFromTensor,
  parseCoordsFlat,
  projectCoords2d,
  type ProteinStructureDisplayerNodeData,
} from "./proteinStructureVizDefaults";
import { ComfyIntField } from "./comfyNumberFields";

const W = 220;
const H = 140;

function patchData(
  id: string,
  patch: Partial<ProteinStructureDisplayerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const prev = defaultProteinStructureDisplayerData((n.data ?? {}) as Partial<ProteinStructureDisplayerNodeData>);
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function ProteinStructureDisplayerNode({ id, data, selected }: NodeProps) {
  const d = defaultProteinStructureDisplayerData((data ?? {}) as Partial<ProteinStructureDisplayerNodeData>);
  const { setNodes } = useReactFlow();
  const resolved = useStore(
    useCallback(
      (state) => {
        return resolveUpstreamTensor(state.nodes as Node[], state.edges as FlowEdge[], id, "coords");
      },
      [id],
    ),
    resolvedTensorEqual,
  );
  const { display, loading } = useHydratedResolved(resolved);

  const fromTensor = useMemo(() => {
    if (display.kind !== "ok") return null;
    return extractStructureCoordsFromTensor(display.shape, display.values, d.sampleIndex ?? 0);
  }, [display, d.sampleIndex]);
  const usingUpstream = Boolean(fromTensor && fromTensor.coords.length > 0);
  const coords = useMemo(
    () =>
      fromTensor?.coords.length
        ? fromTensor.coords
        : parseCoordsFlat(d.coordsFlat),
    [fromTensor, d.coordsFlat],
  );
  const resolvedCoordsFlat = useMemo(
    () => coords.map((c) => `${c[0] ?? 0},${c[1] ?? 0},${c[2] ?? 0}`).join("; "),
    [coords],
  );
  const pts = useMemo(() => projectCoords2d(coords, W, H), [coords]);
  const polyline = useMemo(() => pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "), [pts]);
  const statusLine = useMemo(() => {
    if (usingUpstream && fromTensor) {
      const samplePart =
        fromTensor.sampleCount > 1
          ? `sample ${fromTensor.sampleIndexUsed + 1}/${fromTensor.sampleCount}`
          : "single sample";
      return `${pts.length} points (from upstream tensor, ${samplePart})`;
    }
    if (loading) return "Loading tensor from server…";
    if (display.kind === "none" && resolved.kind !== "none") {
      return `${pts.length} points (fallback/manual; upstream unavailable: ${display.detail})`;
    }
    return pts.length > 0
      ? `${pts.length} points (fallback/manual coordsFlat)`
      : "No coordinates yet. Connect a source or set coordsFlat on node data.";
  }, [usingUpstream, fromTensor, pts.length, loading, display, resolved.kind, d.sampleIndex]);

  const update = useCallback(
    (patch: Partial<ProteinStructureDisplayerNodeData>) => patchData(id, patch, setNodes),
    [id, setNodes],
  );

  useEffect(() => {
    const prev = typeof d.resolvedCoordsFlat === "string" ? d.resolvedCoordsFlat : "";
    if (resolvedCoordsFlat === prev) return;
    update({ resolvedCoordsFlat });
  }, [d.resolvedCoordsFlat, resolvedCoordsFlat, update]);

  return (
    <div
      className={`cr-node cr-node--image-dataset-displayer${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">
            {readInstanceTitle(data as Record<string, unknown>, "Protein structure displayer")}
          </div>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Structure input">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
              <Handle
                type="target"
                position={Position.Left}
                id="coords"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">coords</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap cr-trainer-io-row__rightwrap--full">
              <Handle
                type="source"
                position={Position.Right}
                id="structure"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-handle cr-trainer-handle--output"
              />
              <span className="cr-trainer-output-label">structure</span>
            </div>
          </div>
        </div>
        <svg width={W} height={H} className="cr-training-viz__chart nodrag nopan" role="img" aria-label="Structure preview">
          <rect x={0} y={0} width={W} height={H} fill="rgba(255,255,255,0.02)" />
          {d.showPolyline && pts.length > 1 ? (
            <polyline points={polyline} fill="none" stroke="var(--cr-accent-dataset)" strokeWidth={1.5} />
          ) : null}
          {pts.map((p, i) => (
            <circle key={`${i}-${p.x}-${p.y}`} cx={p.x} cy={p.y} r={2} fill="var(--cr-accent-dataset)" />
          ))}
        </svg>
        <ComfyIntField
          label="sample index"
          value={Math.max(0, Math.floor(Number(d.sampleIndex ?? 0)))}
          min={0}
          onCommit={(n) => update({ sampleIndex: Math.max(0, Math.floor(Number(n))) })}
          ariaLabel="Protein structure sample index"
          title="If the upstream tensor has a batch dimension, choose which sample to render."
        />
        <p className="cr-activation-collect-summary">{statusLine}</p>
      </div>
    </div>
  );
}

