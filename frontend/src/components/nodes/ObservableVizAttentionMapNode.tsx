import { useEffect, useState } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { TensorHeatmap } from "./TensorHeatmap";
import { VizSocketsBar } from "./VizSocketsBar";
import { defaultAttentionMapVizData, type AttentionMapSlice, type AttentionMapVizNodeData } from "./attentionMapVizDefaults";
import { ObservableVizHeaderBar } from "./ObservableVizHeaderBar";
import { useObservableVizHeaderTitle } from "./observableVizTitle";

function patchData(id: string, patch: Partial<AttentionMapVizNodeData>, setNodes: (updater: (nodes: Node[]) => Node[]) => void) {
  setNodes((nodes) => nodes.map((node) => node.id === id ? { ...node, data: { ...defaultAttentionMapVizData(), ...(node.data as AttentionMapVizNodeData), ...patch } } : node));
}

function values(slices: AttentionMapSlice[], key: "layer" | "batch" | "head", matches: Partial<AttentionMapSlice> = {}) {
  return [...new Set(slices.filter((slice) => Object.entries(matches).every(([k, value]) => slice[k as keyof AttentionMapSlice] === value)).map((slice) => slice[key]))].sort((a, b) => a - b);
}

export function ObservableVizAttentionMapNode({ id, data, selected }: NodeProps) {
  const d = { ...defaultAttentionMapVizData(), ...(data as Partial<AttentionMapVizNodeData>) } as AttentionMapVizNodeData;
  const { setNodes } = useReactFlow();
  const title = useObservableVizHeaderTitle(d.pairedObservableId);
  const frames = d.attentionMapFrames ?? [];
  const frame = frames.find((item) => item.step === d.selectedFrameStep) ?? frames.at(-1);
  const slices = frame?.slices ?? [];
  const layers = values(slices, "layer");
  const layer = layers.includes(d.selectedLayer ?? -1) ? d.selectedLayer! : layers[0];
  const batches = values(slices, "batch", { layer });
  const batch = batches.includes(d.selectedBatch ?? -1) ? d.selectedBatch! : batches[0];
  const heads = values(slices, "head", { layer, batch });
  const head = heads.includes(d.selectedHead ?? -1) ? d.selectedHead! : heads[0];
  const slice = slices.find((item) => item.layer === layer && item.batch === batch && item.head === head);
  const currentSlice: AttentionMapSlice = slice ?? {
    layer: -1, batch: -1, head: -1, map: [], token_ids: null, source_shape: [0, 0], row_start: 0, col_start: 0,
  };
  const [hover, setHover] = useState<{ row: number; col: number; value: number } | null>(null);
  useEffect(() => setHover(null), [frame, slice]);
  const valid = slice != null && Array.isArray(currentSlice.map) && currentSlice.map.length > 0 && currentSlice.map.every((row) => Array.isArray(row) && row.length === currentSlice.map[0]?.length);
  const rows = valid ? currentSlice.map.length : 0;
  const cols = valid ? currentSlice.map[0]!.length : 0;
  const tokenIds = valid && Array.isArray(currentSlice.token_ids) && currentSlice.token_ids.length === rows && rows === cols ? currentSlice.token_ids : null;
  const select = (label: string, value: number | undefined, options: number[], onChange: (next: number) => void) => <label className="cr-tviz-metric-select"><span className="cr-tviz-metric-select__lbl">{label}</span><select value={value ?? ""} onChange={(event) => onChange(Number(event.target.value))}>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>;
  const cropped = valid && (currentSlice.source_shape[0] > rows || currentSlice.source_shape[1] > cols);
  return <div className={`cr-node cr-node--observable-viz-user${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}>
    <ObservableVizHeaderBar id={id} pairedObservableId={d.pairedObservableId} title={title} />
    <div className="cr-node__body cr-node__body--tviz"><VizSocketsBar /><div className="cr-tviz-chart-divider" aria-hidden />
      {frames.length > 0 ? <div className="cr-tviz-chart-controls nodrag nopan"><label className="cr-tviz-metric-select"><span className="cr-tviz-metric-select__lbl">frame</span><select value={frame?.step ?? ""} onChange={(event) => patchData(id, { selectedFrameStep: Number(event.target.value) }, setNodes)}>{frames.map((item) => <option key={item.step} value={item.step}>step {item.step}</option>)}</select></label>{select("layer", layer, layers, (next) => patchData(id, { selectedLayer: next, selectedBatch: undefined, selectedHead: undefined }, setNodes))}{select("batch", batch, batches, (next) => patchData(id, { selectedBatch: next, selectedHead: undefined }, setNodes))}{select("head", head, heads, (next) => patchData(id, { selectedHead: next }, setNodes))}</div> : null}
      <div className="cr-tviz-chart-wrap">{frames.length === 0 ? <p className="cr-tviz-hint">No attention maps yet. Maps are recorded at Trainer log ticks.</p> : !valid ? <p className="cr-tviz-hint">The selected frame has no valid recorded attention slice.</p> : <TensorHeatmap shape={[rows, cols]} values={currentSlice.map.flat().map(Number)} axisLabels={{ x: Array.from({ length: cols }, (_, i) => String(tokenIds?.[i] ?? currentSlice.col_start + i)), y: Array.from({ length: rows }, (_, i) => String(tokenIds?.[i] ?? currentSlice.row_start + i)), xTitle: tokenIds ? "key token id" : "key position", yTitle: tokenIds ? "query token id" : "query position" }} onHoverCell={setHover} />}{cropped ? <p className="cr-tviz-hint">sequence {currentSlice.source_shape[0]} x {currentSlice.source_shape[1]}; showing final query/key positions {currentSlice.row_start}-{currentSlice.row_start + rows - 1}.</p> : null}{hover && valid ? <p className="cr-tviz-hint">query {currentSlice.row_start + hover.row}{tokenIds ? ` (token ${tokenIds[hover.row]})` : ""} to key {currentSlice.col_start + hover.col}{tokenIds ? ` (token ${tokenIds[hover.col]})` : ""}: {hover.value.toFixed(3)}</p> : null}</div>
    </div>
  </div>;
}
