/**
 * Paired I/O strip (`AtomicLayerIoStrip`) uses distinct React Flow handle ids — duplicate `id="tensor"`
 * on the same node leaves only one handle registered for connections (XYFlow requires unique ids per node).
 */
export const LAYER_STRIP_TARGET_HANDLE = "tensor_in";
export const LAYER_STRIP_SOURCE_HANDLE = "tensor_out";

/** ``combined_model`` only: inner subgraph tail → wrapper right (paired with ``tensor_out`` to outside). */
export const COMBINED_MODEL_RETURN_TARGET_HANDLE = "tensor_return";

/** React Flow edge type: inward tangents while anchors use real handle geometry (see ``CombinedSubgraphIoEdge``). */
export const COMBINED_SUBGRAPH_IO_EDGE_TYPE = "combined_subgraph_io";

/** Left / target socket on the paired strip (legacy graphs used `tensor` for both sides). */
export function isLayerStripTargetHandle(h: string | null | undefined): boolean {
  const v = (h ?? "").trim();
  return v === LAYER_STRIP_TARGET_HANDLE || v === "tensor";
}

/** Right / source socket on the paired strip (legacy: `tensor`). */
export function isLayerStripSourceHandle(h: string | null | undefined): boolean {
  const v = (h ?? "").trim();
  return v === LAYER_STRIP_SOURCE_HANDLE || v === "tensor";
}
