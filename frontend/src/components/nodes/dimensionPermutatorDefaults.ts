export type DimensionPermutatorNodeData = {
  /** axes[outDim] = inDim (numpy.transpose order). Edited in the UI as Einstein-style `i j k -> k j i`. */
  axes: number[];
};

export function defaultDimensionPermutatorData(): DimensionPermutatorNodeData {
  return { axes: [] };
}
