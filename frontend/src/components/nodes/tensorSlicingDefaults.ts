export type TensorSliceSpec = {
  dimension: number;
  /**
   * Comma-separated integer indices for this dimension (negatives count from the end, e.g. -1 = last).
   * One index collapses the dimension; multiple indices keep it with size=len(indices).
   */
  indices: string;
};

export type TensorSlicingNodeData = {
  slices: TensorSliceSpec[];
};

export function defaultTensorSlicingNodeData(): TensorSlicingNodeData {
  return {
    slices: [{ dimension: 0, indices: "0" }],
  };
}
