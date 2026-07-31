export type TensorMultiInputNodeData = {
  inputCount: number;
  concatDimension?: number;
};

export const TENSOR_MULTI_INPUT_MIN_COUNT = 2;

export function clampTensorInputCount(v: unknown): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return TENSOR_MULTI_INPUT_MIN_COUNT;
  return Math.max(TENSOR_MULTI_INPUT_MIN_COUNT, Math.floor(n));
}

export function defaultTensorStackData(partial?: Partial<TensorMultiInputNodeData>): TensorMultiInputNodeData {
  return {
    inputCount: clampTensorInputCount(partial?.inputCount),
  };
}

export function defaultTensorConcatData(partial?: Partial<TensorMultiInputNodeData>): TensorMultiInputNodeData {
  return {
    inputCount: clampTensorInputCount(partial?.inputCount),
    concatDimension: Number.isFinite(Number(partial?.concatDimension)) ? Math.floor(Number(partial?.concatDimension)) : 0,
  };
}
