import type { CollectedActivationTensor } from "./activationDefaults";
import type { ListOr1 } from "./multiValueUtils";

export type TensorLinspaceSpace = "linear" | "log10";

export type TensorLinspaceNodeData = {
  start: ListOr1<number>;
  end: ListOr1<number>;
  numPoints: ListOr1<number>;
  space: TensorLinspaceSpace;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

export function defaultTensorLinspaceData(partial?: Partial<TensorLinspaceNodeData>): TensorLinspaceNodeData {
  return {
    start: partial?.start !== undefined ? partial.start : 0,
    end: partial?.end !== undefined ? partial.end : 1,
    numPoints: partial?.numPoints !== undefined ? partial.numPoints : 8,
    space: partial?.space ?? "linear",
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
