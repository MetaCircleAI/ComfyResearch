export type FakeTensorDtype = "long" | "float";

export type FakeTensorNodeData = {
  shape: number[];
  dtype: FakeTensorDtype;
  lastError: string | null;
};

export function defaultFakeTensorData(): FakeTensorNodeData {
  return {
    shape: [2, 3, 4],
    dtype: "float",
    lastError: null,
  };
}
