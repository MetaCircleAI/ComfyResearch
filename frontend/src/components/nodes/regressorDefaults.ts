export type RegressorNodeData = {
  /** Bumped on Run and when new inputs re-trigger the fit UI. */
  fitNonce: number;
};

export function defaultRegressorData(): RegressorNodeData {
  return { fitNonce: 0 };
}
