/** Fractional L2 displacement ||w(t) - w(0)||_F / ||w(0)||_F from initial weights. */
export type ObservableWeightDisplacementNodeData = {
  instanceTitle?: string;
};

export function defaultObservableWeightDisplacementData(): ObservableWeightDisplacementNodeData {
  return {};
}
