/** Records first hidden-layer weight snapshots as [n_steps][n_neurons][input_dim] trajectories. */
export type ObservableNeuronTrajectory2dNodeData = {
  instanceTitle?: string;
};

export function defaultObservableNeuronTrajectory2dData(): ObservableNeuronTrajectory2dNodeData {
  return {};
}
