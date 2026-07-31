/** Symmetrized initialization data (Chizat et al. 2019, Section 3.1). */
export type SymmetrizedMlpInitNodeData = {
  instanceTitle?: string;
  /** Gaussian std τ for the first m/2 neurons; second half is mirrored with negated output scalars.
   *  Guarantees zero network output at initialization. Default 1.0. */
  tau?: number | number[];
};

export function defaultSymmetrizedMlpInitData(): SymmetrizedMlpInitNodeData {
  return { tau: 1.0 };
}
