import { intChoices } from "../components/nodes/multiValueUtils";
import type { TrainingLengthMode } from "../components/nodes/trainerDefaults";
import { applyAssignmentsToNodes, type SerializedTrainNode, type TrainSeriesAssignment } from "./trainSeriesPlan";

export type TrainWire = {
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

function trainerDatasetSourceId(edges: TrainWire[], trainerId: string): string | null {
  const e = edges.find((x) => x.target === trainerId && (x.targetHandle ?? null) === "dataset");
  return e?.source ?? null;
}

function scalarNodeIntField(value: unknown, fallback: number): number {
  return intChoices(value, fallback)[0] ?? fallback;
}

export function normalizeTrainingLengthMode(raw: unknown): TrainingLengthMode {
  return raw === "epochs" ? "epochs" : "steps";
}

/** Paper-style: steps = epochs × ceil(trainSize / batchSize); batchSize -1 → full batch. */
export function epochsToTrainingSteps(epochs: number, trainSize: number, batchSize: number): number {
  const ep = Math.max(1, Math.floor(epochs));
  const n = Math.max(1, Math.floor(trainSize));
  let b = Math.floor(batchSize);
  if (b <= 0) b = n;
  return ep * Math.ceil(n / b);
}

/** Read wired dataset `trainSize` (or legacy `train_size`). */
export function resolveDatasetTrainSizeForTrainer(
  nodes: Array<{ id: string; data?: Record<string, unknown> }>,
  edges: TrainWire[],
  trainerId: string,
): number | null {
  const dsId = trainerDatasetSourceId(edges, trainerId);
  if (!dsId) return null;
  const ds = nodes.find((n) => n.id === dsId);
  if (!ds) return null;
  const d = ds.data ?? {};
  const raw = d.trainSize ?? d.train_size;
  const n = scalarNodeIntField(raw, 0);
  return n >= 1 ? n : null;
}

/** Preview resolved steps for UI (epochs mode). */
export function previewTrainingStepsForTrainer(
  trainerData: Record<string, unknown>,
  trainSize: number | null,
): number | null {
  if (normalizeTrainingLengthMode(trainerData.trainingLengthMode) !== "epochs") return null;
  if (trainSize == null || trainSize < 1) return null;
  const epochs = scalarNodeIntField(trainerData.trainingEpochs, 100);
  const batchSize = scalarNodeIntField(trainerData.batchSize, -1);
  return epochsToTrainingSteps(epochs, trainSize, batchSize);
}

/**
 * Epoch-mode training length is resolved by the backend: `prepare_config` reads
 * `trainingLengthMode` / `trainingEpochs` and overrides `training_steps` itself
 * (including the CBS variable-batch formula the frontend cannot reproduce). The
 * frontend must NOT overwrite `trainingSteps` before POST — the backend validates
 * the raw value's bounds before the epoch override, so an inflated placeholder
 * could falsely reject a legal CBS run. `previewTrainingStepsForTrainer` above
 * stays display-only (≈N steps in the UI).
 */
export function resolveTrainingLengthOnNodes(
  nodes: SerializedTrainNode[],
  _edges: TrainWire[],
): SerializedTrainNode[] {
  return nodes;
}

export function applyAssignmentsAndResolveTrainingLength(
  nodes: SerializedTrainNode[],
  combo: TrainSeriesAssignment[],
  edges: TrainWire[],
): SerializedTrainNode[] {
  return resolveTrainingLengthOnNodes(applyAssignmentsToNodes(nodes, combo), edges);
}
