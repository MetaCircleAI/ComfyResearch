import type { Node } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import {
  defaultTrainerData,
  emptyTrainerTrainUi,
  type TrainerHostTrainUi,
  type TrainerNodeData,
  type TrainerTrainUi,
} from "../components/nodes/trainerDefaults";

export type TrainerLocalTrainUi = {
  trainLoading: boolean;
  trainPaused: boolean;
  trainProgressPct: number;
  trainSeriesBarPct: number;
  trainSeriesDual: boolean;
  trainSeriesCaptionLines: string[] | null;
  trainPhaseText: string | null;
  trainError: string | null;
};

export type ResolvedTrainerTrainDisplay = {
  loading: boolean;
  paused: boolean;
  progressPct: number;
  seriesBarPct: number;
  seriesDual: boolean;
  captionLines: string[] | null;
  phaseText: string | null;
  error: string | null;
  showProgress: boolean;
  hostDrivingProgress: boolean;
};

/** Prefer persisted ``trainUi`` (survives canvas remount), then host assist, then local state. */
export function resolveTrainerTrainDisplay(
  local: TrainerLocalTrainUi,
  persisted?: TrainerTrainUi,
  host?: TrainerHostTrainUi,
): ResolvedTrainerTrainDisplay {
  const fromPersisted = Boolean(persisted?.active);
  const loading = fromPersisted ? persisted!.loading : local.trainLoading;
  const paused = fromPersisted ? persisted!.paused : local.trainPaused;
  const hostDrivingProgress = Boolean(host?.active) && !loading && !paused;

  if (hostDrivingProgress) {
    return {
      loading,
      paused,
      progressPct: host?.progressPct ?? 0,
      seriesBarPct: host?.seriesBarPct ?? 0,
      seriesDual: Boolean(host?.seriesDual),
      captionLines: host?.captionLines ?? null,
      phaseText: local.trainPhaseText,
      error: local.trainError,
      showProgress: true,
      hostDrivingProgress: true,
    };
  }

  const progressPct = fromPersisted ? persisted!.progressPct : local.trainProgressPct;
  const seriesBarPct = fromPersisted ? persisted!.seriesBarPct : local.trainSeriesBarPct;
  const seriesDual = fromPersisted ? persisted!.seriesDual : local.trainSeriesDual;
  const captionLines = fromPersisted ? persisted!.captionLines : local.trainSeriesCaptionLines;
  const phaseText = local.trainPhaseText ?? (fromPersisted ? persisted!.phaseText : null);
  const error = fromPersisted ? persisted!.error : local.trainError;

  const showProgress =
    fromPersisted ||
    local.trainLoading ||
    local.trainPaused ||
    progressPct > 0 ||
    seriesBarPct > 0 ||
    seriesDual;

  return {
    loading,
    paused,
    progressPct,
    seriesBarPct,
    seriesDual,
    captionLines,
    phaseText,
    error,
    showProgress,
    hostDrivingProgress: false,
  };
}

/** Persist trainer Train-button UI on node data (survives canvas remount / tab switch). */
export function patchTrainerTrainUi(
  trainerId: string,
  patch: Partial<TrainerTrainUi> | null,
  setNodes: Dispatch<SetStateAction<Node[]>>,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== trainerId || n.type !== "trainer") return n;
      const def = defaultTrainerData();
      const cur = (n.data ?? {}) as Partial<TrainerNodeData>;
      const merged = { ...def, ...cur } as TrainerNodeData;
      if (patch === null) {
        const { trainUi: _t, ...rest } = merged;
        return { ...n, data: rest };
      }
      const prev = merged.trainUi ?? emptyTrainerTrainUi();
      return { ...n, data: { ...merged, trainUi: { ...prev, ...patch, active: true } } };
    }),
  );
}
