import { describe, expect, it } from "vitest";
import { emptyTrainerTrainUi } from "../../components/nodes/trainerDefaults";
import { resolveTrainerTrainDisplay, type TrainerLocalTrainUi } from "../trainerTrainUiResolve";

const local: TrainerLocalTrainUi = {
  trainLoading: true,
  trainPaused: false,
  trainProgressPct: 0,
  trainSeriesBarPct: 0,
  trainSeriesDual: false,
  trainSeriesCaptionLines: null,
  trainPhaseText: "Remote: sending experiment graph...",
  trainError: null,
};

describe("resolveTrainerTrainDisplay", () => {
  it("does not let a persisted bootstrap phase override the live sending phase", () => {
    const persisted = {
      ...emptyTrainerTrainUi(),
      active: true,
      loading: true,
      phaseText: "Remote: bootstrapping runtime...",
    };
    expect(resolveTrainerTrainDisplay(local, persisted).phaseText).toBe(
      "Remote: sending experiment graph...",
    );
  });
});
