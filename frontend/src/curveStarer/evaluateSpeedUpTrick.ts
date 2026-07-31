import type { TargetObjective, TargetPhaseAnalysis } from "./targetPhaseTransition";
import type { TrickTestEvaluation, TrickTestVerdict } from "./speedUpTrickTypes";

function isBetterPeak(trick: number, baseline: number, objective: TargetObjective): boolean {
  if (!Number.isFinite(trick) || !Number.isFinite(baseline)) return false;
  return objective === "higher" ? trick > baseline : trick < baseline;
}

export function evaluateSpeedUpTrick(
  baseline: TargetPhaseAnalysis,
  trick: TargetPhaseAnalysis,
): TrickTestEvaluation {
  const bStep = baseline.crossingStep;
  const tStep = trick.crossingStep;

  let verdict: TrickTestVerdict;
  let message: string;

  if (bStep != null && tStep != null) {
    if (tStep < bStep) {
      verdict = "success";
      message = `Success — crossed @ ${tStep} vs baseline ${bStep} (−${bStep - tStep} steps)`;
    } else {
      verdict = "failure";
      message = `Failure — crossed @ ${tStep} vs baseline ${bStep}`;
    }
  } else if (bStep == null && tStep != null) {
    verdict = "success_reach";
    message = `Success — trick reached threshold; baseline did not (trick @ ${tStep})`;
  } else if (bStep != null && tStep == null) {
    verdict = "failure_reach";
    message = `Failure — baseline reached threshold @ ${bStep}; trick did not`;
  } else if (isBetterPeak(trick.peakValue, baseline.peakValue, baseline.objective)) {
    verdict = "marginal";
    const delta = trick.peakValue - baseline.peakValue;
    message = `Marginal — neither crossed threshold (peaks ${baseline.peakValue.toFixed(3)} / ${trick.peakValue.toFixed(3)}, Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(3)})`;
  } else {
    verdict = "inconclusive";
    message = `Inconclusive — neither reached threshold (peaks ${baseline.peakValue.toFixed(3)} / ${trick.peakValue.toFixed(3)}). Lower threshold or extend training.`;
  }

  return { verdict, message, baseline, trick };
}

export function isTrickWin(verdict: TrickTestVerdict): boolean {
  return verdict === "success" || verdict === "success_reach";
}
