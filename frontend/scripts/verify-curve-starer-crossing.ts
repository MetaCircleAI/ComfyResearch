import {
  analyzeTargetPhase,
  firstThresholdCrossingStep,
  measuredCurvePoints,
} from "../src/curveStarer/targetPhaseTransition";
import type { CurveStarerAnalyzedEntry } from "../src/curveStarer/lpdTypes";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

/** Rising accuracy: raw crosses @450; LPD-smoothed fit crosses later @480. */
const rawPoints = Array.from({ length: 50 }, (_, i) => {
  const t = 400 + i * 2;
  const loss = 0.94 + (i / 49) * 0.02;
  return { t, loss };
});

const lpdFit = rawPoints.map((p) => ({
  t: p.t,
  loss: p.t < 480 ? 0.948 : 0.952,
}));

const entry: CurveStarerAnalyzedEntry = {
  entryId: "viz:test",
  nodeId: "viz",
  seriesId: "test",
  label: "test accuracy",
  yAxisLabel: "acc",
  points: rawPoints,
  lpd: { data: lpdFit },
};

const threshold = 0.95;
const rawCross = firstThresholdCrossingStep(measuredCurvePoints(entry), "higher", threshold);
const lpdCross = firstThresholdCrossingStep(lpdFit, "higher", threshold);
assert(rawCross === 450, `expected raw crossing @450, got ${rawCross}`);
assert(lpdCross === 480, `expected lpd crossing @480, got ${lpdCross}`);

const analysis = analyzeTargetPhase(entry, "higher", threshold);
assert(
  analysis.crossingStep === 450,
  `analyzeTargetPhase must use measured points (got ${analysis.crossingStep}, lpd would be ${lpdCross})`,
);

console.log("verify-curve-starer-crossing: ok");
