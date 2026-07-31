import { buildProposalWithKind, TRY_REG_STRENGTHS } from "../src/curveStarer/speedUpTrickRegistry";
import { isTrickWin } from "../src/curveStarer/evaluateSpeedUpTrick";
import {
  mergePrimaryStrength,
  parseTryStrengthsText,
  proposalUsesRegStrengthSweep,
  regStrengthAttemptsForBatch,
  regStrengthsToTry,
  runTrickStrengthSweep,
  trickKindUsesRegStrength,
} from "../src/curveStarer/regStrengthSweep";
import type { RelatedObservableMatch, TrickTestResult } from "../src/curveStarer/speedUpTrickTypes";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

assert(
  parseTryStrengthsText("{0.0001, 0.00001, 0.000001}").join(",") === "0.0001,0.00001,0.000001",
  "parse brace-wrapped strengths",
);
assert(parseTryStrengthsText("1e-4, 1e-5, 1e-6").length === 3, "parse scientific notation");
assert(parseTryStrengthsText("").length === 3, "empty falls back to defaults");

const DEFAULT_TRY_STRENGTHS_TEXT = `{${TRY_REG_STRENGTHS.join(", ")}}`;

const l2Match = {
  entryId: "obs:train",
  entry: {
    entryId: "obs:train",
    nodeId: "obs",
    seriesId: "train",
    label: "body.0.output · l2_norm(flat)",
    yAxisLabel: "norm",
    points: [],
    lpd: null,
  },
  label: "body.0.output · l2_norm(flat)",
  correlationScore: 1,
  alignmentScore: 1,
  directionRelation: "opposite" as const,
  shellValue: 296,
  source: "algebra_user" as const,
  reductionOps: ["l2_norm"],
  hasAutomatedTrick: true,
  trickKinds: ["l2_reg_shell", "l2_projection_shell"] as const,
  defaultTrickCategory: "reg" as const,
  supportsProjection: true,
} satisfies RelatedObservableMatch;

const l2RegProposal = buildProposalWithKind(l2Match, "l2_reg_shell");
assert(trickKindUsesRegStrength(l2RegProposal.trickKind), "l2 reg uses strength");
assert(proposalUsesRegStrengthSweep(l2RegProposal, "reg"), "l2 reg sweeps when category reg");
assert(
  regStrengthsToTry(l2RegProposal, undefined, "reg").length === 3,
  `expected 3 strengths, got ${regStrengthsToTry(l2RegProposal, undefined, "reg").length}`,
);

assert(
  regStrengthsToTry(l2RegProposal, parseTryStrengthsText("0.001, 0.0001"), "reg").length === 2,
  "custom strengths override defaults",
);

const projectionProposal = buildProposalWithKind(l2Match, "l2_projection_shell");
assert(
  regStrengthsToTry(projectionProposal, undefined, "reg").length === 0,
  "projection should not sweep strengths",
);

const batchAttempts = regStrengthAttemptsForBatch(l2RegProposal, DEFAULT_TRY_STRENGTHS_TEXT, "reg");
assert(batchAttempts.join(",") === "0.0001,0.00001,0.000001", "batch uses Try Strengths only");
assert(
  mergePrimaryStrength(0.01, batchAttempts).join(",") === "0.01,0.0001,0.00001,0.000001",
  "mergePrimaryStrength helper still prepends a primary when needed",
);

const failureReach: TrickTestResult = {
  verdict: "failure_reach",
  message: "Failure — baseline reached threshold @ 1230; trick did not",
  baseline: { crossingStep: 1230, peakValue: 1, objective: "higher", direction: 1, tStart: 0, tEnd: 0 },
  trick: { crossingStep: null, peakValue: 0, objective: "higher", direction: 1, tStart: 0, tEnd: 0 },
  baselineCurve: [],
  trickCurve: [],
};

let calls = 0;
void (async () => {
  const { result, triedStrengths } = await runTrickStrengthSweep({
    proposal: l2RegProposal,
    strengths: [0.0001, 0.00001, 0.000001],
    applyStrength: (p, strength) => ({ ...p, params: { ...p.params, lossScale: strength } }),
    runTest: async () => {
      calls += 1;
      return failureReach;
    },
  });
  assert(calls === 3, `expected 3 sweep attempts, got ${calls}`);
  assert(triedStrengths.length === 3, "records all tried strengths");
  assert(result.message.includes("Tried strengths"), "exhausted sweep appends strength note");
  assert(!isTrickWin(result.verdict), "failure_reach is not a win");

  const { winningStrength, triedStrengths: winTried } = await runTrickStrengthSweep({
    proposal: l2RegProposal,
    strengths: [0.0001, 0.00001],
    applyStrength: (p, strength) => ({ ...p, params: { ...p.params, lossScale: strength } }),
    runTest: async (p) =>
      p.params.lossScale === 0.00001
        ? { ...failureReach, verdict: "success", message: "Success" }
        : failureReach,
  });
  assert(winningStrength === 0.00001, "stops at first winning strength");
  assert(winTried.length === 2, "tried first then second");

  console.log("regStrengthSweep: ok");
})();
