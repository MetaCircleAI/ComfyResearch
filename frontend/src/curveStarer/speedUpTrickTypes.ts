import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import type { CurvePoint } from "./observableCurvePayload";
import type { TargetObjective, TargetPhaseAnalysis } from "./targetPhaseTransition";

export type TrickTypeCategory = "reg" | "projection";

export type SpeedUpTrickKind =
  | "l2_reg_shell"
  | "l2_projection_shell"
  | "l1_reg_shell"
  | "grad_clip_shell";

export type DirectionRelation = "same" | "opposite" | "unknown";

export type ObservableSourceKind = "built_in" | "algebra_user" | "unknown";

export type SpeedUpTrickParams = {
  shellRadius: number;
  lossScale?: number;
};

export type RelatedObservableMatch = {
  entryId: string;
  entry: CurveStarerAnalyzedEntry;
  label: string;
  correlationScore: number;
  /** @deprecated use correlationScore */
  alignmentScore: number;
  directionRelation: DirectionRelation;
  shellValue: number;
  source: ObservableSourceKind;
  reductionOps: string[];
  hasAutomatedTrick: boolean;
  trickKinds: SpeedUpTrickKind[];
  defaultTrickCategory: TrickTypeCategory;
  supportsProjection: boolean;
};

export type SpeedUpTrickProposal = {
  id: string;
  matchEntryId: string;
  observableLabel: string;
  trickKind: SpeedUpTrickKind;
  trickCategory: TrickTypeCategory;
  params: SpeedUpTrickParams;
  correlationScore: number;
  /** @deprecated use correlationScore */
  alignmentScore: number;
  directionRelation: DirectionRelation;
  hasAutomatedTrick: boolean;
};

export type TrickTestVerdict =
  | "success"
  | "failure"
  | "success_reach"
  | "failure_reach"
  | "inconclusive"
  | "marginal"
  | "error";

export type TrickTestEvaluation = {
  verdict: TrickTestVerdict;
  message: string;
  baseline: TargetPhaseAnalysis;
  trick: TargetPhaseAnalysis;
};

export type TrickTestResult = TrickTestEvaluation & {
  baselineCurve: CurvePoint[];
  trickCurve: CurvePoint[];
};

export type CurveStarerTargetConfig = {
  entryId: string;
  objective: TargetObjective;
  threshold: number;
};

export type TrickTestRunState = "idle" | "running" | TrickTestVerdict;

export type WinningTrickRecord = {
  matchEntryId: string;
  observableLabel: string;
  trickKind: SpeedUpTrickKind;
  trickCategory: TrickTypeCategory;
  targetValue: number;
  strength: number | null;
  correlationScore: number;
  verdict: TrickTestVerdict;
  message: string;
  baselineCrossingStep: number | null;
  trickCrossingStep: number | null;
};
