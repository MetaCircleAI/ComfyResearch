import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import type {
  RelatedObservableMatch,
  SpeedUpTrickKind,
  SpeedUpTrickParams,
  SpeedUpTrickProposal,
  TrickTypeCategory,
} from "./speedUpTrickTypes";
import { parseReductionOpsFromLabel } from "./observableCorrelation";
import type { TargetPhaseAnalysis } from "./targetPhaseTransition";

export const DEFAULT_REG_LOSS_SCALE = 0.0001;

/** Reg tricks: if the first strength fails to speed up, try weaker values in order. */
export const TRY_REG_STRENGTHS = [0.0001, 0.00001, 0.000001] as const;

export function inferTrickKindsForEntry(entry: CurveStarerAnalyzedEntry): SpeedUpTrickKind[] {
  const label = entry.label.toLowerCase();
  const ops = parseReductionOpsFromLabel(entry.label);
  const kinds: SpeedUpTrickKind[] = [];
  if (label.includes("weight l2") || ops.includes("l2_norm") || ops.includes("mean")) {
    kinds.push("l2_reg_shell", "l2_projection_shell");
  }
  if (label.includes("weight l1") || ops.includes("l1_norm")) {
    kinds.push("l1_reg_shell");
  }
  if (label.includes("gradient") || ops.includes("std")) {
    kinds.push("grad_clip_shell");
  }
  return kinds;
}

export function supportsProjectionForEntry(entry: CurveStarerAnalyzedEntry): boolean {
  const label = entry.label.toLowerCase();
  const ops = parseReductionOpsFromLabel(entry.label);
  if (ops.includes("entropy")) return false;
  if (label.includes("gradient")) return false;
  return (
    label.includes("weight l2") ||
    ops.includes("l2_norm") ||
    ops.includes("mean") ||
    ops.includes("median")
  );
}

/** Best automated trick for an observable (batch tests, defaults). Honors shell target when possible. */
export function primaryAutomatedTrickKind(entry: CurveStarerAnalyzedEntry): SpeedUpTrickKind | null {
  const kinds = inferTrickKindsForEntry(entry);
  if (kinds.length === 0) return null;
  const priority: SpeedUpTrickKind[] = [
    "grad_clip_shell",
    "l1_reg_shell",
    "l2_projection_shell",
    "l2_reg_shell",
  ];
  for (const kind of priority) {
    if (kinds.includes(kind)) return kind;
  }
  return kinds[0] ?? null;
}

export function trickCategoryForKind(kind: SpeedUpTrickKind): TrickTypeCategory {
  return kind.includes("projection") ? "projection" : "reg";
}

export function trickKindForCategory(
  entry: CurveStarerAnalyzedEntry,
  category: TrickTypeCategory,
): SpeedUpTrickKind | null {
  const kinds = inferTrickKindsForEntry(entry);
  if (kinds.length === 0) return null;
  if (category === "projection") {
    return kinds.find((k) => k.includes("projection")) ?? null;
  }
  const regPriority: SpeedUpTrickKind[] = ["grad_clip_shell", "l1_reg_shell", "l2_reg_shell"];
  for (const kind of regPriority) {
    if (kinds.includes(kind)) return kind;
  }
  return kinds.find((k) => !k.includes("projection")) ?? kinds[0] ?? null;
}

export function defaultParamsForTrick(
  kind: SpeedUpTrickKind,
  shellValue: number,
): SpeedUpTrickParams {
  const shellRadius = Number.isFinite(shellValue) ? shellValue : 1;
  if (kind === "l2_reg_shell" || kind === "l1_reg_shell") {
    const target = shellRadius > 0 ? shellRadius : Math.abs(shellRadius) || 1;
    return { shellRadius: target, lossScale: DEFAULT_REG_LOSS_SCALE };
  }
  return { shellRadius };
}

export function buildProposalFromMatch(
  match: RelatedObservableMatch,
  trickCategory: TrickTypeCategory,
): SpeedUpTrickProposal | null {
  const kind = trickKindForCategory(match.entry, trickCategory);
  if (!kind) return null;
  return buildProposalWithKind(match, kind);
}

export function buildProposalWithKind(
  match: RelatedObservableMatch,
  kind: SpeedUpTrickKind,
): SpeedUpTrickProposal {
  const params = defaultParamsForTrick(kind, match.shellValue);
  return {
    id: `${match.entryId}:${kind}`,
    matchEntryId: match.entryId,
    observableLabel: match.label,
    trickKind: kind,
    trickCategory: trickCategoryForKind(kind),
    params,
    correlationScore: match.correlationScore,
    alignmentScore: match.correlationScore,
    directionRelation: match.directionRelation,
    hasAutomatedTrick: true,
  };
}

export function proposeSpeedUpTricks(
  _target: TargetPhaseAnalysis,
  matches: RelatedObservableMatch[],
): SpeedUpTrickProposal[] {
  const proposals: SpeedUpTrickProposal[] = [];
  for (const match of matches) {
    if (!match.hasAutomatedTrick) continue;
    const category: TrickTypeCategory = "reg";
    const proposal = buildProposalFromMatch(match, category);
    if (proposal) proposals.push(proposal);
  }
  return proposals;
}
