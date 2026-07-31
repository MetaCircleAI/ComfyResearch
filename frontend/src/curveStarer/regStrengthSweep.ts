import { isTrickWin } from "./evaluateSpeedUpTrick";
import { TRY_REG_STRENGTHS } from "./speedUpTrickRegistry";
import type { SpeedUpTrickKind, SpeedUpTrickProposal, TrickTestResult, TrickTypeCategory } from "./speedUpTrickTypes";

export function trickKindUsesRegStrength(kind: SpeedUpTrickKind): boolean {
  return kind === "l2_reg_shell" || kind === "l1_reg_shell";
}

export const DEFAULT_TRY_STRENGTHS_TEXT = `{${TRY_REG_STRENGTHS.join(", ")}}`;

/** Parse comma/space-separated strengths; optional `{…}` wrapper; supports `1e-5` notation. */
export function parseTryStrengthsText(text: string): number[] {
  const stripped = text.trim().replace(/^\{|\}$/g, "").trim();
  if (!stripped) return [...TRY_REG_STRENGTHS];
  const parsed = stripped
    .split(/[,，]+/)
    .map((part) => Number.parseFloat(part.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  return parsed.length > 0 ? parsed : [...TRY_REG_STRENGTHS];
}

/** Whether this proposal should sweep over multiple reg strengths. */
export function proposalUsesRegStrengthSweep(
  proposal: SpeedUpTrickProposal | null,
  trickCategory?: TrickTypeCategory,
): boolean {
  if (!proposal) return false;
  const category = trickCategory ?? proposal.trickCategory;
  if (category !== "reg") return false;
  return trickKindUsesRegStrength(proposal.trickKind) || proposal.params.lossScale != null;
}

/** Strength values to try in order for L1/L2 reg tricks; `[]` means a single run (no sweep). */
export function regStrengthsToTry(
  proposal: SpeedUpTrickProposal | null,
  tryStrengths?: readonly number[],
  trickCategory?: TrickTypeCategory,
): readonly number[] {
  if (!proposalUsesRegStrengthSweep(proposal, trickCategory)) return [];
  return tryStrengths && tryStrengths.length > 0 ? [...tryStrengths] : [...TRY_REG_STRENGTHS];
}

/** Put the UI Strength first, then Try Strengths fallbacks (deduped, order preserved). */
export function mergePrimaryStrength(
  primary: number | null | undefined,
  strengths: readonly number[],
): number[] {
  const out: number[] = [];
  const push = (v: number) => {
    if (!Number.isFinite(v) || v <= 0) return;
    if (!out.some((x) => x === v)) out.push(v);
  };
  if (primary != null) push(primary);
  for (const s of strengths) push(s);
  return out.length > 0 ? out : [...strengths];
}

export function regStrengthAttemptsForBatch(
  proposal: SpeedUpTrickProposal | null,
  tryStrengthsText: string,
  trickCategory: TrickTypeCategory,
): number[] {
  const parsed = parseTryStrengthsText(tryStrengthsText);
  const base = [...regStrengthsToTry(proposal, parsed, trickCategory)];
  if (base.length === 0 && trickCategory === "reg") return parsed;
  return base;
}

export function formatRegStrengthLabel(strength: number | null): string {
  if (strength == null) return "";
  if (strength >= 0.001) return String(strength);
  return strength.toExponential(0).replace("+", "");
}

export type StrengthSweepAttemptContext = {
  strength: number | null;
  attempt: number;
  total: number;
};

/** Try strengths in order; stop on success. On failure, continue until exhausted. */
export async function runTrickStrengthSweep<TProposal>(args: {
  proposal: TProposal;
  strengths: readonly number[];
  applyStrength: (proposal: TProposal, strength: number) => TProposal;
  runTest: (proposal: TProposal) => Promise<TrickTestResult>;
  onAttemptStart?: (ctx: StrengthSweepAttemptContext) => void;
}): Promise<{ result: TrickTestResult; winningStrength: number | null; triedStrengths: number[] }> {
  const triedStrengths: number[] = [];
  let lastResult: TrickTestResult | null = null;
  const total = args.strengths.length;

  for (let i = 0; i < args.strengths.length; i++) {
    const strength = args.strengths[i]!;
    triedStrengths.push(strength);
    args.onAttemptStart?.({ strength, attempt: i + 1, total });

    const testProposal = args.applyStrength(args.proposal, strength);
    const result = await args.runTest(testProposal);
    lastResult = result;

    if (isTrickWin(result.verdict)) {
      return { result, winningStrength: strength, triedStrengths };
    }
    if (result.verdict === "error") {
      return { result, winningStrength: null, triedStrengths };
    }
  }

  let result = lastResult!;
  if (triedStrengths.length > 1) {
    const strengthNote = `Tried strengths: ${triedStrengths.map(formatRegStrengthLabel).join(", ")}.`;
    result = { ...result, message: `${result.message} ${strengthNote}` };
  }
  return { result, winningStrength: null, triedStrengths };
}
