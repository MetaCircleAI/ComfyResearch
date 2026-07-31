import { useCallback, useEffect, useMemo, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { DiscreteMultiSelect } from "../components/nodes/DiscreteMultiSelect";
import { correlateObservables, findTargetMonotonicPhaseForEntry } from "./observableCorrelation";
import {
  buildProposalFromMatch,
  DEFAULT_REG_LOSS_SCALE,
  trickKindForCategory,
} from "./speedUpTrickRegistry";
import { analyzeTargetPhase, measuredCurvePoints } from "./targetPhaseTransition";
import { runSpeedUpTrickTest } from "./runSpeedUpTrickTest";
import { isTrickWin } from "./evaluateSpeedUpTrick";
import {
  DEFAULT_TRY_STRENGTHS_TEXT,
  formatRegStrengthLabel,
  regStrengthAttemptsForBatch,
  runTrickStrengthSweep,
  trickKindUsesRegStrength,
} from "./regStrengthSweep";
import { TryStrengthsInput } from "./TryStrengthsInput";
import { TrickTargetComparisonChart } from "./TrickTargetComparisonChart";
import { TrickTestProgressBar } from "./TrickTestProgressBar";
import { WinningTricksPanel, WinningTricksTable } from "./WinningTricksPanel";
import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import type {
  CurveStarerTargetConfig,
  RelatedObservableMatch,
  SpeedUpTrickProposal,
  TrickTestResult,
  TrickTestRunState,
  TrickTypeCategory,
  WinningTrickRecord,
} from "./speedUpTrickTypes";

const TRICK_TYPE_OPTIONS = [
  { id: "reg", label: "Reg" },
  { id: "projection", label: "Projection" },
] as const;

type CorrelationRowState = {
  match: RelatedObservableMatch;
  trickCategory: TrickTypeCategory;
  targetValue: number;
  strength: number | null;
};

function verdictClass(verdict: TrickTestRunState): string {
  switch (verdict) {
    case "success":
    case "success_reach":
      return "cr-speed-up-tricks__verdict--success";
    case "failure":
    case "failure_reach":
      return "cr-speed-up-tricks__verdict--failure";
    case "marginal":
      return "cr-speed-up-tricks__verdict--marginal";
    case "inconclusive":
      return "cr-speed-up-tricks__verdict--inconclusive";
    case "error":
      return "cr-speed-up-tricks__verdict--error";
    case "running":
      return "cr-speed-up-tricks__verdict--running";
    default:
      return "";
  }
}

function rowToProposal(row: CorrelationRowState): SpeedUpTrickProposal | null {
  const kind = trickKindForCategory(row.match.entry, row.trickCategory);
  if (!kind) return null;
  const base = buildProposalFromMatch(row.match, row.trickCategory);
  if (!base) return null;
  const params = { shellRadius: row.targetValue };
  if (trickKindUsesRegStrength(base.trickKind)) {
    return {
      ...base,
      params: {
        ...params,
        lossScale: row.strength ?? base.params.lossScale ?? DEFAULT_REG_LOSS_SCALE,
      },
    };
  }
  return { ...base, params };
}

function initialRows(matches: RelatedObservableMatch[]): CorrelationRowState[] {
  return matches.map((match) => {
    const category: TrickTypeCategory = "reg";
    const proposal = buildProposalFromMatch(match, category);
    return {
      match,
      trickCategory: category,
      targetValue: proposal?.params.shellRadius ?? match.shellValue,
      strength: category === "reg" ? (proposal?.params.lossScale ?? DEFAULT_REG_LOSS_SCALE) : null,
    };
  });
}

function winningRecordFrom(
  row: CorrelationRowState,
  proposal: SpeedUpTrickProposal,
  result: TrickTestResult,
): WinningTrickRecord {
  return {
    matchEntryId: row.match.entryId,
    observableLabel: row.match.label,
    trickKind: proposal.trickKind,
    trickCategory: row.trickCategory,
    targetValue: row.targetValue,
    strength: row.strength,
    correlationScore: row.match.correlationScore,
    verdict: result.verdict,
    message: result.message,
    baselineCrossingStep: result.baseline.crossingStep,
    trickCrossingStep: result.trick.crossingStep,
  };
}

function upsertWinningTrick(prev: WinningTrickRecord[], record: WinningTrickRecord): WinningTrickRecord[] {
  const idx = prev.findIndex(
    (w) => w.matchEntryId === record.matchEntryId && w.trickCategory === record.trickCategory,
  );
  if (idx < 0) return [...prev, record];
  const existing = prev[idx]!;
  const existingStep = existing.trickCrossingStep ?? Number.POSITIVE_INFINITY;
  const newStep = record.trickCrossingStep ?? Number.POSITIVE_INFINITY;
  if (newStep <= existingStep) {
    const next = prev.slice();
    next[idx] = record;
    return next;
  }
  return prev;
}

export function CorrelationFinderPanel({
  entries,
  targetConfig,
  nodes,
  edges,
  staring,
  stareRequested,
  onStareComplete,
}: {
  entries: CurveStarerAnalyzedEntry[];
  targetConfig: CurveStarerTargetConfig;
  nodes: Node[];
  edges: Edge[];
  staring: boolean;
  stareRequested: number;
  onStareComplete: () => void;
}) {
  const targetEntry = useMemo(
    () => entries.find((e) => e.entryId === targetConfig.entryId) ?? null,
    [entries, targetConfig.entryId],
  );

  const baselineAnalysis = useMemo(() => {
    if (!targetEntry) return null;
    return analyzeTargetPhase(targetEntry, targetConfig.objective, targetConfig.threshold);
  }, [targetEntry, targetConfig]);

  const targetPhase = useMemo(() => {
    if (!targetEntry) return null;
    return findTargetMonotonicPhaseForEntry(targetEntry, targetConfig.objective);
  }, [targetEntry, targetConfig.objective]);

  const matches = useMemo(() => {
    if (!targetEntry) return [];
    return correlateObservables(targetEntry, entries, targetConfig.objective);
  }, [entries, targetEntry, targetConfig.objective]);

  const [rows, setRows] = useState<CorrelationRowState[]>([]);
  const [runState, setRunState] = useState<Record<string, TrickTestRunState>>({});
  const [testProgress, setTestProgress] = useState<Record<string, { pct: number; step: number; total: number }>>({});
  const [testStrengthHint, setTestStrengthHint] = useState<Record<string, string>>({});
  const [results, setResults] = useState<Record<string, TrickTestResult>>({});
  const [testingAll, setTestingAll] = useState(false);
  const [tryStrengthsText, setTryStrengthsText] = useState(DEFAULT_TRY_STRENGTHS_TEXT);
  const [hasFound, setHasFound] = useState(false);
  const [winningTricks, setWinningTricks] = useState<WinningTrickRecord[]>([]);
  const [winningOpen, setWinningOpen] = useState(false);

  useEffect(() => {
    setHasFound(false);
    setRows([]);
    setRunState({});
    setTestProgress({});
    setTestStrengthHint({});
    setResults({});
    setWinningTricks([]);
    setWinningOpen(false);
  }, [targetConfig.entryId, targetConfig.objective]);

  useEffect(() => {
    if (stareRequested === 0) return;
    setRows(initialRows(matches));
    setRunState({});
    setTestProgress({});
    setTestStrengthHint({});
    setResults({});
    setWinningTricks([]);
    setWinningOpen(false);
    setHasFound(true);
    onStareComplete();
  }, [stareRequested, matches, onStareComplete]);

  const patchRow = useCallback((entryId: string, patch: Partial<CorrelationRowState>) => {
    setRows((prev) =>
      prev.map((row) => (row.match.entryId === entryId ? { ...row, ...patch } : row)),
    );
  }, []);

  const testOne = useCallback(
    async (row: CorrelationRowState, sweepStrengths = false): Promise<TrickTestResult | null> => {
      if (!targetEntry || !baselineAnalysis) return null;
      const entryId = row.match.entryId;
      const seedProposal = rowToProposal(row);
      if (!seedProposal) return null;
      const strengths = sweepStrengths
        ? regStrengthAttemptsForBatch(seedProposal, tryStrengthsText, row.trickCategory)
        : [];

      setRunState((s) => ({ ...s, [entryId]: "running" }));
      setTestProgress((p) => ({ ...p, [entryId]: { pct: 0, step: 0, total: 0 } }));
      setResults((r) => {
        const next = { ...r };
        delete next[entryId];
        return next;
      });

      const runOne = async (proposal: SpeedUpTrickProposal): Promise<TrickTestResult> =>
        runSpeedUpTrickTest({
          nodes,
          edges,
          proposal,
          targetConfig,
          baselineEntry: targetEntry,
          baselineAnalysis,
          skipExtraObservables: true,
          onProgress: (pct, step, total) => {
            setTestProgress((p) => ({ ...p, [entryId]: { pct, step, total } }));
          },
        });

      try {
        if (strengths.length === 0) {
          const result = await runOne(seedProposal);
          setResults((r) => ({ ...r, [entryId]: result }));
          setRunState((s) => ({ ...s, [entryId]: result.verdict }));
          if (isTrickWin(result.verdict)) {
            setWinningTricks((prev) =>
              upsertWinningTrick(prev, winningRecordFrom(row, seedProposal, result)),
            );
          }
          return result;
        }

        const { result, winningStrength } = await runTrickStrengthSweep({
          proposal: seedProposal,
          strengths,
          applyStrength: (baseProposal, strength) => ({
            ...baseProposal,
            params: { ...baseProposal.params, lossScale: strength },
          }),
          runTest: runOne,
          onAttemptStart: ({ strength, attempt, total }) => {
            setRunState((s) => ({ ...s, [entryId]: "running" }));
            setTestProgress((p) => ({ ...p, [entryId]: { pct: 0, step: 0, total: 0 } }));
            setTestStrengthHint((h) => ({
              ...h,
              [entryId]: `${formatRegStrengthLabel(strength)} (${attempt}/${total})`,
            }));
          },
        });

        setResults((r) => ({ ...r, [entryId]: result }));
        setRunState((s) => ({ ...s, [entryId]: result.verdict }));

        if (winningStrength != null) {
          if (winningStrength !== row.strength) {
            patchRow(entryId, { strength: winningStrength });
          }
          const winRow = { ...row, strength: winningStrength };
          const winProposal = rowToProposal(winRow);
          if (winProposal) {
            setWinningTricks((prev) =>
              upsertWinningTrick(prev, winningRecordFrom(winRow, winProposal, result)),
            );
          }
        }

        return result;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setRunState((s) => ({ ...s, [entryId]: "error" }));
        setResults((r) => ({
          ...r,
          [entryId]: {
            verdict: "error",
            message: msg,
            baseline: baselineAnalysis,
            trick: baselineAnalysis,
            baselineCurve: measuredCurvePoints(targetEntry),
            trickCurve: [],
          },
        }));
        return null;
      } finally {
        setTestProgress((p) => {
          const next = { ...p };
          delete next[entryId];
          return next;
        });
        setTestStrengthHint((h) => {
          const next = { ...h };
          delete next[entryId];
          return next;
        });
      }
    },
    [baselineAnalysis, edges, nodes, patchRow, targetConfig, targetEntry, tryStrengthsText],
  );

  const testAll = useCallback(async () => {
    setTestingAll(true);
    for (const row of rows) {
      if (!row.match.hasAutomatedTrick) continue;
      await testOne(row, true);
    }
    setTestingAll(false);
  }, [rows, testOne]);

  const targetLabel = targetEntry?.label ?? targetConfig.entryId;

  const winCount = winningTricks.length;
  const automatedCount = rows.filter((r) => r.match.hasAutomatedTrick).length;

  if (!hasFound) {
    return (
      <div className="cr-correlation-finder">
        <p className="cr-curve-starer-modal__empty">
          {staring ? (
            <>Finding target monotonic phase and scoring observables…</>
          ) : (
            <>
              Set <strong>Target curve</strong> and <strong>Objective</strong> (Higher = rising phase, Lower =
              falling), then click <strong>Start Staring</strong> to rank observables by monotonicity R² in that
              window.
            </>
          )}
        </p>
      </div>
    );
  }

  if (!targetPhase) {
    return (
      <div className="cr-correlation-finder">
        <p className="cr-curve-starer-modal__empty">
          No clear {targetConfig.objective === "higher" ? "rising" : "falling"} phase on <strong>{targetLabel}</strong>
          for objective <strong>{targetConfig.objective === "higher" ? "Higher is better" : "Lower is better"}</strong>.
          Train longer, switch objective, or pick another target, then click <strong>Start Staring</strong> again.
        </p>
      </div>
    );
  }

  const phaseSummary = `step ${Math.round(targetPhase.tStart)}–${Math.round(targetPhase.tEnd)} ${targetPhase.direction > 0 ? "↑" : "↓"}`;

  return (
    <div className="cr-correlation-finder">
      <div className="cr-correlation-finder__summary">
        <span>
          Target: {targetLabel} · {phaseSummary} · ranked by observable monotonicity R² in window
        </span>
      </div>
      <div className="cr-correlation-finder__actions">
        {automatedCount > 0 ? (
          <>
            <button
              type="button"
              className="cr-modal__btn cr-modal__btn--primary"
              disabled={testingAll || automatedCount === 0}
              onClick={() => void testAll()}
            >
              {testingAll ? "Testing all…" : "Test all tricks"}
            </button>
            <TryStrengthsInput
              value={tryStrengthsText}
              onChange={setTryStrengthsText}
              disabled={testingAll}
            />
          </>
        ) : null}
        <WinningTricksPanel
          tricks={winningTricks}
          open={winningOpen}
          onToggle={() => setWinningOpen((o) => (winningTricks.length > 0 ? !o : false))}
        />
        {winCount > 0 ? <span>{winCount} win{winCount === 1 ? "" : "s"}</span> : null}
      </div>
      {winningOpen && winningTricks.length > 0 ? <WinningTricksTable tricks={winningTricks} /> : null}
      {rows.length === 0 ? (
        <p className="cr-curve-starer-modal__empty">
          No observables sufficiently monotonic in the target phase. Train longer or add more observables, then click{" "}
          <strong>Start Staring</strong> again.
        </p>
      ) : (
        <ul className="cr-correlation-finder__list">
          {rows.map((row) => {
            const state = runState[row.match.entryId] ?? "idle";
            const result = results[row.match.entryId];
            const trickOptions = row.match.supportsProjection
              ? [...TRICK_TYPE_OPTIONS]
              : TRICK_TYPE_OPTIONS.filter((o) => o.id === "reg");
            return (
              <li key={row.match.entryId} className="cr-correlation-finder__item">
                <div className="cr-correlation-finder__head">
                  <span className="cr-correlation-finder__title">{row.match.label}</span>
                  <span className="cr-correlation-finder__score" title="Monotonicity R² in target phase window (0–1)">
                    R² {row.match.correlationScore.toFixed(2)}
                  </span>
                  <span className="cr-correlation-finder__meta">{row.match.directionRelation}</span>
                </div>
                {row.match.hasAutomatedTrick ? (
                  <div className="cr-correlation-finder__params">
                    <DiscreteMultiSelect
                      label="Trick type"
                      options={trickOptions}
                      value={row.trickCategory}
                      singleSelect
                      matchModalInput
                      ariaLabel={`Trick type for ${row.match.label}`}
                      onCommit={(next) => {
                        const raw = Array.isArray(next) ? next[0] : next;
                        if (raw !== "reg" && raw !== "projection") return;
                        const category = raw as TrickTypeCategory;
                        const proposal = buildProposalFromMatch(row.match, category);
                        patchRow(row.match.entryId, {
                          trickCategory: category,
                          targetValue: proposal?.params.shellRadius ?? row.targetValue,
                          strength: category === "reg" ? (proposal?.params.lossScale ?? DEFAULT_REG_LOSS_SCALE) : null,
                        });
                      }}
                    />
                    <label className="cr-speed-up-tricks-list__field">
                      <span>Target</span>
                      <input
                        type="number"
                        className="cr-modal__input"
                        step="any"
                        value={row.targetValue}
                        onChange={(e) =>
                          patchRow(row.match.entryId, {
                            targetValue: Number.parseFloat(e.target.value) || 0,
                          })
                        }
                      />
                    </label>
                    {row.trickCategory === "reg" ? (
                      <label className="cr-speed-up-tricks-list__field">
                        <span>Strength</span>
                        <input
                          type="number"
                          className="cr-modal__input"
                          step="any"
                          value={row.strength ?? ""}
                          onChange={(e) =>
                            patchRow(row.match.entryId, {
                              strength: Number.parseFloat(e.target.value) || 0,
                            })
                          }
                        />
                      </label>
                    ) : null}
                    <button
                      type="button"
                      className="cr-modal__btn cr-modal__btn--ghost"
                      disabled={state === "running" || testingAll}
                      onClick={() => void testOne(row)}
                    >
                      {state === "running" ? "Testing…" : "Test trick"}
                    </button>
                  </div>
                ) : (
                  <p className="cr-correlation-finder__no-trick">
                    Correlated — no automated trick for this observable (e.g. entropy).
                  </p>
                )}
                {state === "running" && testProgress[row.match.entryId] ? (
                  <>
                    {testStrengthHint[row.match.entryId] ? (
                      <p className="cr-correlation-finder__strength-hint">
                        Strength {testStrengthHint[row.match.entryId]}
                      </p>
                    ) : null}
                    <TrickTestProgressBar
                      pct={testProgress[row.match.entryId]!.pct}
                      step={testProgress[row.match.entryId]!.step}
                      total={testProgress[row.match.entryId]!.total}
                    />
                  </>
                ) : null}
                {result ? (
                  <div className="cr-correlation-finder__result">
                    <p className={`cr-speed-up-tricks-list__verdict ${verdictClass(state)}`}>{result.message}</p>
                    {result.baselineCurve.length >= 2 || result.trickCurve.length >= 2 ? (
                      <TrickTargetComparisonChart
                        baselineCurve={result.baselineCurve}
                        trickCurve={result.trickCurve}
                        yAxisLabel={targetLabel}
                        threshold={targetConfig.threshold}
                        objective={targetConfig.objective}
                        baselineCrossingStep={result.baseline.crossingStep}
                        trickCrossingStep={result.trick.crossingStep}
                      />
                    ) : null}
                  </div>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
