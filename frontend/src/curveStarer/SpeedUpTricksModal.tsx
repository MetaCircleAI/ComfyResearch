import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { Edge, Node } from "@xyflow/react";
import { correlateObservables } from "./observableCorrelation";
import { proposeSpeedUpTricks } from "./speedUpTrickRegistry";
import {
  DEFAULT_TRY_STRENGTHS_TEXT,
  regStrengthAttemptsForBatch,
  runTrickStrengthSweep,
} from "./regStrengthSweep";
import { TryStrengthsInput } from "./TryStrengthsInput";
import { analyzeTargetPhase, measuredCurvePoints } from "./targetPhaseTransition";
import { runSpeedUpTrickTest } from "./runSpeedUpTrickTest";
import { isTrickWin } from "./evaluateSpeedUpTrick";
import { TrickTargetComparisonChart } from "./TrickTargetComparisonChart";
import { TrickTestProgressBar } from "./TrickTestProgressBar";
import type { CurveStarerAnalyzedEntry } from "./lpdTypes";
import type {
  CurveStarerTargetConfig,
  SpeedUpTrickProposal,
  TrickTestResult,
  TrickTestRunState,
} from "./speedUpTrickTypes";

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

export function SpeedUpTricksModal({
  open,
  entries,
  targetConfig,
  nodes,
  edges,
  onClose,
}: {
  open: boolean;
  entries: CurveStarerAnalyzedEntry[];
  targetConfig: CurveStarerTargetConfig;
  nodes: Node[];
  edges: Edge[];
  onClose: () => void;
}) {
  const targetEntry = useMemo(
    () => entries.find((e) => e.entryId === targetConfig.entryId) ?? null,
    [entries, targetConfig.entryId],
  );

  const baselineAnalysis = useMemo(() => {
    if (!targetEntry) return null;
    return analyzeTargetPhase(targetEntry, targetConfig.objective, targetConfig.threshold);
  }, [targetEntry, targetConfig]);

  const initialProposals = useMemo(() => {
    if (!targetEntry || !baselineAnalysis) return [];
    const matches = correlateObservables(targetEntry, entries, targetConfig.objective);
    return proposeSpeedUpTricks(baselineAnalysis, matches);
  }, [baselineAnalysis, entries, targetConfig.objective, targetEntry]);

  const [proposals, setProposals] = useState<SpeedUpTrickProposal[]>(initialProposals);
  const [runState, setRunState] = useState<Record<string, TrickTestRunState>>({});
  const [testProgress, setTestProgress] = useState<Record<string, { pct: number; step: number; total: number }>>({});
  const [results, setResults] = useState<Record<string, TrickTestResult>>({});
  const [testingAll, setTestingAll] = useState(false);
  const [tryStrengthsText, setTryStrengthsText] = useState(DEFAULT_TRY_STRENGTHS_TEXT);

  useEffect(() => {
    if (!open) return;
    setProposals(initialProposals);
    setRunState({});
    setTestProgress({});
    setResults({});
  }, [open, initialProposals]);

  const patchProposal = useCallback((id: string, patch: Partial<SpeedUpTrickProposal["params"]>) => {
    setProposals((prev) =>
      prev.map((p) => (p.id === id ? { ...p, params: { ...p.params, ...patch } } : p)),
    );
  }, []);

  const testOne = useCallback(
    async (proposal: SpeedUpTrickProposal, sweepStrengths = false) => {
      if (!targetEntry || !baselineAnalysis) return;
      const proposalId = proposal.id;
      const strengths = sweepStrengths
        ? regStrengthAttemptsForBatch(proposal, tryStrengthsText, proposal.trickCategory)
        : [];

      setRunState((s) => ({ ...s, [proposalId]: "running" }));
      setTestProgress((p) => ({ ...p, [proposalId]: { pct: 0, step: 0, total: 0 } }));
      setResults((r) => {
        const next = { ...r };
        delete next[proposalId];
        return next;
      });

      const runOne = async (testProposal: SpeedUpTrickProposal): Promise<TrickTestResult> =>
        runSpeedUpTrickTest({
          nodes,
          edges,
          proposal: testProposal,
          targetConfig,
          baselineEntry: targetEntry,
          baselineAnalysis,
          onProgress: (pct, step, total) => {
            setTestProgress((p) => ({ ...p, [proposalId]: { pct, step, total } }));
          },
        });

      try {
        if (strengths.length === 0) {
          const result = await runOne(proposal);
          setResults((r) => ({ ...r, [proposalId]: result }));
          setRunState((s) => ({ ...s, [proposalId]: result.verdict }));
          return;
        }

        const { result, winningStrength } = await runTrickStrengthSweep({
          proposal,
          strengths,
          applyStrength: (base, strength) => ({
            ...base,
            params: { ...base.params, lossScale: strength },
          }),
          runTest: runOne,
          onAttemptStart: ({ strength, attempt, total }) => {
            setRunState((s) => ({ ...s, [proposalId]: "running" }));
            setTestProgress((p) => ({ ...p, [proposalId]: { pct: 0, step: 0, total: 0 } }));
          },
        });

        setResults((r) => ({ ...r, [proposalId]: result }));
        setRunState((s) => ({ ...s, [proposalId]: result.verdict }));

        if (winningStrength != null && winningStrength !== proposal.params.lossScale) {
          patchProposal(proposalId, { lossScale: winningStrength });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setRunState((s) => ({ ...s, [proposalId]: "error" }));
        setResults((r) => ({
          ...r,
          [proposalId]: {
            verdict: "error",
            message: msg,
            baseline: baselineAnalysis,
            trick: baselineAnalysis,
            baselineCurve: measuredCurvePoints(targetEntry),
            trickCurve: [],
          },
        }));
      } finally {
        setTestProgress((p) => {
          const next = { ...p };
          delete next[proposalId];
          return next;
        });
      }
    },
    [baselineAnalysis, edges, nodes, patchProposal, targetConfig, targetEntry, tryStrengthsText],
  );

  const testAll = useCallback(async () => {
    setTestingAll(true);
    for (const p of proposals) {
      await testOne(p, true);
    }
    setTestingAll(false);
  }, [proposals, testOne]);

  if (!open) return null;

  const targetLabel = targetEntry?.label ?? targetConfig.entryId;
  const crossingLabel =
    baselineAnalysis?.crossingStep != null
      ? `@ step ${baselineAnalysis.crossingStep}`
      : "— no threshold crossing yet";

  const winCount = proposals.filter((p) => isTrickWin(runState[p.id] ?? "idle")).length;

  return createPortal(
    <div className="cr-modal-backdrop cr-speed-up-tricks-backdrop" onMouseDown={onClose}>
      <div
        className="cr-modal cr-speed-up-tricks-modal nodrag nopan"
        role="dialog"
        aria-labelledby="speed-up-tricks-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="cr-speed-up-tricks-modal__header">
          <div>
            <h2 id="speed-up-tricks-title" className="cr-modal__title">
              Propose speed up tricks
            </h2>
            <p className="cr-speed-up-tricks-modal__subtitle">
              Target: {targetLabel} · {targetConfig.objective} · threshold{" "}
              {targetConfig.objective === "higher" ? "≥" : "≤"} {targetConfig.threshold}{" "}
              {crossingLabel}
            </p>
          </div>
          <div className="cr-speed-up-tricks-modal__header-actions">
            <div className="cr-speed-up-tricks-modal__test-all">
              <button
                type="button"
                className="cr-modal__btn cr-modal__btn--primary"
                disabled={testingAll || proposals.length === 0}
                onClick={() => void testAll()}
              >
                {testingAll ? "Testing all…" : "Test All Tricks"}
              </button>
              <TryStrengthsInput
                value={tryStrengthsText}
                onChange={setTryStrengthsText}
                disabled={testingAll}
              />
            </div>
            <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onClose}>
              Close
            </button>
          </div>
        </header>
        <div className="cr-speed-up-tricks-modal__summary">
          {proposals.length} trick{proposals.length === 1 ? "" : "s"}
          {winCount > 0 ? ` · ${winCount} win${winCount === 1 ? "" : "s"}` : ""}
          {baselineAnalysis?.crossingStep == null ? (
            <span className="cr-speed-up-tricks-modal__warn">
              Target has not crossed threshold — correlation may be unreliable.
            </span>
          ) : null}
        </div>
        <div className="cr-speed-up-tricks-modal__body">
          {proposals.length === 0 ? (
            <p className="cr-speed-up-tricks-modal__empty">
              No correlated observables with automated tricks. Adjust Target / threshold or train longer,
              then reopen CurveStarer.
            </p>
          ) : (
            <ul className="cr-speed-up-tricks-list">
              {proposals.map((p) => {
                const state = runState[p.id] ?? "idle";
                const result = results[p.id];
                return (
                  <li key={p.id} className="cr-speed-up-tricks-list__item">
                    <div className="cr-speed-up-tricks-list__head">
                      <span className="cr-speed-up-tricks-list__title">{p.observableLabel}</span>
                      <span className="cr-speed-up-tricks-list__meta">
                        {p.directionRelation} · alignment {p.alignmentScore.toFixed(2)}
                      </span>
                    </div>
                    <div className="cr-speed-up-tricks-list__params">
                      <label className="cr-speed-up-tricks-list__field">
                        <span>shell</span>
                        <input
                          type="number"
                          className="cr-modal__input"
                          step="any"
                          value={p.params.shellRadius}
                          onChange={(e) =>
                            patchProposal(p.id, { shellRadius: Number.parseFloat(e.target.value) || 0 })
                          }
                        />
                      </label>
                      {p.params.lossScale != null ? (
                        <label className="cr-speed-up-tricks-list__field">
                          <span>strength</span>
                          <input
                            type="number"
                            className="cr-modal__input"
                            step="any"
                            value={p.params.lossScale}
                            onChange={(e) =>
                              patchProposal(p.id, { lossScale: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </label>
                      ) : null}
                      <button
                        type="button"
                        className="cr-modal__btn cr-modal__btn--ghost"
                        disabled={state === "running" || testingAll}
                        onClick={() => void testOne(p)}
                      >
                        {state === "running" ? "Testing…" : "Test trick"}
                      </button>
                    </div>
                    {state === "running" && testProgress[p.id] ? (
                      <TrickTestProgressBar
                        pct={testProgress[p.id]!.pct}
                        step={testProgress[p.id]!.step}
                        total={testProgress[p.id]!.total}
                      />
                    ) : null}
                    {result ? (
                      <div className="cr-correlation-finder__result">
                        <p className={`cr-speed-up-tricks-list__verdict ${verdictClass(state)}`}>
                          {result.message}
                        </p>
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
      </div>
    </div>,
    document.body,
  );
}
