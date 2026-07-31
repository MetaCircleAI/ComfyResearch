export function TrickTestProgressBar({
  pct,
  step,
  total,
}: {
  pct: number;
  step?: number;
  total?: number;
}) {
  const clamped = Math.min(100, Math.max(0, Math.round(pct)));
  const detail =
    step != null && total != null && total > 0 ? ` · step ${step}/${total}` : "";
  return (
    <div
      className="cr-trick-test-progress"
      role="progressbar"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Trick test training ${clamped}%${detail}`}
    >
      <div className="cr-trick-test-progress__fill" style={{ width: `${clamped}%` }} />
      <span className="cr-trick-test-progress__label">
        Training {clamped}%{detail}
      </span>
    </div>
  );
}
