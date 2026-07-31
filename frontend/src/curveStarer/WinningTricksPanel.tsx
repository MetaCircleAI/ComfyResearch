import type { SpeedUpTrickKind, WinningTrickRecord } from "./speedUpTrickTypes";

function trickKindLabel(kind: SpeedUpTrickKind): string {
  switch (kind) {
    case "l2_reg_shell":
      return "Reg (L2)";
    case "l1_reg_shell":
      return "Reg (L1)";
    case "l2_projection_shell":
      return "Projection (L2)";
    case "grad_clip_shell":
      return "Grad clip";
    default:
      return kind;
  }
}

function formatStep(step: number | null): string {
  return step != null ? String(step) : "—";
}

export function WinningTricksTable({ tricks }: { tricks: WinningTrickRecord[] }) {
  return (
    <div className="cr-correlation-finder__winning-panel nodrag nopan" role="region" aria-label="Winning tricks">
      <table className="cr-correlation-finder__winning-table">
        <thead>
          <tr>
            <th>Observable</th>
            <th>Trick</th>
            <th>Target</th>
            <th>Strength</th>
            <th>R²</th>
            <th>Baseline @</th>
            <th>Trick @</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {tricks.map((row) => (
            <tr key={`${row.matchEntryId}:${row.trickCategory}:${row.trickKind}`}>
              <td className="cr-correlation-finder__winning-obs">{row.observableLabel}</td>
              <td>{trickKindLabel(row.trickKind)}</td>
              <td>{row.targetValue.toPrecision(4)}</td>
              <td>{row.strength != null ? row.strength : "—"}</td>
              <td>{row.correlationScore.toFixed(2)}</td>
              <td>{formatStep(row.baselineCrossingStep)}</td>
              <td>{formatStep(row.trickCrossingStep)}</td>
              <td className="cr-correlation-finder__winning-msg">{row.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function WinningTricksPanel({
  tricks,
  open,
  onToggle,
}: {
  tricks: WinningTrickRecord[];
  open: boolean;
  onToggle: () => void;
  onClose?: () => void;
}) {
  return (
    <button
      type="button"
      className="cr-modal__btn cr-modal__btn--ghost cr-correlation-finder__winning-btn"
      disabled={tricks.length === 0}
      aria-expanded={open}
      onClick={onToggle}
    >
      Winning tricks ({tricks.length})
    </button>
  );
}
