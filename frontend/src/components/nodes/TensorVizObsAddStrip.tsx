/** Shared row: optional observable name + “Add obs” (below node title; avoids cramped header + width:100% inputs). */

type TensorVizObsAddStripProps = {
  nodeId: string;
  nameDraft: string;
  onNameChange: (v: string) => void;
  canAdd: boolean;
  busy: boolean;
  onAdd: () => void;
};

export function TensorVizObsAddStrip({
  nodeId,
  nameDraft,
  onNameChange,
  canAdd,
  busy,
  onAdd,
}: TensorVizObsAddStripProps) {
  return (
    <div className="cr-node__obs-add-strip nodrag nopan" aria-label="Add user observable">
      <input
        type="text"
        id={`obs-add-name-${nodeId}`}
        name={`obs-add-name-${nodeId}`}
        className="cr-node__obs-add-strip-input"
        placeholder="Name (optional)"
        value={nameDraft}
        onChange={(e) => onNameChange(e.target.value)}
        autoComplete="off"
      />
      <button
        type="button"
        className="cr-btn cr-btn--add-obs"
        disabled={!canAdd || busy}
        onClick={() => void onAdd()}
      >
        {busy ? "…" : "Add obs"}
      </button>
    </div>
  );
}
