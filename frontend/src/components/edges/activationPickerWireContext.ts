import { createContext, useContext } from "react";

export type ActivationPickerWireEditorContextValue = {
  nameDraft: string;
  setNameDraft: (v: string) => void;
  /** Add a new pick or update the existing pick for this wire; Enter triggers when allowed. */
  commitWirePick: () => void;
  canCommitWirePick: boolean;
  /** When true, show the text input; when false, show the label (after Enter commit or blur). */
  nameEditActive: boolean;
  /** Leave edit mode (show label) without committing. */
  exitNameEditMode: () => void;
  /** Shown in the label when draft and saved label are both empty. */
  defaultNameDisplay: string;
  /** Bumps when the wire name field should receive focus (e.g. double-click gauge). */
  wireNameFocusNonce: number;
  /** Enter edit mode and focus the name field for the selected wire (no-op if a different edge id is passed). */
  openWireEditorForEdge: (edgeId: string) => void;
  /** Remove the saved pick for this wire (if any). Clears picker selection when it is the selected edge. */
  removePickForEdge: (edgeId: string, afterModuleIndexFromEdge?: number | null) => void;
};

export const ActivationPickerWireEditorContext = createContext<ActivationPickerWireEditorContextValue | null>(null);

export function useActivationPickerWireEditor(): ActivationPickerWireEditorContextValue | null {
  return useContext(ActivationPickerWireEditorContext);
}
