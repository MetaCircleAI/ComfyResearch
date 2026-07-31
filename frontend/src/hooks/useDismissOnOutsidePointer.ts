import { type RefObject, useEffect, useRef } from "react";

/**
 * Closes floating UI when the user presses outside `containerRef`.
 * Uses **capture** phase so React Flow (and other code) cannot block the event via `stopPropagation`.
 */
export function useDismissOnOutsidePointer(
  open: boolean,
  onDismiss: () => void,
  containerRef: RefObject<HTMLElement | null>,
  options?: { closeOnEscape?: boolean },
): void {
  const closeOnEscape = options?.closeOnEscape ?? true;
  const dismissRef = useRef(onDismiss);
  dismissRef.current = onDismiss;

  useEffect(() => {
    if (!open) return;
    const outside = (e: Event) => {
      const t = e.target;
      if (!(t instanceof Node)) return;
      if (containerRef.current?.contains(t)) return;
      dismissRef.current();
    };
    const onKey = (e: KeyboardEvent) => {
      if (closeOnEscape && e.key === "Escape") dismissRef.current();
    };
    document.addEventListener("mousedown", outside, true);
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", outside, true);
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, containerRef, closeOnEscape]);
}
