import { useEffect } from "react";

/**
 * Blurs a focused native `<select>` when the user pointer-downs outside it.
 * OS dropdowns often stay open when React Flow stops propagation on the pane; capture phase fixes that.
 */
export function useBlurOpenSelectOnOutsidePointer(): void {
  useEffect(() => {
    const onPointerDownCapture = (e: PointerEvent) => {
      const t = e.target;
      if (!(t instanceof Node)) return;
      const active = document.activeElement;
      if (!(active instanceof HTMLSelectElement)) return;
      if (active.contains(t)) return;
      active.blur();
    };
    document.addEventListener("pointerdown", onPointerDownCapture, true);
    return () => document.removeEventListener("pointerdown", onPointerDownCapture, true);
  }, []);
}
