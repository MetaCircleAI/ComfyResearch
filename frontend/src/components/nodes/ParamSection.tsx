/**
 * Collapsible parameter section for dense nodes (UI redesign, task #99).
 * Studio theme: an eyebrow header with chevron that collapses its children
 * (progressive disclosure — all params stay reachable, nothing is removed).
 * Classic theme: renders children directly, preserving the legacy DOM.
 */
import { useState, type ReactNode } from "react";
import { useOptionalTheme } from "../../themeContext";
import { normalizeTheme } from "../../theme";

export function ParamSection({
  title,
  defaultOpen = true,
  forceOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  /** Keeps the section open regardless of toggle state — used when it
   * contains live status/error output that must stay visible. */
  forceOpen?: boolean;
  children: ReactNode;
}) {
  const contextTheme = useOptionalTheme()?.theme;
  const isStudio =
    (contextTheme ?? normalizeTheme(document.documentElement.dataset.crTheme)) !== "classic";
  const [open, setOpen] = useState(defaultOpen);
  const effectiveOpen = open || forceOpen;
  if (!isStudio) return <>{children}</>;
  return (
    <section className="cr-param-section">
      <button
        type="button"
        className="cr-param-section__head nodrag nopan"
        aria-expanded={effectiveOpen}
        onClick={() => setOpen((v) => (forceOpen ? true : !v))}
      >
        <span className="cr-param-section__title">{title}</span>
        <span className="cr-param-section__chevron" aria-hidden>
          {effectiveOpen ? "▾" : "▸"}
        </span>
      </button>
      {effectiveOpen ? <div className="cr-param-section__body">{children}</div> : null}
    </section>
  );
}
