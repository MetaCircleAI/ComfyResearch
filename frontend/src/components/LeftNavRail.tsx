import { useEffect, useRef, useState } from "react";
import type { RailPrimarySection } from "./railTypes";
import { useTheme } from "../themeContext";
import type { CrTheme } from "../theme";

type RailItem = {
  id: RailPrimarySection;
  label: string;
};

const primaryItems: RailItem[] = [
  { id: "nodes", label: "Nodes" },
  { id: "observables", label: "Observables" },
  { id: "templates", label: "Templates" },
];

const THEME_OPTIONS: { id: CrTheme; label: string }[] = [
  { id: "studio", label: "Studio" },
  { id: "paper", label: "Paper" },
  { id: "classic", label: "Classic" },
];

function RailIcon({ name }: { name: string }) {
  const common = { width: 20, height: 20, stroke: "currentColor", fill: "none", strokeWidth: 1.6 };
  switch (name) {
    case "nodes":
      return (
        <svg viewBox="0 0 24 24" aria-hidden {...common}>
          <rect x="4" y="4" width="16" height="6" rx="1.5" />
          <rect x="4" y="14" width="10" height="6" rx="1.5" />
        </svg>
      );
    case "observables":
      return (
        <svg viewBox="0 0 24 24" aria-hidden {...common}>
          <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none" />
          <path d="M12 4v3M12 17v3M4 12h3M17 12h3M6.3 6.3l2.1 2.1M15.6 15.6l2.1 2.1M17.7 6.3l-2.1 2.1M8.4 15.6l-2.1 2.1" strokeLinecap="round" />
        </svg>
      );
    case "templates":
      return (
        <svg viewBox="0 0 24 24" aria-hidden {...common}>
          <rect x="6" y="3" width="12" height="18" rx="1" />
          <path d="M9 7h6M9 11h6M9 15h4" strokeLinecap="round" />
        </svg>
      );
    case "settings":
      return (
        <svg viewBox="0 0 24 24" aria-hidden {...common}>
          <circle cx="12" cy="12" r="3" />
          <path
            d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"
            strokeLinecap="round"
          />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 24 24" aria-hidden {...common}>
          <circle cx="12" cy="12" r="8" />
        </svg>
      );
  }
}

export function LeftNavRail({
  activeSection,
  onSelectSection,
}: {
  activeSection: RailPrimarySection | null;
  onSelectSection: (section: RailPrimarySection) => void;
}) {
  const { theme, setTheme } = useTheme();
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const footerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!themeMenuOpen) return;
    const onPointerDown = (ev: PointerEvent) => {
      if (!footerRef.current?.contains(ev.target as Node)) setThemeMenuOpen(false);
    };
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setThemeMenuOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [themeMenuOpen]);

  return (
    <nav className="cr-rail" aria-label="Primary">
      {theme !== "classic" ? (
        <div className="cr-rail__brand" aria-hidden title="ComfyResearch">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
            <circle cx="6" cy="6" r="2.6" />
            <circle cx="18" cy="9" r="2.6" />
            <circle cx="10" cy="18" r="2.6" />
            <path d="M8.4 6.8 15.5 8.6M7 8.4l2.2 7.2M16.2 11l-4.3 5.2" />
          </svg>
        </div>
      ) : null}
      <div className="cr-rail__main">
        {primaryItems.map((item) => {
          const active = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`cr-rail__btn${active ? " cr-rail__btn--active" : ""}`}
              data-cr-rail-section={item.id}
              aria-current={active ? "page" : undefined}
              onClick={() => onSelectSection(item.id)}
            >
              <span className="cr-rail__btn-icon" aria-hidden>
                <RailIcon name={item.id} />
              </span>
              <span className="cr-rail__label">{item.label}</span>
            </button>
          );
        })}
      </div>
      <div className="cr-rail__footer" ref={footerRef}>
        {themeMenuOpen ? (
          <div className="cr-theme-menu" role="menu" aria-label="Theme">
            {THEME_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                role="menuitemradio"
                aria-checked={theme === opt.id}
                className={`cr-theme-menu__option${theme === opt.id ? " cr-theme-menu__option--active" : ""}`}
                onClick={() => {
                  setTheme(opt.id);
                  setThemeMenuOpen(false);
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        ) : null}
        <button
          type="button"
          className="cr-rail__btn"
          aria-label="Theme settings"
          aria-haspopup="menu"
          aria-expanded={themeMenuOpen}
          onClick={() => setThemeMenuOpen((v) => !v)}
        >
          <span className="cr-rail__btn-icon" aria-hidden>
            <RailIcon name="settings" />
          </span>
          <span className="cr-rail__label">Theme</span>
        </button>
      </div>
    </nav>
  );
}
