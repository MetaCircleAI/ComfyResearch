/**
 * Reactive theme source of truth for components (spec §6). PR-2 param
 * primitives read useTheme() to pick studio vs classic control forms.
 */
import {
  createContext,
  useCallback,
  useContext,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { applyTheme, persistTheme, readStoredTheme, type CrTheme } from "./theme";

type ThemeContextValue = { theme: CrTheme; setTheme: (theme: CrTheme) => void };

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<CrTheme>(() => readStoredTheme());
  // Self-contained: keep <html data-cr-theme> in sync even when rendered
  // without main.tsx's initTheme() (tests, alternate entrypoints).
  useLayoutEffect(() => {
    applyTheme(theme);
  }, [theme]);
  const setTheme = useCallback((next: CrTheme) => {
    applyTheme(next);
    persistTheme(next);
    setThemeState(next);
  }, []);
  const value = useMemo(() => ({ theme, setTheme }), [theme, setTheme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}

/**
 * Non-throwing variant for low-level shared widgets that may render in bare
 * test harnesses without a provider. Returns null when unwrapped.
 */
export function useOptionalTheme(): ThemeContextValue | null {
  return useContext(ThemeContext);
}
