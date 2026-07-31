/**
 * UI theme switching. Spec §6: root attribute data-cr-theme on <html>,
 * default "studio", persisted under comfyresearch.theme.
 */
export type CrTheme = "studio" | "classic" | "paper";

export const THEME_STORAGE_KEY = "comfyresearch.theme";
export const DEFAULT_THEME: CrTheme = "studio";

export function normalizeTheme(value: unknown): CrTheme {
  return value === "classic" || value === "studio" || value === "paper" ? value : DEFAULT_THEME;
}

export function readStoredTheme(): CrTheme {
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return DEFAULT_THEME;
  }
}

export function applyTheme(theme: CrTheme): void {
  document.documentElement.dataset.crTheme = theme;
}

export function persistTheme(theme: CrTheme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* storage unavailable (private mode) — theme still applies for the session */
  }
}

export function initTheme(): CrTheme {
  const theme = readStoredTheme();
  applyTheme(theme);
  return theme;
}
