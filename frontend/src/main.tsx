import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "katex/dist/katex.min.css";
import "./fonts.css";
import "./tokens.css";
import "./index.css";
import "./studio.css";
import { initTheme } from "./theme";
import { ThemeProvider } from "./themeContext";

initTheme();

// Cancel browser page zoom before canvas handlers can stop propagation.
const preventBrowserZoom = (event: WheelEvent) => {
  if (event.ctrlKey || event.metaKey) event.preventDefault();
};

window.addEventListener(
  "wheel",
  preventBrowserZoom,
  { capture: true, passive: false },
);

// Safari dispatches this non-standard event for trackpad pinch zoom.
document.addEventListener("gesturestart", (event) => event.preventDefault(), { capture: true, passive: false });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
