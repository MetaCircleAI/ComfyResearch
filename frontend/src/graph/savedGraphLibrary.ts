import type { GraphDocument } from "../types/graph";
import type { GraphFileExportTier } from "./graphFileExportTier";

/** Legacy keys (browser localStorage); migrated once to server files under ``data/graph_library/``. */
const LEGACY_KEY_WORKFLOW = "comfyresearch.library.workflows.v1";
const LEGACY_KEY_TEMPLATE = "comfyresearch.library.templates.v1";

export type SavedGraphKind = "workflow" | "template";

/**
 * Template names ending with ``(YYYY-MM-DD)`` are shown under Blogs in the Templates rail;
 * all other user templates appear under Tutorials.
 */
export function isBlogStyleTemplateName(name: string): boolean {
  return /\(\d{4}-\d{2}-\d{2}\)\s*$/.test(name.trim());
}

export type SavedGraphEntry = {
  id: string;
  name: string;
  /** Granularity used when saved. */
  tier: GraphFileExportTier;
  document: GraphDocument;
  savedAt: number;
  /**
   * Combined subgraphs: stored as a **workflow** entry for the Nodes library (not the Templates rail).
   * Legacy rows may still exist under templates only.
   */
  libraryOrigin?: "combined_model";
};
/** Display-only titles for bundled templates; persisted template metadata remains canonical. */
const TEMPLATE_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  "5d1a2ab4-825d-4251-8a5a-d7b83b42c1d7": "The emergence of sparse attention: impact of data distribution and benefits of repetition",
  "84b21298-67aa-456d-b823-b97ce9352892": "Deep Double Descent: Where Bigger Models and More Data Hurt",
  "97ff01a8-1724-43ec-8c38-fdb62bbe5faf": "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability",
  "a04f21b3-e31c-4ba3-b8b9-d3af752f77d4": "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks",
  "b15a6036-0e38-4f8f-84ba-8b763c408dc9": "Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics",
  "d2e8cdd7-d14c-42c3-94dc-ba2b419c07f9": "On Lazy Training in Differentiable Programming",
  "dafb8339-a932-4b10-b3b6-185fc53a5a4f": "Opening the Black Box of Deep Neural Networks via Information",
  "repro-random-label-memorization-fig1a": "Understanding Deep Learning Requires Rethinking Generalization",
  "repro-rank-collapse-tinyshakespeare-pretraining": "Tracing the Representation Geometry of Language Models from Pretraining to Post-training",
  "repro-spectral-bias-fig1a": "On the Spectral Bias of Neural Networks",
  "repro-thilak-fig1-slingshot": "The Slingshot Effect: A Late-Stage Optimization Anomaly in Adaptive Gradient Methods",
  "repro-linear-mode-connectivity-cifar10": "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs",
  "repro-diffusion-same-init-different-seed": "The Emergence of Reproducibility and Generalizability in Diffusion Models",
  "repro-jastrzbski-fig1-vgg11": "Three Factors Influencing Minima in SGD",
  "repro-keskar-fig23-sb-lb": "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima",
};

function withTemplateDisplayNames(entries: SavedGraphEntry[]): SavedGraphEntry[] {
  return entries.map((entry) => {
    const name = TEMPLATE_DISPLAY_NAMES[entry.id];
    return name && name !== entry.name ? { ...entry, name } : entry;
  });
}

/** Template topics that belong in the classic-paper reproduction collection. */
const CLASSIC_PAPER_TEMPLATE_TOPICS = [
  "neural mechanics",
  "neural_mechanics",
  "phase trans",
  "phase_trans",
  "in context associative recall",
  "double descent",
  "lazy vs rich",
  "lazy v.s rich",
  "edge of stability",
  "staggered sv",
  "grokking",
] as const;

/** Stable IDs for classic-paper templates whose descriptive titles no longer contain legacy topic keywords. */
const CLASSIC_PAPER_TEMPLATE_IDS = new Set([
  "dafb8339-a932-4b10-b3b6-185fc53a5a4f",
  "d2e8cdd7-d14c-42c3-94dc-ba2b419c07f9",
  "5d1a2ab4-825d-4251-8a5a-d7b83b42c1d7",
  "a04f21b3-e31c-4ba3-b8b9-d3af752f77d4",
]);

/** Template IDs intentionally shown in the Blogs group despite matching a legacy topic keyword. */
const BLOG_TEMPLATE_IDS = new Set(["6c68c470-4602-4db3-9e87-a0d47f49749e"]);

/** Repro IDs and names are both accepted for legacy entries. */
export function isClassicPaperReproductionTemplate(entry: Pick<SavedGraphEntry, "id" | "name">): boolean {
  const id = entry.id.trim().toLowerCase();
  const name = entry.name.trim().toLowerCase();
  if (BLOG_TEMPLATE_IDS.has(id)) return false;
  return (
    id.startsWith("repro-") ||
    CLASSIC_PAPER_TEMPLATE_IDS.has(id) ||
    name.startsWith("repro:") ||
    CLASSIC_PAPER_TEMPLATE_TOPICS.some((topic) => name.includes(topic))
  );
}


function kindToSegment(kind: SavedGraphKind): "workflows" | "templates" {
  if (kind === "workflow") return "workflows";
  return "templates";
}

async function readResponseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return (await res.json()) as T;
}

export async function fetchSavedGraphLibrary(kind: SavedGraphKind): Promise<SavedGraphEntry[]> {
  const res = await fetch(`/api/graph-library/${kindToSegment(kind)}`);
  const entries = await readResponseJson<SavedGraphEntry[]>(res);
  return kind === "template" ? withTemplateDisplayNames(entries) : entries;
}

export async function addSavedGraphEntry(
  kind: SavedGraphKind,
  entry: SavedGraphEntry,
): Promise<SavedGraphEntry[]> {
  const res = await fetch(`/api/graph-library/${kindToSegment(kind)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  const entries = await readResponseJson<SavedGraphEntry[]>(res);
  return kind === "template" ? withTemplateDisplayNames(entries) : entries;
}

export async function removeSavedGraphEntry(
  kind: SavedGraphKind,
  id: string,
): Promise<SavedGraphEntry[]> {
  const seg = kindToSegment(kind);
  const res = await fetch(`/api/graph-library/${seg}/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  const entries = await readResponseJson<SavedGraphEntry[]>(res);
  return kind === "template" ? withTemplateDisplayNames(entries) : entries;
}

/**
 * One-time: if server libraries are empty but legacy localStorage has entries, POST them
 * then clear localStorage keys.
 */
export async function migrateLegacyLocalStorageToServer(): Promise<void> {
  for (const kind of ["workflow", "template"] as const) {
    const legacyKey = kind === "workflow" ? LEGACY_KEY_WORKFLOW : LEGACY_KEY_TEMPLATE;
    let raw: string | null;
    try {
      raw = localStorage.getItem(legacyKey);
    } catch {
      continue;
    }
    if (!raw) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      try {
        localStorage.removeItem(legacyKey);
      } catch {
        /* ignore */
      }
      continue;
    }

    if (!Array.isArray(parsed) || parsed.length === 0) {
      try {
        localStorage.removeItem(legacyKey);
      } catch {
        /* ignore */
      }
      continue;
    }

    const segment = kindToSegment(kind);
    const res = await fetch(`/api/graph-library/${segment}`);
    if (!res.ok) continue;
    const server = (await res.json()) as unknown;
    if (Array.isArray(server) && server.length > 0) {
      try {
        localStorage.removeItem(legacyKey);
      } catch {
        /* ignore */
      }
      continue;
    }

    for (const item of parsed) {
      try {
        await fetch(`/api/graph-library/${segment}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item),
        });
      } catch {
        /* best-effort per entry */
      }
    }
    try {
      localStorage.removeItem(legacyKey);
    } catch {
      /* ignore */
    }
  }
}
