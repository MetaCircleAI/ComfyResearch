import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { buildCanvasStandalonePython } from "../src/graph/nodeDefinitionCode";

type GraphDoc = {
  nodes?: unknown[];
  edges?: unknown[];
};

type GoldenEntry = {
  sha256: string;
  bytes: number;
  nodes: number;
  edges: number;
};

type GoldenManifest = {
  version: 1;
  generatedBy: string;
  fixtures: Record<string, GoldenEntry>;
};

export const CODEGEN_GOLDEN_VERSION = 1;
export const CODEGEN_GOLDEN_PATH = "src/graph/__tests__/__snapshots__/codegenGolden.sha256.json";

function repoRoot(): string {
  return resolve(process.cwd(), "..");
}

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, "utf8"));
}

function documentFromFixture(path: string): GraphDoc {
  const raw = readJson(path);
  if (raw && typeof raw === "object" && "document" in raw) {
    return (raw as { document: GraphDoc }).document;
  }
  return raw as GraphDoc;
}

function hashCodegenForDocument(doc: GraphDoc): GoldenEntry {
  const nodes = Array.isArray(doc.nodes) ? doc.nodes : [];
  const edges = Array.isArray(doc.edges) ? doc.edges : [];
  const python = buildCanvasStandalonePython(nodes as never, edges as never);
  const bytes = Buffer.byteLength(python, "utf8");
  return {
    sha256: createHash("sha256").update(python, "utf8").digest("hex"),
    bytes,
    nodes: nodes.length,
    edges: edges.length,
  };
}

export function buildCodegenGoldenManifest(): GoldenManifest {
  const root = repoRoot();
  const fixtures: Record<string, GoldenEntry> = {};
  const defaultGraph = resolve(root, "comfy_research/schemas/default_research_graph.json");
  fixtures["default_research_graph.json"] = hashCodegenForDocument(documentFromFixture(defaultGraph));

  const templatesDir = resolve(root, "data/graph_library/templates");
  for (const name of readdirSync(templatesDir).filter((x) => x.endsWith(".json")).sort((a, b) => a.localeCompare(b))) {
    const path = resolve(templatesDir, name);
    fixtures[`templates/${basename(name)}`] = hashCodegenForDocument(documentFromFixture(path));
  }

  return {
    version: CODEGEN_GOLDEN_VERSION,
    generatedBy: "frontend/scripts/generate-codegen-golden.ts",
    fixtures,
  };
}

function main(): void {
  const out = resolve(process.cwd(), CODEGEN_GOLDEN_PATH);
  const next = `${JSON.stringify(buildCodegenGoldenManifest(), null, 2)}\n`;
  if (process.argv.includes("--check")) {
    let current = "";
    try {
      current = readFileSync(out, "utf8");
    } catch {
      console.error(`Codegen golden snapshot is missing: ${out}`);
      process.exit(1);
    }
    if (current !== next) {
      console.error("Codegen golden snapshot is stale. Regenerate it with npm run generate:codegen-golden.");
      process.exit(1);
    }
    console.log("OK: codegen golden snapshot is current.");
    return;
  }
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, next);
  console.log(`Wrote codegen golden snapshot for ${Object.keys(buildCodegenGoldenManifest().fixtures).length} fixtures.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
