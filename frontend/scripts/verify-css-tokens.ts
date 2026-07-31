/**
 * Hex-ratchet guard (spec §9). tokens.css is the only file allowed to contain
 * raw hex colors freely. Every other src CSS file has a frozen per-value
 * baseline: any NEW hex value, or an increased count of an existing value,
 * fails the build; decreases prompt a baseline ratchet-down. Migrating styles
 * onto tokens (PR-2/PR-3) drives the baseline to {} — the PR-3 exit gate.
 * Known accepted gap: adding one more occurrence of an ALREADY-baselined hex
 * value while removing another occurrence of the same value in the same file
 * is invisible to the multiset. The ratchet's goal is monotonic decrease to
 * zero, at which point the gap closes itself.
 * Usage (cwd = frontend/): node <bundle> [--update]
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

const frontendRoot = process.cwd();
const srcDir = path.resolve(frontendRoot, "src");
const baselinePath = path.resolve(frontendRoot, "scripts/cssHexBaseline.json");
const HEX_RE = /#[0-9a-fA-F]{3,8}\b/g;

/**
 * tokens.css is exempt from the hex ratchet, so keep it honest: it may only
 * contain comments, the two palette selectors, and declarations of custom
 * properties (--*) plus color-scheme. Anything else must live in component
 * CSS where the ratchet applies.
 */
function verifyTokensCssPurity(): string[] {
  const tokensPath = path.join(srcDir, "tokens.css");
  let text: string;
  try {
    text = readFileSync(tokensPath, "utf8");
  } catch {
    return ["src/tokens.css missing"];
  }
  const problems: string[] = [];
  const noComments = text.replace(/\/\*[\s\S]*?\*\//g, "");
  const ALLOWED_SELECTORS = new Set([":root", ':root[data-cr-theme="classic"]', ':root[data-cr-theme="paper"]']);
  for (const m of noComments.matchAll(/(^|\})\s*([^{}]+?)\s*\{/g)) {
    const sel = m[2]!.trim();
    if (!ALLOWED_SELECTORS.has(sel)) problems.push(`tokens.css: unexpected selector "${sel}"`);
  }
  for (const m of noComments.matchAll(/\{([\s\S]*?)\}/g)) {
    for (const decl of m[1]!.split(";")) {
      const d = decl.trim();
      if (!d) continue;
      if (!d.startsWith("--") && !d.startsWith("color-scheme")) {
        problems.push(`tokens.css: non-token declaration "${d.slice(0, 60)}"`);
      }
    }
  }
  return problems;
}

type HexCounts = Record<string, number>;

function cssFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fp = path.join(dir, entry);
    if (statSync(fp).isDirectory()) out.push(...cssFiles(fp));
    else if (entry.endsWith(".css")) out.push(fp);
  }
  return out;
}

const current: Record<string, HexCounts> = {};
for (const fp of cssFiles(srcDir)) {
  const rel = path.relative(srcDir, fp).split(path.sep).join("/");
  if (rel === "tokens.css") continue;
  const counts: HexCounts = {};
  for (const m of readFileSync(fp, "utf8").match(HEX_RE) ?? []) {
    const hex = m.toLowerCase();
    counts[hex] = (counts[hex] ?? 0) + 1;
  }
  current[rel] = counts;
}

if (process.argv.includes("--update")) {
  writeFileSync(baselinePath, JSON.stringify(current, null, 2) + "\n");
  const total = Object.values(current).reduce(
    (n, f) => n + Object.values(f).reduce((a, b) => a + b, 0), 0);
  console.log(`Updated cssHexBaseline.json (${total} hex occurrences frozen).`);
  process.exit(0);
}

let baseline: Record<string, HexCounts>;
try {
  baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
} catch {
  console.error("ERROR: scripts/cssHexBaseline.json missing. Run: npm run verify:css-tokens -- --update");
  process.exit(1);
}

const errors: string[] = [];
let shrunk = false;
for (const [rel, counts] of Object.entries(current)) {
  const base: HexCounts = baseline[rel] ?? {};
  for (const [hex, n] of Object.entries(counts)) {
    const b = base[hex] ?? 0;
    if (n > b) errors.push(`${rel}: ${hex} count ${b} -> ${n} — use design tokens instead of raw hex`);
    else if (n < b) shrunk = true;
  }
  for (const hex of Object.keys(base)) if (!(hex in counts)) shrunk = true;
}
for (const rel of Object.keys(baseline)) {
  if (!(rel in current)) shrunk = true;
}

errors.push(...verifyTokensCssPurity());

if (errors.length) {
  console.error("CSS token guard failed:\n  " + errors.join("\n  "));
  process.exit(1);
}
if (shrunk) {
  console.error(
    "CSS token guard: hex usage decreased (good!) — lock it in with: npm run verify:css-tokens -- --update",
  );
  process.exit(1);
}
console.log(`OK: css hex ratchet holds for ${Object.keys(current).length} file(s).`);
