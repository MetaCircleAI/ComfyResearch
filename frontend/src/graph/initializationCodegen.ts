/** Initialization notebook-codegen fns(自 nodeRegistrySpec.ts 抽出;
 * saxe/symmetrized 原为 inline arrow,提名同文案)。 */

export function buildMupInitializationCell(pySym: string, title: string): string {
  return `# === ${title} (mup_initialization) ===
# μP-style re-init of Linear / Embedding / LayerNorm is applied on the server before optimization when this node is wired.


def fn_${pySym}_mup_init_note() -> str:
    return "mup_initialization is consumed by POST /api/train when connected to a model initialization socket."
`;
}

export function buildSaxeInitializationCell(pySym: string, title: string): string {
  void pySym;
  return `# === ${title} (saxe_initialization) ===
# Server-side: POST /api/train applies scaled orthogonal initialization amplitude*Q to every nn.Linear
# when this node is wired to a model initialization socket. Documentation stub only.
`;
}

export function buildSymmetrizedMlpInitCell(pySym: string, title: string): string {
  void pySym;
  return `# === ${title} (symmetrized_mlp_init) ===
# Server-side: POST /api/train mirrors the second half of hidden neurons (output exactly 0 at init,
# Chizat et al. 2019 §3.1) when wired to a model initialization socket. Documentation stub only.
`;
}
