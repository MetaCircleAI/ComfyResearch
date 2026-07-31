import { useCallback, useEffect, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { KatexMixedInline } from "./KatexMixedInline";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  defaultToyLanguageDatasetData,
  type ToyLanguageDatasetKind,
  type ToyLanguageDatasetNodeData,
} from "./toyLanguageDatasetDefaults";
import { enumChoices, intChoices, packEnumList, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import type { DatasetNodeInfoKind } from "./datasetNodeInfoContent";
import { generateToyLanguageDatasetSpecCode } from "../../graph/specCode/toyLanguageDatasetSpecCode";

const DS_SRC_OPTS = [
  { id: "synthetic" as const, label: "synthetic (offline)" },
  { id: "download" as const, label: "download / cache" },
];

const FORMAL_LANG_OPTS = [
  { id: "anbn", label: "a^n b^n" },
  { id: "anbncn", label: "a^n b^n c^n" },
  { id: "palindrome", label: "palindrome" },
  { id: "parity", label: "parity (xor)" },
];

const BIOGRAPHY_AUG_OPTS = [
  { id: "template", label: "template order" },
  { id: "shuffle_fields", label: "shuffle field blocks" },
  { id: "noise_slots", label: "noise entity slots" },
];

const RELATION_MODE_OPTS = [
  { id: "forward", label: "forward triple" },
  { id: "inverse", label: "inverse query (OBJ,REL → SUBJ)" },
];

const PLAYGROUND_FAMILY_OPTS = [
  { id: "depo", label: "depo (local parity)" },
  { id: "brevo", label: "brevo (copy / long-range)" },
  { id: "mano", label: "mano (modular state trace)" },
  { id: "capo", label: "capo (fact snippets)" },
  { id: "lano", label: "lano (nested brackets)" },
];

const PCFG_GEN_MODE_OPTS = [
  { id: "binary_tree", label: "binary tree (legacy i.i.d. leaves)" },
  { id: "cfg_sentence", label: "CFG sentence (weighted rules + lexicon)" },
] as const;

const PCFG_GRAMMAR_ID_OPTS = [{ id: "world_model", label: "world_model (Adj/N/V toy)" }] as const;

const INSPECT_FORMAT_OPTS = [
  { id: "id" as const, label: "ids (numeric tensors)" },
  { id: "word" as const, label: "words (readable preview)" },
];

function replaceNodeData(
  id: string,
  data: ToyLanguageDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function inferKind(type: string | undefined): ToyLanguageDatasetKind {
  const t = (type ?? "pcfg_dataset") as ToyLanguageDatasetKind;
  return t;
}

/** Dyck LM vocab is exactly ``2 * k`` token ids (paired opens/closes for each bracket type). */
function dyckVocabFromBracketK(numBracketTypes: unknown): number[] {
  const ks = intChoices(numBracketTypes, 1);
  return ks.map((k) => 2 * Math.max(1, k));
}

export function ToyLanguageLmDatasetNode({ id, data, selected, type }: NodeProps) {
  const kind = inferKind(type);
  const defs = defaultToyLanguageDatasetData(kind);
  const d = { ...defs, ...(data as Partial<ToyLanguageDatasetNodeData>) } as ToyLanguageDatasetNodeData;
  const { setNodes } = useReactFlow();
  const infoKind = kind as DatasetNodeInfoKind;

  const seedVal = intChoices(d.seed ?? d.initSeed ?? 0, 0)[0];
  const specName = (d.specCodeName ?? `${kind}Spec`).trim() || `${kind}Spec`;
  const generatedCode = useMemo(
    () => generateToyLanguageDatasetSpecCode(kind, d, specName),
    [d, kind, specName],
  );

  const update = useCallback(
    (patch: Partial<ToyLanguageDatasetNodeData>) => {
      const next = { ...d, ...patch } as ToyLanguageDatasetNodeData;
      if (patch.seed != null) next.initSeed = patch.seed;
      if (patch.initSeed != null) next.seed = patch.initSeed;
      replaceNodeData(id, next, setNodes);
    },
    [d, id, setNodes],
  );

  useEffect(() => {
    if (kind !== "dyck_dataset") return;
    const expected = dyckVocabFromBracketK(d.numBracketTypes);
    const cur = intChoices(d.vocabSize, expected[0] ?? 2);
    const mismatch =
      cur.length !== expected.length || cur.some((v, i) => v !== expected[i]!);
    if (!mismatch) return;
    update({
      vocabSize: packIntList(expected),
      vocabCap: packIntList(expected),
    });
  }, [kind, d.numBracketTypes, d.vocabSize, update]);

  const dataSource = enumChoices(d.dataSource, new Set(["synthetic", "download"] as const), "synthetic")[0];

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType={infoKind}
        nodeId={id}
        graphNodeType={kind}
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d as Record<string, unknown>, kind.replace(/_/g, " "))}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <DatasetSourceSockets />
        <DiscreteMultiSelect<DatasetSamplingMode>
          label="train data sampling"
          options={DATASET_SAMPLING_MODE_OPTIONS}
          value={d.samplingMode ?? "fixed"}
          onCommit={(v) =>
            update({
              samplingMode: (typeof v === "string" ? v : v[0] ?? "fixed") as DatasetSamplingMode,
            })
          }
          ariaLabel="Train data sampling mode for the trainer"
          singleSelect
        />
        <DiscreteMultiSelect<"id" | "word">
          label="tensor inspect format"
          options={INSPECT_FORMAT_OPTS.map((o) => ({ id: o.id, label: o.label }))}
          value={enumChoices(d.inspectFormat, new Set(["id", "word"] as const), "id")[0]}
          singleSelect
          onCommit={(v) =>
            update({
              inspectFormat: packEnumList([(typeof v === "string" ? v : v[0] ?? "id") as "id" | "word"]),
            })
          }
          ariaLabel="Tensor reader preview: token ids or readable words"
        />
        {(kind === "scan_dataset" ||
          kind === "cogs_dataset" ||
          kind === "listops_dataset" ||
          kind === "tinystories_dataset" ||
          kind === "phi1_style_dataset") && (
          <DiscreteMultiSelect
            label="data source"
            options={DS_SRC_OPTS.map((o) => ({ id: o.id, label: o.label }))}
            value={dataSource}
            singleSelect
            onCommit={(v) =>
              update({
                dataSource: packEnumList([
                  ((typeof v === "string" ? v : v[0] ?? "synthetic") === "download" ? "download" : "synthetic") as
                    | "synthetic"
                    | "download",
                ]),
              })
            }
            ariaLabel="Dataset source synthetic or download"
          />
        )}
        {(kind === "scan_dataset" ||
          kind === "cogs_dataset" ||
          kind === "listops_dataset" ||
          kind === "tinystories_dataset" ||
          kind === "phi1_style_dataset") && (
          <label className="cr-field cr-field--inline">
            <span className="cr-field__label">cache / URL hints</span>
            <input
              className="cr-input"
              type="text"
              value={d.cacheDir ?? ""}
              placeholder="optional cache dir override"
              onChange={(e) => update({ cacheDir: e.target.value })}
            />
          </label>
        )}
        {kind === "tinystories_dataset" ? (
          <label className="cr-field cr-field--inline">
            <span className="cr-field__label">TinyStories URL</span>
            <input
              className="cr-input"
              type="text"
              value={d.tinyStoriesUrl ?? ""}
              placeholder="optional .txt URL"
              onChange={(e) => update({ tinyStoriesUrl: e.target.value })}
            />
          </label>
        ) : null}
        {kind === "scan_dataset" ? (
          <label className="cr-field cr-field--inline">
            <span className="cr-field__label">SCAN tasks URL</span>
            <input
              className="cr-input"
              type="text"
              value={d.scanUrl ?? ""}
              placeholder="default: GitHub SCAN tasks.txt"
              onChange={(e) => update({ scanUrl: e.target.value })}
            />
          </label>
        ) : null}
        {kind !== "dyck_dataset" ? (
          <ComfyIntListField
            label="vocab size $v$"
            values={intChoices(d.vocabSize ?? d.vocabCap ?? 32, 32)}
            min={2}
            title={
              kind === "cogs_dataset"
                ? "LM token ids live in a range derived from v (hash of each surface word). Word inspect shows a small fixed lexicon; v does not add more English words—it changes id mapping and collisions."
                : kind === "scan_dataset"
                  ? "LM token ids from v and per-word hashing. Synthetic SCAN uses fixed command/action pools; v is the id modulus, not how many distinct commands exist in text."
                  : "Integer tokens in [0, V − 1]"
            }
            ariaLabel="Vocabulary size"
            onCommit={(vals) => update({ vocabSize: packIntList(vals), vocabCap: packIntList(vals) })}
          />
        ) : null}
        <ComfyIntListField
          label="context length $l$"
          values={intChoices(d.contextLength, 16)}
          min={1}
          title="Input sequence length (next-token target uses one extra draw)"
          ariaLabel="Context length"
          onCommit={(vals) => update({ contextLength: packIntList(vals) })}
        />
        {kind === "pcfg_dataset" ? (
          <>
            <DiscreteMultiSelect
              label="PCFG generator"
              options={PCFG_GEN_MODE_OPTS.map((o) => ({ id: o.id, label: o.label }))}
              value={enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0]}
              singleSelect
              onCommit={(v) =>
                update({
                  pcfgGenMode: packEnumList([(typeof v === "string" ? v : v[0] ?? "binary_tree") as string]),
                })
              }
              ariaLabel="PCFG generation mode"
            />
            {enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0] ===
            "cfg_sentence" ? (
              <DiscreteMultiSelect
                label="builtin grammar"
                options={PCFG_GRAMMAR_ID_OPTS.map((o) => ({ id: o.id, label: o.label }))}
                value={enumChoices(d.pcfgGrammarId, new Set(PCFG_GRAMMAR_ID_OPTS.map((x) => x.id)), "world_model")[0]}
                singleSelect
                onCommit={(v) =>
                  update({
                    pcfgGrammarId: packEnumList([(typeof v === "string" ? v : v[0] ?? "world_model") as string]),
                  })
                }
                ariaLabel="PCFG builtin grammar id"
              />
            ) : null}
            <ComfyIntListField
              label={
                enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0] ===
                "cfg_sentence"
                  ? String.raw`expansion depth cap`
                  : String.raw`PCFG max depth`
              }
              values={intChoices(d.pcfgMaxDepth ?? 8, 8)}
              min={1}
              title={
                enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0] ===
                "cfg_sentence"
                  ? "Safety limit on CFG recursion (raised with this value × 8, minimum 64)"
                  : "Maximum depth for binary-tree expansion"
              }
              ariaLabel={
                enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0] ===
                "cfg_sentence"
                  ? "CFG expansion safety depth"
                  : "PCFG max depth"
              }
              onCommit={(vals) => update({ pcfgMaxDepth: packIntList(vals) })}
            />
            {enumChoices(d.pcfgGenMode, new Set(PCFG_GEN_MODE_OPTS.map((x) => x.id)), "binary_tree")[0] ===
            "binary_tree" ? (
              <ComfyFloatListField
                label={String.raw`terminal probability`}
                values={Array.isArray(d.pcfgTermProb) ? d.pcfgTermProb : [d.pcfgTermProb ?? 0.35]}
                min={0.05}
                max={0.95}
                title="Chance to emit a terminal during recursive expansion"
                ariaLabel="PCFG terminal probability"
                onCommit={(vals) => update({ pcfgTermProb: packFloatList(vals) })}
              />
            ) : null}
          </>
        ) : null}
        {kind === "dyck_dataset" ? (
          <>
            <ComfyIntListField
              label={String.raw`bracket types $k$`}
              values={intChoices(d.numBracketTypes ?? 1, 1)}
              min={1}
              ariaLabel="Number of Dyck bracket types"
              onCommit={(vals) => {
                const vocabVals = vals.map((k) => 2 * Math.max(1, k));
                update({
                  numBracketTypes: packIntList(vals),
                  vocabSize: packIntList(vocabVals),
                  vocabCap: packIntList(vocabVals),
                });
              }}
            />
            <ComfyIntListField
              label={String.raw`max nesting depth`}
              values={intChoices(d.maxNestingDepth ?? 0, 0)}
              min={0}
              title="Cap on unmatched opens in the balanced prefix; 0 means no cap (depth still bounded by context length)"
              ariaLabel="Dyck max nesting depth"
              onCommit={(vals) => update({ maxNestingDepth: packIntList(vals) })}
            />
            <div className="cr-comfy-field">
              <div className="cr-comfy-widget cr-comfy-widget--flush">
                <span
                  className="cr-comfy-widget__label"
                  title="Vocabulary size is 2 × k (one open and one close token id per bracket type)."
                >
                  <KatexMixedInline
                    text="vocab size $v$ (fixed $2k$)"
                    className="cr-katex-mixed-inline"
                  />
                </span>
                <div className="cr-comfy-widget__control-col">
                  <input
                    type="text"
                    className="cr-input cr-comfy-widget__control cr-comfy-widget__control--num cr-comfy-widget__control--derived-readonly"
                    readOnly
                    tabIndex={-1}
                    value={dyckVocabFromBracketK(d.numBracketTypes).join(", ")}
                    title="Vocabulary size is 2 × k (one open and one close token id per bracket type)."
                    aria-label="Dyck vocabulary size derived as twice bracket types"
                  />
                </div>
              </div>
            </div>
          </>
        ) : null}
        {kind === "ngram_language_dataset" ? (
          <>
            <ComfyIntListField
              label={String.raw`Markov order $n$`}
              values={intChoices(d.orderN ?? 3, 3)}
              min={2}
              ariaLabel="n-gram order"
              onCommit={(vals) => update({ orderN: packIntList(vals) })}
            />
            <ComfyFloatListField
              label={String.raw`Dirichlet $\alpha$`}
              values={Array.isArray(d.dirichletAlpha) ? d.dirichletAlpha : [d.dirichletAlpha ?? 1.0]}
              min={1e-3}
              title="Transition row concentration"
              ariaLabel="Dirichlet alpha"
              onCommit={(vals) => update({ dirichletAlpha: packFloatList(vals) })}
            />
          </>
        ) : null}
        {kind === "formal_language_suite_dataset" ? (
          <DiscreteMultiSelect
            label="language type"
            options={FORMAL_LANG_OPTS.map((o) => ({ id: o.id, label: o.label }))}
            value={enumChoices(d.languageType, new Set(FORMAL_LANG_OPTS.map((x) => x.id)), "anbn")[0]}
            singleSelect
            onCommit={(v) =>
              update({
                languageType: packEnumList([(typeof v === "string" ? v : v[0] ?? "anbn") as string]),
              })
            }
            ariaLabel="Formal language suite language type"
          />
        ) : null}
        {(kind === "tinystories_dataset" || kind === "phi1_style_dataset") && (
          <>
            <DiscreteMultiSelect
              label="tokenizer"
              options={[
                { id: "char", label: "char" },
                { id: "word", label: "word" },
              ]}
              value={enumChoices(d.tokenizerMode, new Set(["char", "word"] as const), "char")[0]}
              singleSelect
              onCommit={(v) =>
                update({
                  tokenizerMode: packEnumList([
                    ((typeof v === "string" ? v : v[0] ?? "char") === "word" ? "word" : "char") as "char" | "word",
                  ]),
                })
              }
              ariaLabel="Tokenizer char or word"
            />
            <ComfyIntListField
              label="doc chunk length"
              values={intChoices(d.seqLen ?? 512, 512)}
              min={ctxLenSafe(d.contextLength) + 1}
              ariaLabel="Approximate tokens per synthetic document chunk"
              onCommit={(vals) => update({ seqLen: packIntList(vals) })}
            />
            <ComfyIntListField
              label="stride"
              values={intChoices(d.stride ?? 64, 64)}
              min={1}
              ariaLabel="Sliding window stride"
              onCommit={(vals) => update({ stride: packIntList(vals) })}
            />
          </>
        )}
        {kind === "phi1_style_dataset" ? (
          <label className="cr-field cr-field--inline">
            <span className="cr-field__label">domain mix</span>
            <input
              className="cr-input"
              type="text"
              value={scalarDomainMix(d.domainMix)}
              placeholder="mixed | textbook | qa | code"
              onChange={(e) => update({ domainMix: packEnumList([e.target.value]) })}
            />
          </label>
        ) : null}
        {kind === "biography_lm_dataset" ? (
          <>
            <DiscreteMultiSelect
              label="field augmentation"
              options={BIOGRAPHY_AUG_OPTS.map((o) => ({ id: o.id, label: o.label }))}
              value={
                enumChoices(d.biographyAugmentation, new Set(BIOGRAPHY_AUG_OPTS.map((x) => x.id)), "template")[0]
              }
              singleSelect
              onCommit={(v) =>
                update({
                  biographyAugmentation: packEnumList([(typeof v === "string" ? v : v[0] ?? "template") as string]),
                })
              }
              ariaLabel="Biography field augmentation mode"
            />
            <ComfyFloatListField
              label={String.raw`slot noise probability`}
              values={
                Array.isArray(d.slotNoiseProb) ? d.slotNoiseProb : [d.slotNoiseProb ?? 0]
              }
              min={0}
              max={1}
              title="When augmentation is noise_slots, corrupt sampled entity tokens with this probability"
              ariaLabel="Slot noise probability"
              onCommit={(vals) => update({ slotNoiseProb: packFloatList(vals) })}
            />
          </>
        ) : null}
        {kind === "relation_tuple_dataset" ? (
          <DiscreteMultiSelect
            label="relation mode"
            options={RELATION_MODE_OPTS.map((o) => ({ id: o.id, label: o.label }))}
            value={enumChoices(d.relationMode, new Set(RELATION_MODE_OPTS.map((x) => x.id)), "forward")[0]}
            singleSelect
            onCommit={(v) =>
              update({
                relationMode: packEnumList([(typeof v === "string" ? v : v[0] ?? "forward") as string]),
              })
            }
            ariaLabel="Relation tuple dataset mode"
          />
        ) : null}
        {kind === "multi_hop_fact_chain_dataset" ? (
          <ComfyIntListField
            label={String.raw`chain hops`}
            values={intChoices(d.chainHops ?? 3, 3)}
            min={1}
            title="Number of binary relational facts before the query segment"
            ariaLabel="Multi-hop chain hops"
            onCommit={(vals) => update({ chainHops: packIntList(vals) })}
          />
        ) : null}
        {kind === "synthetic_playground_dataset" ? (
          <>
            <DiscreteMultiSelect
              label="playground family"
              options={PLAYGROUND_FAMILY_OPTS.map((o) => ({ id: o.id, label: o.label }))}
              value={enumChoices(d.playgroundFamily, new Set(PLAYGROUND_FAMILY_OPTS.map((x) => x.id)), "depo")[0]}
              singleSelect
              onCommit={(v) =>
                update({
                  playgroundFamily: packEnumList([(typeof v === "string" ? v : v[0] ?? "depo") as string]),
                })
              }
              ariaLabel="Synthetic playground family"
            />
            <ComfyIntListField
              label={String.raw`Depo window`}
              values={intChoices(d.depoWindow ?? 4, 4)}
              min={2}
              title="Parity window size for depo family"
              ariaLabel="Depo window"
              onCommit={(vals) => update({ depoWindow: packIntList(vals) })}
            />
            <ComfyIntListField
              label={String.raw`Mano modulus`}
              values={intChoices(d.manoModulus ?? 17, 17)}
              min={2}
              title="Hidden counter modulus for mano family"
              ariaLabel="Mano modulus"
              onCommit={(vals) => update({ manoModulus: packIntList(vals) })}
            />
            <ComfyIntListField
              label={String.raw`Lano nesting depth`}
              values={intChoices(d.lanoNestingDepth ?? 4, 4)}
              min={2}
              title="Bracket nesting recursion depth for lano family"
              ariaLabel="Lano nesting depth"
              onCommit={(vals) => update({ lanoNestingDepth: packIntList(vals) })}
            />
          </>
        ) : null}
        <ComfyIntListField
          label="train size"
          values={intChoices(d.trainSize, 800)}
          min={1}
          ariaLabel="Train size"
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          values={intChoices(d.testSize, 200)}
          min={0}
          ariaLabel="Test size"
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(seedVal, 0)}
          min={0}
          title="Controls deterministic sampling (mirrors seed field)"
          ariaLabel="Init seed"
          onCommit={(vals) => {
            const s = packIntList(vals);
            update({ seed: s, initSeed: s });
          }}
        />
      </div>
    </div>
  );
}

function ctxLenSafe(ctx: ToyLanguageDatasetNodeData["contextLength"]): number {
  const v = Array.isArray(ctx) ? ctx[0] : ctx;
  return typeof v === "number" && Number.isFinite(v) ? Math.max(1, v) : 16;
}

function scalarDomainMix(v: ToyLanguageDatasetNodeData["domainMix"]): string {
  const x = Array.isArray(v) ? v[0] : v;
  return typeof x === "string" ? x : "mixed";
}
