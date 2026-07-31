"""Human-readable lines for toy language dataset tensor preview (inspect format = word)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from comfy_research.engine.datasets.toy_language_common import dataset_rng_seed, resize_sequence, scalar_float, scalar_int, scalar_str
from comfy_research.engine.datasets.toy_language_external_runtime import (
    SCAN_SIMPLE_URL,
    _scan_parse_lines,
    build_synthetic_phi_style_documents,
    build_synthetic_tinystories_documents,
    download_if_missing,
    resolve_cache_dir,
)
from comfy_research.engine.datasets.toy_language_ngram_runtime import build_ngram_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_pcfg_runtime import (
    PCFG_GEN_CFG_SENTENCE,
    _expand_cfg,
    _expand_terminals,
    _grammar_world_model,
    _pcfg_gen_mode,
    _pcfg_grammar_id,
    _sentence_to_ids,
    build_pcfg_lm_arrays_from_seed,
)
from comfy_research.generated.node_capabilities import node_types_with_capability

TOY_LANGUAGE_INSPECT_TYPES: frozenset[str] = frozenset(node_types_with_capability("toy_language_token_dataset"))


def _train_test_sizes(data: dict[str, Any]) -> tuple[int, int]:
    train_sz = max(0, scalar_int(data.get("trainSize"), 800))
    test_sz = max(0, scalar_int(data.get("testSize"), 200))
    return train_sz, test_sz


def _slice_lm_labels(full_rows: list[list[str]], ctx: int, tensor_key: str) -> list[str]:
    out: list[str] = []
    for lab in full_rows:
        if tensor_key == "input":
            seg = lab[:ctx]
        else:
            seg = lab[1 : ctx + 1] if len(lab) >= ctx + 1 else lab[1:]
        out.append(" ".join(seg))
    return out


def _scan_full_label_rows(data: dict[str, Any], train_n: int, test_n: int) -> tuple[list[list[str]], list[list[str]]]:
    ctx = max(1, scalar_int(data.get("contextLength"), 24))
    _ = max(8, scalar_int(data.get("vocabSize"), 64))
    seed = dataset_rng_seed(data)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1)
    src = scalar_str(data.get("dataSource"), "synthetic").strip().lower()
    cmds_pool = ["jump", "walk", "look", "run", "turn", "left", "right", "twice", "thrice", "opposite"]
    acts_pool = ["JUMP", "WALK", "LOOK", "RUN", "LEFT", "RIGHT"]

    def synth_labels(rng: np.random.Generator) -> list[str]:
        n_cmd = int(rng.integers(1, min(6, ctx)))
        cmd_words = [str(rng.choice(cmds_pool)) for _ in range(n_cmd)]
        first_act = str(rng.choice(acts_pool))
        labels = [cmd_words[j] if j < len(cmd_words) else "<pad>" for j in range(ctx)]
        labels.append(first_act)
        return labels

    def rows_from_pairs(
        pairs_subset: list[tuple[list[str], list[str]]], n: int, rng: np.random.Generator
    ) -> list[list[str]]:
        out: list[list[str]] = []
        if not pairs_subset:
            return out
        for _ in range(n):
            cmd, act = pairs_subset[int(rng.integers(0, len(pairs_subset)))]
            cmd_use = cmd[:ctx]
            labels = [cmd_use[j] if j < len(cmd_use) else "<pad>" for j in range(ctx)]
            first_act = act[0] if act else "JUMP"
            labels.append(first_act)
            out.append(labels)
        return out

    lab_tr: list[list[str]] = []
    lab_te: list[list[str]] = []
    if src == "download":
        cache = resolve_cache_dir(data, "scan")
        dest = cache / "tasks.txt"
        pairs: list[tuple[list[str], list[str]]] = []
        try:
            download_if_missing(scalar_str(data.get("scanUrl"), SCAN_SIMPLE_URL) or SCAN_SIMPLE_URL, dest)
            pairs = _scan_parse_lines(dest.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pairs = []
        if pairs:
            perm = rng_tr.permutation(len(pairs))
            n_tr = max(1, int(len(pairs) * 0.9))
            tr_pairs = [pairs[int(i)] for i in perm[:n_tr]]
            te_pairs = [pairs[int(i)] for i in perm[n_tr:]]
            lab_tr = rows_from_pairs(tr_pairs, train_n, rng_tr)
            lab_te = rows_from_pairs(te_pairs if te_pairs else tr_pairs, test_n, rng_te)
        else:
            src = "synthetic"
    if src != "download" or not lab_tr:
        lab_tr = [synth_labels(rng_tr) for _ in range(train_n)]
        lab_te = [synth_labels(rng_te) for _ in range(test_n)]
    return lab_tr, lab_te


def _cogs_full_label_rows(data: dict[str, Any], train_n: int, test_n: int) -> tuple[list[list[str]], list[list[str]]]:
    ctx = max(1, scalar_int(data.get("contextLength"), 32))
    _ = max(16, scalar_int(data.get("vocabSize"), 128))
    seed = dataset_rng_seed(data)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1)
    nouns = ["cat", "dog", "teacher", "student", "cake"]
    verbs = ["chased", "saw", "liked"]
    det = ["the", "a"]

    def synth_labels(rng: np.random.Generator) -> list[str]:
        words = [
            str(rng.choice(det)),
            str(rng.choice(nouns)),
            str(rng.choice(verbs)),
            str(rng.choice(det)),
            str(rng.choice(nouns)),
        ]
        lf = f"* chase ( {rng.choice(nouns)} , {rng.choice(nouns)} )"
        wi = words[:ctx]
        while len(wi) < ctx:
            wi.append("<pad>")
        tgt_tok = lf.split()[0]
        return wi + [tgt_tok]

    return [synth_labels(rng_tr) for _ in range(train_n)], [synth_labels(rng_te) for _ in range(test_n)]


def _listops_full_label_rows(data: dict[str, Any], train_n: int, test_n: int) -> tuple[list[list[str]], list[list[str]]]:
    ctx = max(1, scalar_int(data.get("contextLength"), 48))
    v = max(16, scalar_int(data.get("vocabSize"), 64))
    seed = dataset_rng_seed(data)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1)

    def synth_labels(rng: np.random.Generator) -> list[str]:
        inner = [int(rng.integers(0, min(20, v))), int(rng.integers(0, min(20, v)))]
        ans = int(min(inner) % v)
        core_labels = ["<LB>", "min", str(inner[0] % v), str(inner[1] % v), "<RB>"]
        seq_list = (core_labels + ["<pad>"] * (ctx + 1))[: ctx + 1]
        seq_list[-1] = str(ans % v)
        return seq_list

    return [synth_labels(rng_tr) for _ in range(train_n)], [synth_labels(rng_te) for _ in range(test_n)]


def _dyck_token_str(tid: int, k: int) -> str:
    pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("⟨", "⟩"), ("⁽", "⁾")]
    t = int(tid)
    for i in range(min(k, len(pairs))):
        oi, ci = 2 * i, 2 * i + 1
        if t == oi:
            return pairs[i][0]
        if t == ci:
            return pairs[i][1]
    return f"·{t}"


def _dyck_lines_from_array(arr: np.ndarray, k: int) -> list[str]:
    out: list[str] = []
    for i in range(arr.shape[0]):
        out.append(" ".join(_dyck_token_str(int(t), k) for t in arr[i]))
    return out


def _formal_token_str(tid: int, lang: str) -> str:
    t = int(tid)
    lang_l = lang.strip().lower()
    if lang_l in {"anbn", "anbncn", "parity"} and 0 <= t < 26:
        return chr(ord("a") + t)
    return str(t)


def _formal_lines_from_array(arr: np.ndarray, lang: str) -> list[str]:
    return [" ".join(_formal_token_str(int(x), lang) for x in arr[i]) for i in range(arr.shape[0])]


def _tokenize_id_label_pairs(mode: str, text: str, vocab_cap: int) -> tuple[list[int], list[str]]:
    v = max(2, int(vocab_cap))
    text = text.strip()
    if not text:
        return [0], ["<empty>"]
    if mode.strip().lower() == "word":
        parts = re.split(r"\s+", text)
        ids = [hash(w) % v for w in parts if w]
        labs = [w for w in parts if w]
        return ids, labs
    ids = [ord(c) % v for c in text]
    labs = ["\\n" if c == "\n" else c for c in text]
    return ids, labs


def _sliding_id_and_label_rows(
    flat_ids: list[int], flat_labs: list[str], ctx: int, stride: int, rng: np.random.Generator, max_windows: int
) -> tuple[np.ndarray, list[list[str]]]:
    """Same windowing RNG side-effects as ``sliding_lm_windows``; returns int rows + parallel label rows."""
    ctx = max(1, int(ctx))
    L = ctx + 1
    st = max(1, int(stride))
    if len(flat_ids) < L:
        pad_ids = flat_ids + [0] * (L - len(flat_ids))
        pad_labs = flat_labs + ["<pad>"] * (L - len(flat_labs))
        row_ids = np.asarray(pad_ids[:L], dtype=np.int64).reshape(1, L)
        return row_ids, [pad_labs[:L]]
    starts = list(range(0, len(flat_ids) - L + 1, st))
    if not starts:
        starts = [0]
    if len(starts) > max_windows:
        choice = rng.choice(len(starts), size=max_windows, replace=False)
        starts = [starts[int(i)] for i in sorted(choice)]
    elif len(starts) > max_windows * 2:
        starts = starts[:max_windows]
    rows = np.empty((len(starts), L), dtype=np.int64)
    lab_rows: list[list[str]] = []
    for i, s in enumerate(starts):
        rows[i] = np.asarray(flat_ids[s : s + L], dtype=np.int64)
        lab_rows.append(flat_labs[s : s + L])
    return rows, lab_rows


def _corpus_lm_word_lines(
    data: dict[str, Any], train_n: int, test_n: int, corpus_kind: str, split: str, tensor_key: str
) -> list[str]:
    ctx = max(1, scalar_int(data.get("contextLength"), 64))
    seq_doc = max(ctx + 5, scalar_int(data.get("seqLen"), 256))
    stride = max(1, scalar_int(data.get("stride"), ctx + 1))
    vocab_cap = max(2, scalar_int(data.get("vocabCap"), 256))
    tok_mode = scalar_str(data.get("tokenizerMode"), "char")
    data_src = scalar_str(data.get("dataSource"), "synthetic").strip().lower()
    seed = dataset_rng_seed(data)
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)
    rng = rng_train if split == "train" else rng_test
    n = train_n if split == "train" else test_n

    docs: list[str] = []
    if data_src == "download" and corpus_kind == "tinystories":
        cache = resolve_cache_dir(data, "tinystories")
        dest = cache / "tiny_sample.txt"
        try:
            url = scalar_str(data.get("tinyStoriesUrl"), "").strip() or scalar_str(data.get("downloadUrl"), "").strip()
            if url:
                download_if_missing(url, dest)
            if dest.exists():
                docs = dest.read_text(encoding="utf-8", errors="ignore").split("\n\n")[:2000]
        except Exception:
            docs = []
        if not docs:
            docs = build_synthetic_tinystories_documents(rng_train, max(8, train_n // max(1, seq_doc // 32)))
    elif data_src == "download" and corpus_kind == "phi_style":
        docs = build_synthetic_phi_style_documents(
            rng_train, max(8, train_n // max(1, seq_doc // 64)), scalar_str(data.get("domainMix"), "mixed")
        )
    else:
        if corpus_kind == "tinystories":
            docs = build_synthetic_tinystories_documents(rng_train, max(8, train_n // max(1, seq_doc // 32)))
        else:
            docs = build_synthetic_phi_style_documents(
                rng_train, max(8, train_n // max(1, seq_doc // 64)), scalar_str(data.get("domainMix"), "mixed")
            )

    flat_ids: list[int] = []
    flat_labs: list[str] = []
    budget = seq_doc * len(docs)
    for d in docs:
        ids, labs = _tokenize_id_label_pairs(tok_mode, d[: max(4000, seq_doc)], vocab_cap)
        flat_ids.extend(ids[:seq_doc])
        flat_labs.extend(labs[:seq_doc])
        if len(flat_ids) >= budget:
            break
    if not flat_ids:
        flat_ids, flat_labs = [0], ["<empty>"]

    rows_np, lab_rows = _sliding_id_and_label_rows(flat_ids, flat_labs, ctx, stride, rng, max(n, 1))
    if n > 0 and rows_np.shape[0] > n:
        idx = rng.choice(rows_np.shape[0], size=n, replace=False)
        lab_rows = [lab_rows[int(i)] for i in sorted(idx)]
    elif n > 0 and 0 < rows_np.shape[0] < n:
        idx = rng.integers(0, rows_np.shape[0], size=n)
        lab_rows = [lab_rows[int(i)] for i in idx]

    lines: list[str] = []
    for lab in lab_rows:
        if tensor_key == "input":
            seg = lab[:ctx]
        else:
            seg = lab[1 : ctx + 1] if len(lab) >= ctx + 1 else lab[1:]
        lines.append(" ".join(seg))
    return lines


def _physics_special_str(tid: int) -> str:
    names: dict[int, str] = {
        0: "PAD",
        1: "BOS",
        2: "EOS",
        3: "T_NAME",
        4: "T_BORN",
        5: "T_CITY",
        6: "SEP",
        7: "T_SUB",
        8: "T_REL",
        9: "T_OBJ",
        10: "T_Q",
        11: "T_ENT",
        12: "T_REL_CHAIN",
        13: "T_Q_CHAIN",
        14: "T_ANS",
    }
    t = int(tid)
    if t in names:
        return names[t]
    if t < 24:
        return f"spec_{t}"
    return f"ent_{t}"


def _lines_from_int_array(arr: np.ndarray, mapper: Any) -> list[str]:
    out: list[str] = []
    for i in range(arr.shape[0]):
        out.append(" ".join(mapper(int(t)) for t in arr[i]))
    return out


def _pcfg_cfg_full_rows(data: dict[str, Any], train_n: int, test_n: int) -> tuple[list[list[str]], list[list[str]]]:
    ctx = max(1, scalar_int(data.get("contextLength"), 16))
    v = max(2, scalar_int(data.get("vocabSize"), 32))
    productions, tok2id = _grammar_world_model()
    n_term = len(tok2id)
    v = max(v, n_term)
    safety = max(64, scalar_int(data.get("pcfgMaxDepth"), 8) * 8)
    seed = dataset_rng_seed(data)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1)
    target_len = ctx + 1

    def sample_row(rng: np.random.Generator) -> list[str]:
        words = _expand_cfg("S", productions, rng, safety)
        toks = _sentence_to_ids(words, tok2id, rng, v)
        row_ids = resize_sequence(toks, target_len, rng, v)
        inv_map: dict[int, str] = {tid: w for w, tid in tok2id.items()}
        labels: list[str] = []
        for tid in row_ids.tolist():
            t = int(tid)
            labels.append(inv_map.get(t, f"·{t}"))
        return labels

    return [sample_row(rng_tr) for _ in range(train_n)], [sample_row(rng_te) for _ in range(test_n)]


def _pcfg_binary_full_rows(data: dict[str, Any], train_n: int, test_n: int) -> tuple[list[list[str]], list[list[str]]]:
    ctx = max(1, scalar_int(data.get("contextLength"), 16))
    v = max(2, scalar_int(data.get("vocabSize"), 32))
    max_depth = max(1, scalar_int(data.get("pcfgMaxDepth"), 8))
    term_prob = float(np.clip(scalar_float(data.get("pcfgTermProb"), 0.35), 0.05, 0.95))
    target_len = ctx + 1
    seed = dataset_rng_seed(data)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1)

    def sample_row(rng: np.random.Generator) -> list[str]:
        toks = _expand_terminals(rng, v, max_depth, term_prob)
        row_ids = resize_sequence(toks, target_len, rng, v)
        return [f"t{int(t)}" for t in row_ids.tolist()]

    return [sample_row(rng_tr) for _ in range(train_n)], [sample_row(rng_te) for _ in range(test_n)]


def toy_language_word_inspect_lines(
    ds_type: str, data: dict[str, Any], split: str, tensor_key: str
) -> tuple[list[str], list[int], str | None]:
    """Return (one line per batch row, tensor shape, optional footnote)."""
    train_n, test_n = _train_test_sizes(data)
    foot: str | None = None

    if ds_type == "scan_dataset":
        ctx = max(1, scalar_int(data.get("contextLength"), 24))
        tr, te = _scan_full_label_rows(data, train_n, test_n)
        full = tr if split == "train" else te
        lines = _slice_lm_labels(full, ctx, tensor_key)
        return lines, [len(lines), ctx], None

    if ds_type == "cogs_dataset":
        ctx = max(1, scalar_int(data.get("contextLength"), 32))
        tr, te = _cogs_full_label_rows(data, train_n, test_n)
        full = tr if split == "train" else te
        lines = _slice_lm_labels(full, ctx, tensor_key)
        return lines, [len(lines), ctx], None

    if ds_type == "listops_dataset":
        ctx = max(1, scalar_int(data.get("contextLength"), 48))
        tr, te = _listops_full_label_rows(data, train_n, test_n)
        full = tr if split == "train" else te
        lines = _slice_lm_labels(full, ctx, tensor_key)
        return lines, [len(lines), ctx], None

    if ds_type == "dyck_dataset":
        from comfy_research.engine.datasets.toy_language_dyck_runtime import build_dyck_lm_arrays_from_seed

        x_tr, y_tr, x_te, y_te = build_dyck_lm_arrays_from_seed(data, train_n, test_n)
        k = max(1, scalar_int(data.get("numBracketTypes"), 1))
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _dyck_lines_from_array(arr, k)
        return lines, list(arr.shape), None

    if ds_type == "formal_language_suite_dataset":
        from comfy_research.engine.datasets.toy_language_formal_runtime import build_formal_suite_lm_arrays_from_seed

        lang = scalar_str(data.get("languageType"), "anbn")
        x_tr, y_tr, x_te, y_te = build_formal_suite_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _formal_lines_from_array(arr, lang)
        return lines, list(arr.shape), None

    if ds_type == "ngram_language_dataset":
        x_tr, y_tr, x_te, y_te = build_ngram_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _lines_from_int_array(arr, lambda t: f"t{t}")
        return lines, list(arr.shape), None

    if ds_type == "pcfg_dataset":
        ctx = max(1, scalar_int(data.get("contextLength"), 16))
        mode = _pcfg_gen_mode(data)
        if mode == PCFG_GEN_CFG_SENTENCE and _pcfg_grammar_id(data) == "world_model":
            tr, te = _pcfg_cfg_full_rows(data, train_n, test_n)
        else:
            tr, te = _pcfg_binary_full_rows(data, train_n, test_n)
        full = tr if split == "train" else te
        lines = _slice_lm_labels(full, ctx, tensor_key)
        x_tr, _, _, _ = build_pcfg_lm_arrays_from_seed(data, train_n, test_n)
        return lines, [len(lines), x_tr.shape[1]], None

    if ds_type == "tinystories_dataset":
        lines = _corpus_lm_word_lines(data, train_n, test_n, "tinystories", split, tensor_key)
        ctx = max(1, scalar_int(data.get("contextLength"), 64))
        return lines, [len(lines), ctx], None

    if ds_type == "phi1_style_dataset":
        lines = _corpus_lm_word_lines(data, train_n, test_n, "phi_style", split, tensor_key)
        ctx = max(1, scalar_int(data.get("contextLength"), 96))
        return lines, [len(lines), ctx], None

    if ds_type == "biography_lm_dataset":
        from comfy_research.engine.datasets.toy_language_physics_lm_runtime import build_biography_lm_arrays_from_seed

        x_tr, y_tr, x_te, y_te = build_biography_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _lines_from_int_array(arr, _physics_special_str)
        foot = "Biography LM: low ids are structure markers; high ids are sampled entity tokens."
        return lines, list(arr.shape), foot

    if ds_type == "relation_tuple_dataset":
        from comfy_research.engine.datasets.toy_language_physics_lm_runtime import build_relation_tuple_lm_arrays_from_seed

        x_tr, y_tr, x_te, y_te = build_relation_tuple_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _lines_from_int_array(arr, _physics_special_str)
        return lines, list(arr.shape), None

    if ds_type == "multi_hop_fact_chain_dataset":
        from comfy_research.engine.datasets.toy_language_physics_lm_runtime import build_multi_hop_fact_chain_lm_arrays_from_seed

        x_tr, y_tr, x_te, y_te = build_multi_hop_fact_chain_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _lines_from_int_array(arr, _physics_special_str)
        return lines, list(arr.shape), None

    if ds_type == "synthetic_playground_dataset":
        from comfy_research.engine.datasets.toy_language_physics_lm_runtime import build_synthetic_playground_lm_arrays_from_seed

        x_tr, y_tr, x_te, y_te = build_synthetic_playground_lm_arrays_from_seed(data, train_n, test_n)
        arr = (x_tr if tensor_key == "input" else y_tr) if split == "train" else (x_te if tensor_key == "input" else y_te)
        lines = _lines_from_int_array(arr, lambda t: f"t{t}")
        foot = "Playground preset: token ids as t{n} (no fixed lexicon)."
        return lines, list(arr.shape), foot

    raise ValueError(f"unsupported toy language inspect type: {ds_type}")
