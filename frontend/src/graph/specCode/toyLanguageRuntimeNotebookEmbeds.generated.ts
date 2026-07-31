/**
 * AUTO-GENERATED — run: ``python scripts/gen_toy_language_notebook_embeds.py``
 * from the repo root after editing ``comfy_research/engine/toy_language_*.py``.
 * Python slices are escaped for TS template literals (`` ` ``, ``\``, ``${``).
 */

export const PCFG_LM_NOTEBOOK_IMPL_BLOCK = String.raw`
    # --- cfg_sentence: builtin toy grammars (extend with new ids in future) ---
    
    PCFG_GEN_BINARY_TREE = "binary_tree"
    PCFG_GEN_CFG_SENTENCE = "cfg_sentence"
    
    # NP -> Adj N | N ; VP -> V NP ; fixed lexicon (research-style toy).
    _WORLD_MODEL_PRODUCTIONS: dict[str, list[tuple[tuple[str, ...], float]]] = {
        "S": [(("NP", "VP"), 1.0)],
        "NP": [(("Adj", "N"), 0.5), (("N",), 0.5)],
        "VP": [(("V", "NP"), 1.0)],
        "Adj": [(("big",), 0.5), (("small",), 0.5)],
        "N": [(("planet",), 0.5), (("star",), 0.5)],
        "V": [(("attracts",), 0.5), (("repels",), 0.5)],
    }
    
    _WORLD_MODEL_TERMINALS_ORDERED = ("attracts", "big", "planet", "repels", "small", "star")
    
    
    def _pcfg_gen_mode(data: dict[str, Any]) -> str:
        raw = data.get("pcfgGenMode", PCFG_GEN_BINARY_TREE)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            raw = raw[0]
        if raw == PCFG_GEN_CFG_SENTENCE:
            return PCFG_GEN_CFG_SENTENCE
        return PCFG_GEN_BINARY_TREE
    
    
    def _pcfg_grammar_id(data: dict[str, Any]) -> str:
        raw = data.get("pcfgGrammarId", "world_model")
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            raw = raw[0]
        return str(raw) if raw else "world_model"
    
    
    def _grammar_world_model() -> tuple[dict[str, list[tuple[tuple[str, ...], float]]], dict[str, int]]:
        tok2id = {t: i for i, t in enumerate(_WORLD_MODEL_TERMINALS_ORDERED)}
        return _WORLD_MODEL_PRODUCTIONS, tok2id
    
    
    def _pick_rhs(rng: np.random.Generator, rules: list[tuple[tuple[str, ...], float]]) -> tuple[str, ...]:
        weights = np.asarray([w for _, w in rules], dtype=np.float64)
        if weights.size == 0:
            raise ValueError("empty production list")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("invalid production weights")
        s = float(weights.sum())
        if s <= 0:
            raise ValueError("production weights sum to zero")
        u = float(rng.random()) * s
        acc = 0.0
        for rhs, w in rules:
            acc += float(w)
            if u <= acc:
                return rhs
        return rules[-1][0]
    
    
    def _expand_cfg(
        sym: str,
        productions: dict[str, list[tuple[tuple[str, ...], float]]],
        rng: np.random.Generator,
        depth_left: int,
    ) -> list[str]:
        if depth_left < 0:
            raise RuntimeError("PCFG expansion exceeded safety depth — check for missing productions")
        rules = productions.get(sym)
        if rules is None:
            return [sym]
        rhs = _pick_rhs(rng, rules)
        out: list[str] = []
        for c in rhs:
            out.extend(_expand_cfg(c, productions, rng, depth_left - 1))
        return out
    
    
    def _sentence_to_ids(words: list[str], tok2id: dict[str, int], rng: np.random.Generator, vocab_size: int) -> list[int]:
        v = max(2, int(vocab_size))
        row: list[int] = []
        for w in words:
            tid = tok2id.get(w)
            if tid is None:
                row.append(int(rng.integers(0, v)))
            else:
                if tid >= v:
                    row.append(int(tid % v))
                else:
                    row.append(int(tid))
        return row
    
    
    def _expand_terminals(rng: np.random.Generator, vocab_size: int, max_depth: int, term_prob: float) -> list[int]:
        v = max(2, int(vocab_size))
        if max_depth <= 0 or float(rng.random()) < term_prob:
            return [int(rng.integers(0, v))]
        left = _expand_terminals(rng, v, max_depth - 1, term_prob)
        right = _expand_terminals(rng, v, max_depth - 1, term_prob)
        return left + right
    
    
    def build_pcfg_lm_arrays(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
        rng_train: np.random.Generator,
        rng_test: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mode = _pcfg_gen_mode(data)
        v = max(2, scalar_int(data.get("vocabSize"), 32))
        ctx = max(1, scalar_int(data.get("contextLength"), 16))
        target_len = ctx + 1
    
        if mode == PCFG_GEN_CFG_SENTENCE:
            gid = _pcfg_grammar_id(data)
            if gid != "world_model":
                raise ValueError(f"unsupported pcfgGrammarId: {gid!r} (supported: 'world_model')")
            productions, tok2id = _grammar_world_model()
            n_term = len(tok2id)
            v = max(v, n_term)
            safety = max(64, scalar_int(data.get("pcfgMaxDepth"), 8) * 8)
    
            def sample_cfg_row(rng: np.random.Generator) -> list[int]:
                words = _expand_cfg("S", productions, rng, safety)
                return _sentence_to_ids(words, tok2id, rng, v)
    
            def sample_split_cfg(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
                if n <= 0:
                    z = np.zeros((0, ctx), dtype=np.int64)
                    return z, np.zeros((0, ctx), dtype=np.int64)
                rows = np.empty((n, target_len), dtype=np.int64)
                for i in range(n):
                    toks = sample_cfg_row(rng)
                    rows[i] = resize_sequence(toks, target_len, rng, v)
                return slice_shifted_window_lm(rows, ctx)
    
            x_tr, y_tr = sample_split_cfg(train_n, rng_train)
            x_te, y_te = sample_split_cfg(test_n, rng_test)
            return x_tr, y_tr, x_te, y_te
    
        max_depth = max(1, scalar_int(data.get("pcfgMaxDepth"), 8))
        term_prob = float(np.clip(scalar_float(data.get("pcfgTermProb"), 0.35), 0.05, 0.95))
    
        def sample_split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                z = np.zeros((0, ctx), dtype=np.int64)
                return z, np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                toks = _expand_terminals(rng, v, max_depth, term_prob)
                rows[i] = resize_sequence(toks, target_len, rng, v)
            return slice_shifted_window_lm(rows, ctx)
    
        x_tr, y_tr = sample_split(train_n, rng_train)
        x_te, y_te = sample_split(test_n, rng_test)
        return x_tr, y_tr, x_te, y_te
    
    
    def build_pcfg_lm_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return build_pcfg_lm_arrays(data, train_n, test_n, np.random.default_rng(seed), np.random.default_rng(seed + 1))
`;

export const NGRAM_LM_NOTEBOOK_IMPL_BLOCK = String.raw`
    from typing import Any
    
    import numpy as np
    
    from comfy_research.engine.toy_language_common import dataset_rng_seed, scalar_float, scalar_int, slice_shifted_window_lm
    
    
    def _build_transition_tables(
        rng_table: np.random.Generator,
        vocab_size: int,
        order: int,
        dirichlet_alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (pi0, probs, effective_markov_order)."""
        v = max(2, int(vocab_size))
        n_order = max(2, int(order))
        max_states = 65_536
        while n_order > 2 and v ** (n_order - 1) > max_states:
            n_order -= 1
        alpha = max(1e-3, float(dirichlet_alpha))
        num_states = int(round(v ** (n_order - 1)))
    
        pi0 = np.ones((v,), dtype=np.float64) / float(v)
        logits_states = rng_table.standard_normal((num_states, v)).astype(np.float64)
        alpha_vec = np.full((v,), alpha, dtype=np.float64)
        probs = np.empty_like(logits_states)
        for s in range(num_states):
            row = logits_states[s] * alpha_vec
            row = row - np.max(row)
            exp_row = np.exp(row)
            probs[s] = exp_row / np.sum(exp_row)
    
        return pi0, probs, n_order
    
    
    def _state_index(tokens_tail: np.ndarray, v: int) -> int:
        idx = 0
        for t in tokens_tail:
            idx = idx * v + int(t)
        return idx
    
    
    def build_ngram_lm_arrays(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
        rng_train: np.random.Generator,
        rng_table_train: np.random.Generator,
        rng_test: np.random.Generator,
        rng_table_test: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = max(2, scalar_int(data.get("vocabSize"), 32))
        ctx = max(1, scalar_int(data.get("contextLength"), 16))
        order = max(2, scalar_int(data.get("orderN"), 3))
        alpha = scalar_float(data.get("dirichletAlpha"), 1.0)
        target_len = ctx + 1
    
        def split(n: int, rng_samp: np.random.Generator, rng_tab: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                return np.zeros((0, ctx), dtype=np.int64), np.zeros((0, ctx), dtype=np.int64)
            pi0, probs, n_order = _build_transition_tables(rng_tab, v, order, alpha)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                seq = [int(rng_samp.choice(v, p=pi0))]
                while len(seq) < n_order - 1:
                    seq.append(int(rng_samp.integers(0, v)))
                while len(seq) < target_len:
                    tail = np.asarray(seq[-(n_order - 1) :], dtype=np.int64)
                    if tail.size < n_order - 1:
                        tail = np.pad(tail, (n_order - 1 - tail.size, 0), mode="constant", constant_values=0)
                    sidx = _state_index(tail, v)
                    if sidx >= probs.shape[0]:
                        sidx = sidx % probs.shape[0]
                    nxt = int(rng_samp.choice(v, p=probs[sidx]))
                    seq.append(nxt)
                rows[i] = np.asarray(seq[:target_len], dtype=np.int64)
            return slice_shifted_window_lm(rows, ctx)
    
        x_tr, y_tr = split(train_n, rng_train, rng_table_train)
        x_te, y_te = split(test_n, rng_test, rng_table_test)
        return x_tr, y_tr, x_te, y_te
    
    
    def build_ngram_lm_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return build_ngram_lm_arrays(
            data,
            train_n,
            test_n,
            np.random.default_rng(seed),
            np.random.default_rng(seed + 11),
            np.random.default_rng(seed + 1),
            np.random.default_rng(seed + 111),
        )
`;

export const FORMAL_SUITE_LM_NOTEBOOK_IMPL_BLOCK = String.raw`
    from typing import Any
    
    import numpy as np
    
    from comfy_research.engine.toy_language_common import dataset_rng_seed, resize_sequence, scalar_int, scalar_str, slice_shifted_window_lm
    
    
    def _sample_formal_row(
        rng: np.random.Generator,
        lang: str,
        vocab_size: int,
        context_length: int,
    ) -> np.ndarray:
        v = max(2, int(vocab_size))
        ctx = max(1, int(context_length))
        target_len = ctx + 1
        lang_l = lang.strip().lower()
    
        if lang_l == "anbn":
            max_n = max(1, (target_len - 1) // 2)
            n = int(rng.integers(1, max_n + 1))
            core = [0] * n + [1] * n
            seq = resize_sequence(core, target_len, rng, v)
            return seq
    
        if lang_l == "anbncn":
            max_n = max(1, (target_len - 1) // 3)
            n = int(rng.integers(1, max_n + 1))
            core = [0] * n + [1] * n + [2] * n
            seq = resize_sequence(core, target_len, rng, v)
            return seq
    
        if lang_l == "palindrome":
            half_len = max(1, (target_len + 1) // 2)
            half = [int(rng.integers(0, v)) for _ in range(half_len)]
            core = half + half[:-1][::-1] if target_len % 2 == 1 else half + half[::-1]
            seq = np.asarray(core[:target_len], dtype=np.int64)
            if seq.shape[0] < target_len:
                seq = resize_sequence(seq.tolist(), target_len, rng, v)
            return seq[:target_len]
    
        if lang_l == "parity":
            bits = rng.integers(0, min(2, v), size=(target_len - 1,), dtype=np.int64)
            parity = int(np.sum(bits) % min(2, v))
            seq = np.concatenate([bits.astype(np.int64), np.asarray([parity], dtype=np.int64)])
            return seq
    
        # fallback: random
        return rng.integers(0, v, size=(target_len,), dtype=np.int64)
    
    
    def build_formal_suite_lm_arrays(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
        rng_train: np.random.Generator,
        rng_test: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = max(2, scalar_int(data.get("vocabSize"), 8))
        ctx = max(1, scalar_int(data.get("contextLength"), 16))
        lang = scalar_str(data.get("languageType"), "anbn")
        target_len = ctx + 1
    
        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                return np.zeros((0, ctx), dtype=np.int64), np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                rows[i] = _sample_formal_row(rng, lang, v, ctx)
            return slice_shifted_window_lm(rows, ctx)
    
        x_tr, y_tr = split(train_n, rng_train)
        x_te, y_te = split(test_n, rng_test)
        return x_tr, y_tr, x_te, y_te
    
    
    def build_formal_suite_lm_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return build_formal_suite_lm_arrays(data, train_n, test_n, np.random.default_rng(seed), np.random.default_rng(seed + 1))
`;

export const EXTERNAL_TOY_LM_NOTEBOOK_IMPL_BLOCK = String.raw`
    import re
    import urllib.request
    from pathlib import Path
    from typing import Any
    
    import numpy as np
    
    from comfy_research.engine.toy_language_common import dataset_rng_seed, scalar_int, scalar_str, slice_shifted_window_lm
    
    
    SCAN_SIMPLE_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/tasks.txt"
    
    
    def resolve_cache_dir(data: dict[str, Any], subfolder: str) -> Path:
        raw = scalar_str(data.get("cacheDir"), "").strip()
        base = Path(raw).expanduser() if raw else Path.home() / ".cache" / "comfyresearch" / "toy_language" / subfolder
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    
    def download_if_missing(url: str, dest_path: Path) -> Path:
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyResearchToyDataset/1"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — intentional dataset fetch
            dest_path.write_bytes(resp.read())
        return dest_path
    
    
    def _tokenize_text(mode: str, text: str, vocab_cap: int) -> list[int]:
        v = max(2, int(vocab_cap))
        text = text.strip()
        if not text:
            return [0]
        if mode.strip().lower() == "word":
            parts = re.split(r"\\s+", text)
            return [hash(w) % v for w in parts if w]
        # char
        return [ord(c) % v for c in text]
    
    
    def sliding_lm_windows(token_ids: list[int], context_length: int, stride: int, rng: np.random.Generator, max_windows: int) -> np.ndarray:
        """Return shape [n, context_length+1] samples."""
        ctx = max(1, int(context_length))
        L = ctx + 1
        st = max(1, int(stride))
        if len(token_ids) < L:
            pad = token_ids + [0] * (L - len(token_ids))
            row = np.asarray(pad[:L], dtype=np.int64)
            return row.reshape(1, L)
    
        starts = list(range(0, len(token_ids) - L + 1, st))
        if not starts:
            starts = [0]
        if len(starts) > max_windows:
            choice = rng.choice(len(starts), size=max_windows, replace=False)
            starts = [starts[int(i)] for i in sorted(choice)]
        elif len(starts) > max_windows * 2:
            starts = starts[:max_windows]
    
        rows = np.empty((len(starts), L), dtype=np.int64)
        for i, s in enumerate(starts):
            rows[i] = np.asarray(token_ids[s : s + L], dtype=np.int64)
        return rows
    
    
    def build_synthetic_tinystories_documents(rng: np.random.Generator, n_docs: int) -> list[str]:
        subjects = ["a bunny", "the sun", "Sam", "a frog", "Emma", "the moon"]
        verbs = ["likes", "finds", "sees", "helps", "plays with"]
        objs = ["a red ball", "flowers", "friends", "the lake", "a toy"]
        places = ["at home", "in the garden", "by the river", "near school"]
        conns = ["Then", "After that", "Soon", "Later"]
        docs: list[str] = []
        for _ in range(max(4, n_docs)):
            parts: list[str] = []
            n_sent = int(rng.integers(2, 6))
            for _s in range(n_sent):
                parts.append(
                    f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objs)} {rng.choice(places)}.".capitalize()
                )
                if _s + 1 < n_sent and float(rng.random()) < 0.55:
                    parts.append(f"{rng.choice(conns)} ")
            docs.append(" ".join(parts))
        return docs
    
    
    def build_synthetic_phi_style_documents(rng: np.random.Generator, n_docs: int, domain_mix: str) -> list[str]:
        docs: list[str] = []
        mix = domain_mix.strip().lower()
        for _ in range(max(4, n_docs)):
            r = float(rng.random())
            pick = mix
            if mix == "mixed":
                pick = "textbook" if r < 0.45 else ("qa" if r < 0.75 else "code")
            if pick == "qa":
                a, b = int(rng.integers(1, 20)), int(rng.integers(1, 20))
                docs.append(f"Question: What is {a} plus {b}? Answer: The sum is {a + b}.")
            elif pick == "code":
                x = int(rng.integers(0, 10))
                docs.append(f"def inc(z):\\n    return z + {x}\\n\\nassert inc(3) == {3 + x}")
            else:
                docs.append(
                    f"Definition {int(rng.integers(1, 99))}: A linear map satisfies f(ax)=af(x). "
                    f"Example {int(rng.integers(1, 20))}: Short exercises reinforce structure."
                )
        return docs
    
    
    def load_or_synthetic_corpus_lm(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
        rng_train: np.random.Generator,
        rng_test: np.random.Generator,
        *,
        corpus_kind: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = max(1, scalar_int(data.get("contextLength"), 64))
        seq_doc = max(ctx + 5, scalar_int(data.get("seqLen"), 256))
        stride = max(1, scalar_int(data.get("stride"), ctx + 1))
        vocab_cap = max(2, scalar_int(data.get("vocabCap"), 256))
        tok_mode = scalar_str(data.get("tokenizerMode"), "char")
        data_src = scalar_str(data.get("dataSource"), "synthetic").strip().lower()
    
        docs: list[str] = []
        if data_src == "download" and corpus_kind == "tinystories":
            cache = resolve_cache_dir(data, "tinystories")
            dest = cache / "tiny_sample.txt"
            try:
                url = scalar_str(data.get("tinyStoriesUrl"), "").strip() or scalar_str(data.get("downloadUrl"), "").strip()
                if url:
                    download_if_missing(url, dest)
                if dest.exists():
                    docs = dest.read_text(encoding="utf-8", errors="ignore").split("\\n\\n")[:2000]
            except Exception:
                docs = []
            if not docs:
                docs = build_synthetic_tinystories_documents(rng_train, max(8, train_n // max(1, seq_doc // 32)))
        elif data_src == "download" and corpus_kind == "phi_style":
            docs = build_synthetic_phi_style_documents(
                rng_train,
                max(8, train_n // max(1, seq_doc // 64)),
                scalar_str(data.get("domainMix"), "mixed"),
            )
        else:
            if corpus_kind == "tinystories":
                docs = build_synthetic_tinystories_documents(rng_train, max(8, train_n // max(1, seq_doc // 32)))
            else:
                docs = build_synthetic_phi_style_documents(
                    rng_train,
                    max(8, train_n // max(1, seq_doc // 64)),
                    scalar_str(data.get("domainMix"), "mixed"),
                )
    
        flat_chunks: list[int] = []
        budget = seq_doc * len(docs)
        for d in docs:
            toks = _tokenize_text(tok_mode, d[: max(4000, seq_doc)], vocab_cap)
            flat_chunks.extend(toks[:seq_doc])
            if len(flat_chunks) >= budget:
                break
        if not flat_chunks:
            flat_chunks = [0]
    
        rows_tr = sliding_lm_windows(flat_chunks, ctx, stride, rng_train, max(train_n, 1))
        if train_n > 0 and rows_tr.shape[0] > train_n:
            idx = rng_train.choice(rows_tr.shape[0], size=train_n, replace=False)
            rows_tr = rows_tr[idx]
        elif train_n > 0 and 0 < rows_tr.shape[0] < train_n:
            idx = rng_train.integers(0, rows_tr.shape[0], size=train_n)
            rows_tr = rows_tr[idx]
        rows_te = sliding_lm_windows(flat_chunks, ctx, stride, rng_test, max(test_n, 1))
        if test_n > 0 and rows_te.shape[0] > test_n:
            idx = rng_test.choice(rows_te.shape[0], size=test_n, replace=False)
            rows_te = rows_te[idx]
        elif test_n > 0 and 0 < rows_te.shape[0] < test_n:
            idx = rng_test.integers(0, rows_te.shape[0], size=test_n)
            rows_te = rows_te[idx]
    
        x_tr, y_tr = slice_shifted_window_lm(rows_tr, ctx) if rows_tr.size else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        x_te, y_te = slice_shifted_window_lm(rows_te, ctx) if rows_te.size else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        return x_tr, y_tr, x_te, y_te
    
    
    def _scan_parse_lines(text: str) -> list[tuple[list[str], list[str]]]:
        pairs: list[tuple[list[str], list[str]]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if "IN:" in line and "OUT:" in line:
                inp_part = line.split("OUT:")[0].replace("IN:", "").strip()
                out_part = line.split("OUT:", 1)[1].strip()
                pairs.append((inp_part.split(), out_part.split()))
        return pairs
    
    
    def build_scan_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = max(1, scalar_int(data.get("contextLength"), 24))
        v = max(8, scalar_int(data.get("vocabSize"), 64))
        seed = dataset_rng_seed(data)
        rng_tr = np.random.default_rng(seed)
        rng_te = np.random.default_rng(seed + 1)
        src = scalar_str(data.get("dataSource"), "synthetic").strip().lower()
    
        cmds_pool = ["jump", "walk", "look", "run", "turn", "left", "right", "twice", "thrice", "opposite"]
        acts_pool = ["JUMP", "WALK", "LOOK", "RUN", "LEFT", "RIGHT"]
    
        def encode(ws: list[str]) -> list[int]:
            return [1 + (hash(w) % (v - 2)) for w in ws]
    
        rows_tr: list[np.ndarray] = []
        rows_te: list[np.ndarray] = []
    
        def synth_pair(rng: np.random.Generator) -> tuple[list[int], int]:
            n_cmd = int(rng.integers(1, min(6, ctx)))
            cmd_words = [str(rng.choice(cmds_pool)) for _ in range(n_cmd)]
            first_act = str(rng.choice(acts_pool))
            inp_ids = encode(cmd_words)
            target = 1 + (hash(first_act) % (v - 2))
            padded_inp = inp_ids[:ctx]
            while len(padded_inp) < ctx:
                padded_inp.append(0)
            seq = padded_inp + [target]
            return seq[: ctx + 1], target
    
        if src == "download":
            cache = resolve_cache_dir(data, "scan")
            dest = cache / "tasks.txt"
            try:
                download_if_missing(scalar_str(data.get("scanUrl"), SCAN_SIMPLE_URL) or SCAN_SIMPLE_URL, dest)
                pairs = _scan_parse_lines(dest.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pairs = []
    
            def rows_from_pairs(pairs_subset: list[tuple[list[str], list[str]]], n: int, rng: np.random.Generator) -> list[np.ndarray]:
                out: list[np.ndarray] = []
                if not pairs_subset:
                    return out
                for _ in range(n):
                    cmd, act = pairs_subset[int(rng.integers(0, len(pairs_subset)))]
                    inp_ids = encode(cmd[:ctx])
                    while len(inp_ids) < ctx:
                        inp_ids.append(0)
                    first_act = act[0] if act else "JUMP"
                    target = 1 + (hash(first_act) % (v - 2))
                    seq = inp_ids[:ctx] + [target]
                    out.append(np.asarray(seq[: ctx + 1], dtype=np.int64))
                return out
    
            if pairs:
                perm = rng_tr.permutation(len(pairs))
                n_tr = max(1, int(len(pairs) * 0.9))
                tr_pairs = [pairs[int(i)] for i in perm[:n_tr]]
                te_pairs = [pairs[int(i)] for i in perm[n_tr:]]
                rows_tr = rows_from_pairs(tr_pairs, train_n, rng_tr)
                rows_te = rows_from_pairs(te_pairs if te_pairs else tr_pairs, test_n, rng_te)
            else:
                src = "synthetic"
    
        if src != "download" or not rows_tr:
            rows_tr = []
            for _ in range(train_n):
                seq, _ = synth_pair(rng_tr)
                rows_tr.append(np.asarray(seq, dtype=np.int64))
            rows_te = []
            for _ in range(test_n):
                seq, _ = synth_pair(rng_te)
                rows_te.append(np.asarray(seq, dtype=np.int64))
    
        x_tr, y_tr = slice_shifted_window_lm(np.stack(rows_tr, axis=0), ctx) if rows_tr else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        x_te, y_te = slice_shifted_window_lm(np.stack(rows_te, axis=0), ctx) if rows_te else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        return x_tr, y_tr, x_te, y_te
    
    
    def build_cogs_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = max(1, scalar_int(data.get("contextLength"), 32))
        v = max(16, scalar_int(data.get("vocabSize"), 128))
        seed = dataset_rng_seed(data)
        rng_tr = np.random.default_rng(seed)
        rng_te = np.random.default_rng(seed + 1)
    
        nouns = ["cat", "dog", "teacher", "student", "cake"]
        verbs = ["chased", "saw", "liked"]
        det = ["the", "a"]
    
        def synth_row(rng: np.random.Generator) -> np.ndarray:
            sent = f"{rng.choice(det)} {rng.choice(nouns)} {rng.choice(verbs)} {rng.choice(det)} {rng.choice(nouns)}"
            lf = f"* chase ( {rng.choice(nouns)} , {rng.choice(nouns)} )"
            wi = [1 + (hash(w) % (v - 2)) for w in sent.split()]
            wi = wi[:ctx]
            while len(wi) < ctx:
                wi.append(0)
            tgt = 1 + (hash(lf.split()[0]) % (v - 2))
            seq = wi + [tgt]
            return np.asarray(seq[: ctx + 1], dtype=np.int64)
    
        rows_tr = [synth_row(rng_tr) for _ in range(train_n)]
        rows_te = [synth_row(rng_te) for _ in range(test_n)]
        x_tr, y_tr = slice_shifted_window_lm(np.stack(rows_tr, axis=0), ctx) if rows_tr else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        x_te, y_te = slice_shifted_window_lm(np.stack(rows_te, axis=0), ctx) if rows_te else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        return x_tr, y_tr, x_te, y_te
    
    
    def build_listops_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = max(1, scalar_int(data.get("contextLength"), 48))
        v = max(16, scalar_int(data.get("vocabSize"), 64))
        seed = dataset_rng_seed(data)
        rng_tr = np.random.default_rng(seed)
        rng_te = np.random.default_rng(seed + 1)
    
        def eval_min(vals: list[int]) -> int:
            return min(vals) % v
    
        def synth_row(rng: np.random.Generator) -> np.ndarray:
            inner = [int(rng.integers(0, min(20, v))), int(rng.integers(0, min(20, v)))]
            ans = int(eval_min(inner))
            t_min = min(v - 1, 12)
            t_lb = min(v - 1, 13)
            t_rb = min(v - 1, 14)
            core = [t_lb, t_min, inner[0] % v, inner[1] % v, t_rb]
            seq_list = (core + [0] * (ctx + 1))[: ctx + 1]
            seq_list[-1] = ans % v
            return np.asarray(seq_list, dtype=np.int64)
    
        rows_tr = [synth_row(rng_tr) for _ in range(train_n)]
        rows_te = [synth_row(rng_te) for _ in range(test_n)]
        x_tr, y_tr = slice_shifted_window_lm(np.stack(rows_tr, axis=0), ctx) if rows_tr else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        x_te, y_te = slice_shifted_window_lm(np.stack(rows_te, axis=0), ctx) if rows_te else (np.zeros((0, ctx), np.int64), np.zeros((0, ctx), np.int64))
        return x_tr, y_tr, x_te, y_te
    
    
    def tiny_stories_lm_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return load_or_synthetic_corpus_lm(data, train_n, test_n, np.random.default_rng(seed), np.random.default_rng(seed + 1), corpus_kind="tinystories")
    
    
    def phi_style_lm_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return load_or_synthetic_corpus_lm(data, train_n, test_n, np.random.default_rng(seed), np.random.default_rng(seed + 1), corpus_kind="phi_style")
`;

export const PHYSICS_LM_NOTEBOOK_IMPL_BLOCK = String.raw`
    """Synthetic Physics-of-LLM-style token LM datasets (biography, tuples, multi-hop, playground presets)."""
    
    from __future__ import annotations
    
    from typing import Any
    
    import numpy as np
    
    from comfy_research.engine.toy_language_common import dataset_rng_seed, resize_sequence, scalar_float, scalar_int, scalar_str, slice_shifted_window_lm
    
    
    def _num_special_reserve() -> int:
        """Reserve low token ids for structure markers."""
        return 24
    
    
    def _sample_entity_tok(rng: np.random.Generator, v: int, lo: int) -> int:
        return int(rng.integers(lo, max(lo + 1, v)))
    
    
    def _biography_sequence(
        rng: np.random.Generator,
        v: int,
        aug: str,
        noise_prob: float,
    ) -> list[int]:
        """One biography block as token ids; specials in [0, S), entities in [S, v)."""
        s0 = _num_special_reserve()
        # specials (fixed meanings)
        BOS, EOS = 1, 2
        T_NAME, T_BORN, T_CITY = 3, 4, 5
        SEP = 6
    
        name_t = _sample_entity_tok(rng, v, s0)
        year_t = _sample_entity_tok(rng, v, s0)
        city_t = _sample_entity_tok(rng, v, s0)
        blocks = [
            [T_NAME, name_t],
            [T_BORN, year_t],
            [T_CITY, city_t],
        ]
        if aug.strip().lower() in ("shuffle_fields", "shuffle", "permute"):
            rng.shuffle(blocks)
        seq = [BOS]
        for b in blocks:
            seq.extend(b)
            seq.append(SEP)
        if aug.strip().lower() in ("noise_slots", "noise") and float(noise_prob) > 0.0:
            for i in range(len(seq)):
                if seq[i] >= s0 and float(rng.random()) < float(noise_prob):
                    seq[i] = _sample_entity_tok(rng, v, s0)
        seq.append(EOS)
        return seq
    
    
    def build_biography_lm_arrays_from_seed(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = max(_num_special_reserve() + 2, scalar_int(data.get("vocabSize"), 64))
        ctx = max(1, scalar_int(data.get("contextLength"), 32))
        aug = scalar_str(data.get("biographyAugmentation"), "template")
        noise_p = float(np.clip(scalar_float(data.get("slotNoiseProb"), 0.0), 0.0, 1.0))
        target_len = ctx + 1
        seed = dataset_rng_seed(data)
    
        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                z = np.zeros((0, ctx), dtype=np.int64)
                return z, np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                toks: list[int] = []
                while len(toks) < target_len:
                    toks.extend(_biography_sequence(rng, v, aug, noise_p))
                rows[i] = resize_sequence(toks, target_len, rng, v)
            return slice_shifted_window_lm(rows, ctx)
    
        r0 = np.random.default_rng(seed)
        r1 = np.random.default_rng(seed + 1)
        x_tr, y_tr = split(train_n, r0)
        x_te, y_te = split(test_n, r1)
        return x_tr, y_tr, x_te, y_te
    
    
    def _relation_sequence_forward(rng: np.random.Generator, v: int) -> list[int]:
        s0 = _num_special_reserve()
        BOS, EOS = 1, 2
        T_SUB, T_REL, T_OBJ = 7, 8, 9
        SEP = 6
        sub = _sample_entity_tok(rng, v, s0)
        rel = _sample_entity_tok(rng, v, s0)
        obj = _sample_entity_tok(rng, v, s0)
        return [BOS, T_SUB, sub, T_REL, rel, T_OBJ, obj, SEP, EOS]
    
    
    def _relation_sequence_inverse(rng: np.random.Generator, v: int) -> list[int]:
        """Query gives OBJ and REL; answer SUB at end (next-token LM learns to emit SUB)."""
        s0 = _num_special_reserve()
        BOS, EOS = 1, 2
        T_SUB, T_REL, T_OBJ = 7, 8, 9
        T_Q = 10
        SEP = 6
        sub = _sample_entity_tok(rng, v, s0)
        rel = _sample_entity_tok(rng, v, s0)
        obj = _sample_entity_tok(rng, v, s0)
        return [BOS, T_Q, T_OBJ, obj, T_REL, rel, T_SUB, sub, SEP, EOS]
    
    
    def build_relation_tuple_lm_arrays_from_seed(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = max(_num_special_reserve() + 2, scalar_int(data.get("vocabSize"), 64))
        ctx = max(1, scalar_int(data.get("contextLength"), 24))
        mode = scalar_str(data.get("relationMode"), "forward").strip().lower()
        target_len = ctx + 1
        seed = dataset_rng_seed(data)
    
        def one_seq(rng: np.random.Generator) -> list[int]:
            if mode in ("inverse", "inv"):
                return _relation_sequence_inverse(rng, v)
            return _relation_sequence_forward(rng, v)
    
        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                z = np.zeros((0, ctx), dtype=np.int64)
                return z, np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                toks: list[int] = []
                while len(toks) < target_len:
                    toks.extend(one_seq(rng))
                rows[i] = resize_sequence(toks, target_len, rng, v)
            return slice_shifted_window_lm(rows, ctx)
    
        r0 = np.random.default_rng(seed)
        r1 = np.random.default_rng(seed + 1)
        x_tr, y_tr = split(train_n, r0)
        x_te, y_te = split(test_n, r1)
        return x_tr, y_tr, x_te, y_te
    
    
    def build_multi_hop_fact_chain_lm_arrays_from_seed(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linear chain of binary facts + query; target token is final entity."""
        v = max(_num_special_reserve() + 2, scalar_int(data.get("vocabSize"), 96))
        ctx = max(1, scalar_int(data.get("contextLength"), 48))
        hops = max(1, scalar_int(data.get("chainHops"), 3))
        target_len = ctx + 1
        seed = dataset_rng_seed(data)
        s0 = _num_special_reserve()
        BOS, EOS = 1, 2
        T_ENT, T_REL, T_Q, T_ANS = 11, 12, 13, 14
        SEP = 6
    
        def one_chain(rng: np.random.Generator) -> list[int]:
            ents = [_sample_entity_tok(rng, v, s0) for _ in range(hops + 1)]
            seq: list[int] = [BOS]
            for i in range(hops):
                seq.extend([T_ENT, ents[i], T_REL, _sample_entity_tok(rng, v, s0), T_ENT, ents[i + 1], SEP])
            seq.extend([T_Q, T_ENT, ents[0], T_ANS, ents[-1], EOS])
            return seq
    
        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                z = np.zeros((0, ctx), dtype=np.int64)
                return z, np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                toks: list[int] = []
                while len(toks) < target_len:
                    toks.extend(one_chain(rng))
                rows[i] = resize_sequence(toks, target_len, rng, v)
            return slice_shifted_window_lm(rows, ctx)
    
        r0 = np.random.default_rng(seed)
        r1 = np.random.default_rng(seed + 1)
        x_tr, y_tr = split(train_n, r0)
        x_te, y_te = split(test_n, r1)
        return x_tr, y_tr, x_te, y_te
    
    
    def _playground_depo(rng: np.random.Generator, v: int, window: int, length: int) -> list[int]:
        """Parity of previous \`\`window\`\` tokens (binary core) mapped into vocab."""
        w = max(2, min(window, 12))
        bits = [int(rng.integers(0, 2)) for _ in range(w)]
        tok_lo = max(16, min(v // 2, v - 2))
        out = [tok_lo + b for b in bits]
        for _ in range(length - w):
            nb = sum(bits[-w:]) % 2
            bits.append(nb)
            out.append(tok_lo + nb)
        return out[:length]
    
    
    def _playground_brevo(rng: np.random.Generator, v: int, length: int) -> list[int]:
        """Keyed copy: marker then filler then repeat segment from early positions."""
        marker = 3
        gap = int(rng.integers(2, max(3, min(length // 4, 16))))
        seg_len = max(2, min(8, length // 4))
        head = [int(rng.integers(16, max(17, v))) for _ in range(seg_len)]
        seq = head + [marker] + [int(rng.integers(16, max(17, v))) for _ in range(gap)]
        # copy head after gap region
        seq.extend(head[: min(seg_len, length - len(seq))])
        while len(seq) < length:
            seq.append(int(rng.integers(0, max(1, v))))
        return seq[:length]
    
    
    def _playground_mano(rng: np.random.Generator, v: int, mod: int, length: int) -> list[int]:
        """Reveal counter state + symbol; deterministic transitions."""
        m = max(2, min(mod, 64))
        sym_lo = 16
        state = int(rng.integers(0, m))
        seq: list[int] = []
        for _ in range(length):
            sym = int(rng.integers(sym_lo, max(sym_lo + 1, v)))
            seq.append((state * 7 + sym) % v)
            state = (state + sym + 1) % m
        return seq
    
    
    def _playground_capo(rng: np.random.Generator, v: int, length: int) -> list[int]:
        """Short biography-like snippets concatenated."""
        toks: list[int] = []
        while len(toks) < length:
            toks.extend(_biography_sequence(rng, v, "template", 0.0))
        return toks[:length]
    
    
    def _playground_lano(rng: np.random.Generator, v: int, depth: int, length: int) -> list[int]:
        """Nested brackets: multiple bracket types in Dyck-like fashion."""
        k = max(1, min(depth, 8))
        lo = 16
        pairs = [(lo + 2 * i, lo + 2 * i + 1) for i in range(k)]
    
        def gen(d: int) -> list[int]:
            if d <= 0 or float(rng.random()) < 0.45:
                return []
            inner = gen(d - 1)
            open_c, close_c = pairs[int(rng.integers(0, k))]
            return [open_c] + inner + [close_c]
    
        out = gen(max(2, min(depth, 6)))
        while len(out) < length:
            out.extend(gen(max(2, min(depth, 6))))
        return resize_sequence(out, length, rng, v).tolist()
    
    
    def build_synthetic_playground_lm_arrays_from_seed(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = max(32, scalar_int(data.get("vocabSize"), 64))
        ctx = max(1, scalar_int(data.get("contextLength"), 32))
        family = scalar_str(data.get("playgroundFamily"), "depo").strip().lower()
        target_len = ctx + 1
        seed = dataset_rng_seed(data)
        depo_w = max(2, scalar_int(data.get("depoWindow"), 4))
        mano_mod = max(2, scalar_int(data.get("manoModulus"), 17))
        lano_depth = max(2, scalar_int(data.get("lanoNestingDepth"), 4))
    
        def gen_tokens(rng: np.random.Generator) -> list[int]:
            if family in ("brevo", "long_range", "copy"):
                return _playground_brevo(rng, v, target_len)
            if family in ("mano", "state", "fsa"):
                return _playground_mano(rng, v, mano_mod, target_len)
            if family in ("capo", "facts", "knowledge"):
                return _playground_capo(rng, v, target_len)
            if family in ("lano", "nested", "hier"):
                return _playground_lano(rng, v, lano_depth, target_len)
            # depo default
            return _playground_depo(rng, v, depo_w, target_len)
    
        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                z = np.zeros((0, ctx), dtype=np.int64)
                return z, np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                rows[i] = np.asarray(gen_tokens(rng), dtype=np.int64)
            return slice_shifted_window_lm(rows, ctx)
    
        r0 = np.random.default_rng(seed)
        r1 = np.random.default_rng(seed + 1)
        x_tr, y_tr = split(train_n, r0)
        x_te, y_te = split(test_n, r1)
        return x_tr, y_tr, x_te, y_te
`;
