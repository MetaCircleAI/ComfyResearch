"""Toy benchmarks / corpus loaders with synthetic fallback (SCAN-style, ListOps-like, TinyStories-style, phi-style)."""

from __future__ import annotations

import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from comfy_research.engine.datasets.toy_language_common import dataset_rng_seed, scalar_int, scalar_str, slice_shifted_window_lm


SCAN_SIMPLE_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/tasks.txt"
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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
        parts = re.split(r"\s+", text)
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
            docs.append(f"def inc(z):\n    return z + {x}\n\nassert inc(3) == {3 + x}")
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
                docs = dest.read_text(encoding="utf-8", errors="ignore").split("\n\n")[:2000]
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


def tiny_shakespeare_lm_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Real TinyShakespeare windows with a corpus-derived word vocabulary.

    This intentionally raises on a failed download instead of silently using a
    synthetic corpus: a reproduction template must not change data provenance.
    """
    context = max(1, scalar_int(data.get("contextLength"), 32))
    vocab_size = max(8, scalar_int(data.get("vocabSize"), 256))
    stride = max(1, scalar_int(data.get("stride"), 1))
    seed = dataset_rng_seed(data)
    cache = resolve_cache_dir(data, "tinyshakespeare")
    corpus_path = download_if_missing(TINY_SHAKESPEARE_URL, cache / "input.txt")
    text = corpus_path.read_text(encoding="utf-8", errors="ignore").lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    if not words:
        raise RuntimeError("TinyShakespeare download contained no word tokens.")
    # 0 PAD, 1 BOS, 2 EOS, 3 UNK; the remaining IDs are empirical top-frequency words.
    word_to_id = {word: index + 4 for index, (word, _) in enumerate(Counter(words).most_common(vocab_size - 4))}
    token_ids = [word_to_id.get(word, 3) for word in words]
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)
    rows_train = sliding_lm_windows(token_ids, context, stride, rng_train, max(train_n, 1))
    rows_test = sliding_lm_windows(token_ids, context, stride, rng_test, max(test_n, 1)) if test_n > 0 else np.zeros((0, context + 1), dtype=np.int64)
    x_train, y_train = slice_shifted_window_lm(rows_train, context)
    x_test, y_test = slice_shifted_window_lm(rows_test, context) if test_n > 0 else (
        np.zeros((0, context), dtype=np.int64), np.zeros((0, context), dtype=np.int64)
    )
    return x_train, y_train, x_test, y_test
