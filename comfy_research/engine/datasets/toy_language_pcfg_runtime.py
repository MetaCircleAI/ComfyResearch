"""PCFG-style synthetic token sequences for next-token LM.

Two generators (``pcfgGenMode``):

- ``binary_tree`` (default): random full binary tree whose leaves are i.i.d. vocab ids — legacy
  behaviour for sweeps over ``pcfgMaxDepth`` / ``pcfgTermProb``.
- ``cfg_sentence``: weighted productions over explicit nonterminals (NLTK-style PCFG sampling);
  terminals are short word-like tokens mapped to low integer ids. Useful as a **controlled** proxy
  where next-token mass is **not** uniform because expansions share latent structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comfy_research.engine.datasets.toy_language_common import dataset_rng_seed, resize_sequence, scalar_float, scalar_int, slice_shifted_window_lm

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
