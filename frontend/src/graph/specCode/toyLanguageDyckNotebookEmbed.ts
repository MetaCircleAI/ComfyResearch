/**
 * Verbatim Dyck LM sampling logic from ``comfy_research/engine/toy_language_dyck_runtime.py``.
 * Keep in sync when editing that file — this is surfaced in the Code notebook for Dyck nodes.
 *
 * Indented with 4 spaces so it can be placed inside ``def <specName>():`` in generated Python.
 */
export const DYCK_LM_NOTEBOOK_IMPL_BLOCK = String.raw`    import numpy as np
    from typing import Any

    from comfy_research.engine.toy_language_common import dataset_rng_seed, scalar_int, slice_shifted_window_lm

    def _sample_dyck_row(
        rng: np.random.Generator,
        k: int,
        target_len: int,
        v: int,
        max_nesting: int,
    ) -> np.ndarray:
        """Generate \`\`target_len\`\` tokens: balanced Dyck prefix on bracket ids, then neutral fillers.

        \`\`max_nesting\`\` > 0 caps how many unmatched opens may sit on the stack at once (nesting depth).
        \`\`max_nesting\`\` <= 0 means no explicit cap (still bounded by prefix length).
        """
        open_ids = np.minimum(np.arange(k, dtype=np.int64) * 2, v - 1)
        close_ids = np.minimum(open_ids + 1, v - 1)

        even_prefix = target_len if target_len % 2 == 0 else target_len - 1
        seq: list[int] = []
        stack: list[int] = []

        while len(seq) < even_prefix:
            rem = even_prefix - len(seq)
            must_close = len(stack) > 0 and rem <= len(stack)
            cannot_open = max_nesting > 0 and len(stack) >= max_nesting
            if must_close or cannot_open or (len(stack) > 0 and float(rng.random()) < 0.48):
                bt = stack.pop()
                seq.append(int(close_ids[int(bt)]))
            else:
                bt = int(rng.integers(0, k))
                seq.append(int(open_ids[bt]))
                stack.append(bt)

        while stack and len(seq) < even_prefix:
            bt = stack.pop()
            seq.append(int(close_ids[int(bt)]))

        seq = seq[:even_prefix]

        filler_lo = min(v - 1, 2 * k)
        while len(seq) < target_len:
            lo, hi = filler_lo, v
            if lo >= hi - 1:
                lo = 0
            seq.append(int(rng.integers(lo, hi)))

        return np.asarray(seq[:target_len], dtype=np.int64)

    def build_dyck_lm_arrays(
        data: dict[str, Any],
        train_n: int,
        test_n: int,
        rng_train: np.random.Generator,
        rng_test: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = max(1, scalar_int(data.get("contextLength"), 24))
        k_bracket = max(1, scalar_int(data.get("numBracketTypes"), 1))
        v = 2 * k_bracket
        k_eff = k_bracket
        target_len = ctx + 1
        max_nesting = scalar_int(data.get("maxNestingDepth"), 0)

        def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            if n <= 0:
                return np.zeros((0, ctx), dtype=np.int64), np.zeros((0, ctx), dtype=np.int64)
            rows = np.empty((n, target_len), dtype=np.int64)
            for i in range(n):
                rows[i] = _sample_dyck_row(rng, k_eff, target_len, v, max_nesting)
            return slice_shifted_window_lm(rows, ctx)

        x_tr, y_tr = split(train_n, rng_train)
        x_te, y_te = split(test_n, rng_test)
        return x_tr, y_tr, x_te, y_te

    def build_dyck_lm_arrays_from_seed(data: dict[str, Any], train_n: int, test_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = dataset_rng_seed(data)
        return build_dyck_lm_arrays(data, train_n, test_n, np.random.default_rng(seed), np.random.default_rng(seed + 1))
`;
