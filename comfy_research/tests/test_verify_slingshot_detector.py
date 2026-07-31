"""verify_slingshot_repro.detect_slingshots — 纯判据器单测(合成序列)。

真实图跑通由 tools/verify_slingshot_repro.py 实跑背书(PR 附实测 JSON);
此处锚判据本身:acc 门、尖峰比值、尖峰前 norm 增长、跨周期阶梯、去重推进。
"""
from __future__ import annotations

import math

from comfy_research.tools.verify_slingshot_repro import detect_slingshots


def _flat(n: int, v: float) -> list[float]:
    return [v] * n


def _synthetic_two_cycles() -> tuple[list[float], list[float], list[float]]:
    """acc 在 idx 10 达 1.0;尖峰在 ~100 与 ~200,尖峰前 norm 线性爬升。"""
    n = 300
    loss = [1e-8] * n
    for i in range(10):
        loss[i] = 1.0  # 训练早期
    loss[100] = 1.0  # ratio 1e8
    loss[200] = 1.0
    norm = [1.0 + 0.001 * i for i in range(n)]  # 单调爬升(尖峰前增长恒正)
    acc = [0.5] * 10 + [1.0] * (n - 10)
    return loss, norm, acc


def test_detects_two_qualified_cycles_and_staircase() -> None:
    loss, norm, acc = _synthetic_two_cycles()
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert det["first_perfect_acc_index"] == 10
    assert det["qualified_cycles"] == 2
    assert det["norm_staircase"] is True
    assert [c["spike_index"] for c in det["cycles"]] == [100, 200]
    assert all(c["ratio"] >= 1e4 for c in det["cycles"])


def test_no_cycles_before_perfect_accuracy() -> None:
    loss, norm, _ = _synthetic_two_cycles()
    acc = _flat(len(loss), 0.9)  # 永不达 1.0 → 弹弓不计
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert det["first_perfect_acc_index"] is None
    assert det["qualified_cycles"] == 0


def test_small_spikes_below_ratio_do_not_count() -> None:
    n = 300
    loss = [1e-8] * n
    loss[100] = 1e-6  # ratio 1e2 < 1e4
    norm = [1.0 + 0.001 * i for i in range(n)]
    acc = _flat(n, 1.0)
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert det["qualified_cycles"] == 0


def test_flat_norm_spike_is_not_qualified() -> None:
    n = 300
    loss = [1e-8] * n
    loss[100] = 1.0
    norm = _flat(n, 2.0)  # 无增长
    acc = _flat(n, 1.0)
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert len(det["cycles"]) == 1
    assert det["qualified_cycles"] == 0
    assert det["norm_staircase"] is False


def test_zero_loss_floor_does_not_disable_detection() -> None:
    """float32 晚期 loss 下溢到精确 0 后,后续尖峰仍须被检出(12k 步实跑回归)。"""
    n = 400
    loss = [1e-8] * 100 + [0.0] * 100 + [1e-9] * 200
    loss[150] = 1.0  # trough 已是精确 0 —— 旧守卫在此永久关闭检测
    loss[300] = 1.0
    norm = [1.0 + 0.001 * i for i in range(n)]
    acc = _flat(n, 1.0)
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert det["qualified_cycles"] == 2
    assert [c["spike_index"] for c in det["cycles"]] == [150, 300]


def test_zero_trough_then_nonzero_floor_keeps_detecting() -> None:
    """review  形状:精确 0 trough 后地板回到持续 1e-9,后续尖峰不得被跳过。"""
    n = 400
    loss = [0.0] * 100 + [1e-9] * 300
    loss[150] = 1.0
    loss[300] = 1.0
    norm = [1.0 + 0.001 * i for i in range(n)]
    acc = _flat(n, 1.0)
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert [c["spike_index"] for c in det["cycles"]] == [150, 300]
    assert det["qualified_cycles"] == 2


def test_nan_tolerant() -> None:
    loss, norm, acc = _synthetic_two_cycles()
    loss[50] = math.nan
    norm[51] = math.nan
    det = detect_slingshots(loss, norm, acc, 1e4)
    assert det["qualified_cycles"] == 2
