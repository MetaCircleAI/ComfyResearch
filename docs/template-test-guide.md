# Comfy Research 实验 Template 测试规范

## 1. 目标

每个保存到 Comfy Research 的实验 Template，都应配套一个测试文件，用于确认：

* Template 仍保持提交时确定的默认实验设置；
* Template 仍能被正常加载和运行；
* 如有必要，其目标科学现象仍能被观察到。

这里的“默认设置”是指：

> **该 Template 自身被提交、确认时采用的 Node、连接和参数。**

它不表示：

* 必须与论文原文完全一致；
* 必须使用某种特定 Dataset 或 Model；
* 必须使用 Node 定义中的系统默认参数。

例如，一个 Grokking Template 可以使用 `p=127`，也可以采用不同于原论文的 Dataset。测试只需要确认它仍然保持这个 Template 原本选择的设置。

---

## 2. Template Baseline Test

### 要求

检查 Template 当前的 Node、连接和关键参数是否仍与原始 Template 设置一致。

测试失败不一定表示新设置错误，而是表示：

> Template 的基准实验设置发生了变化，需要确认这是有意修改还是意外漂移。

### 应检查

* 核心 Node 的类型；
* 核心 Node 之间的连接；
* 会改变实验含义的关键参数；
* 必要的 Observable。

### 不建议检查

* Node 坐标；
* 随机生成的 Node ID；
* `savedAt` 等时间戳；
* 不影响实验含义的界面字段；
* 所有参数的完整 JSON 快照。

### 示例

假设某个 Grokking Template 原本采用：

```text
Dataset: modular_addition_dataset
p: 127
Model: mlp_model
Optimizer: adamw_optimizer
```

那么测试应锁定这套 Template 设置，而不是要求它必须符合某篇论文的原始参数。

```python
def test_template_baseline():
    graph = load_template("grokking_p127")

    dataset = find_node(graph, "dataset")
    model = find_node(graph, "model")
    optimizer = find_node(graph, "optimizer")

    assert dataset.type == "modular_addition_dataset"
    assert dataset.data["p"] == 127
    assert model.type == "mlp_model"
    assert optimizer.type == "adamw_optimizer"

    assert is_connected(graph, dataset, "trainer")
    assert is_connected(graph, model, "trainer")
```

如果 Template 被有意改为 `p=113`，则应同时更新 Template 和测试中的基准值，并在代码审查中说明原因。

---

## 3. Smoke Reproduction Test

### 要求

确认 Template 可以完整经过：

```text
加载 Template
→ 解析 Graph
→ 构建 Dataset
→ 构建 Model、Loss 和 Optimizer
→ 运行 Trainer
→ 产生完成事件
```

CI 中不需要运行完整实验。可以在 Template 副本上缩小：

* training steps；
* train/test size；
* model width；
* batch size。

不得修改磁盘上的原始 Template。

### 示例

```python
def test_template_smoke_run():
    graph = load_template("grokking_p127")
    graph = override_for_ci(
        graph,
        training_steps=5,
        train_size=32,
        test_size=16,
    )

    events = list(run_training(graph))

    assert events[-1]["type"] == "complete"
    assert all(math.isfinite(x) for x in events[-1]["loss_history"])
```

Smoke Test 主要确认实验链路没有断裂，不要求在几步训练内重新产生论文 Figure。

---

## 4. Scientific Behavior Test

### 要求

当 Template 明确用于展示某个科学现象，并且该现象能够以**稳定、低成本**的方式检测时，可以增加科学行为测试。

这类测试建议检查：

* 大小关系；
* 趋势；
* 相变；
* 对称性；
* 单调性；
* 端点或边界性质。

不应依赖某一个浮点数精确相等。

### 示例：Grokking

只有当这个 Template 明确用于展示 Grokking，并且 CI 预算足以稳定出现该现象时，才检查训练后期验证准确率相对于早期明显提高：

```python
def test_grokking_behavior():
    history = run_reduced_reproduction()

    early = np.mean(history.validation_accuracy[:20])
    late = np.mean(history.validation_accuracy[-20:])

    assert late > early + 0.4
```

不建议写成：

```python
assert history.validation_accuracy[-1] == 1.0
```

因为精确结果容易受到 seed、PyTorch 版本和硬件数值误差影响。

如果缩小后的 CI 实验无法稳定表现该现象，应将科学行为测试放到 nightly 测试中，或者只保留 Baseline Test 和 Smoke Test。

---

## 5. 每个 Template 的推荐测试结构

```python
"""
Template:
    grokking_p127

Purpose:
    A Grokking-style experiment using the template's own selected settings.
"""

pytestmark = pytest.mark.repro


def test_template_baseline():
    """Node、连接和关键参数仍符合该 Template 的基准设置."""
    ...


def test_template_smoke_run():
    """缩小运行预算后，实验链路仍可正常完成."""
    ...


@pytest.mark.slow
def test_scientific_behavior():
    """可选：验证该 Template 声明的核心科学现象."""
    ...
```

推荐目录：

```text
data/graph_library/templates/
    grokking_p127.json

comfy_research/tests/repro/
    test_grokking_p127.py
```

---

## 6. 最低提交要求

每个新的实验 Template 至少应包含：

* **Template Baseline Test**

  * 核心 Node 类型；
  * 核心连接；
  * 关键参数。
* **Smoke Reproduction Test**

  * Template 可加载；
  * 实验可构建；
  * 小规模 CPU 运行可完成；
  * 输出不包含 NaN 或 Inf。

Scientific Behavior Test 根据实验性质决定是否加入。

最终目标是确保：
> 每个 Template 都有一套明确、可审查、不会被无意改变的实验基准设置。
