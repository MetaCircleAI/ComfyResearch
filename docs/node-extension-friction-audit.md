# Node 扩展性摩擦审计

> 数据来源:node 扩展性试金石系列(2026-07-11/12)——以 PR #18 的真实实验为需求,
> 在当前架构下严格照规范接入,只记录摩擦、不顺手重构。
> 已交付:PR #70(Exp A: saxe init + weight_product_sv)、PR #71(Exp B: EoS 前端增强)、
> PR #72(Exp C: symmetrized init + outputScale + 2 observables + 专属 viz)。
> **覆盖缺口**:PR-D(dataset family + model family 全链)与 PR-E(工具 nodes)经用户决定顺延——
> dataset/model 两条扩展链只有静态审计(见 §1 表),**没有实测摩擦数据**。

## 1. 实测成本:新增一个 node 要改多少文件

| 扩展类型 | 静态审计预估 | 实测 | 样本 |
|---|---|---|---|
| observable(新 viz variant) | 10–12 | **18**(前端 10 + 生成物 4 + 后端 3 + 测试 1) | weight_product_sv(PR-A) |
| observable(复用 user 通路) | 10–12 | **9** | weight_displacement(PR-C) |
| observable(embedding 通路 + 专属 viz node) | — | **14** | neuron_trajectory_2d(PR-C) |
| init/transform 第 2 个 kind | ~11 | **14**(含 bool→node 状态迁移 3 文件) | saxe(PR-A) |
| init/transform 第 3 个 kind | — | **8**(后端仅 集合+1、elif+2 行) | symmetrized(PR-C) |
| 已有 node 加一个参数 | — | **7 文件 / 5 份手工表**(见 §2) | outputScale(PR-C) |
| 已有 observable 的 viz 增强 | — | 3 | 2/η 参考线(PR-B) |
| dataset family(全链) | 15–18(静态) | 未实测(PR-D 顺延) | — |
| model family(全链) | 12–14(静态) | 未实测(PR-D 顺延) | — |

本质工作量占比极低:saxe 核心 18 行、weight_product_sv 30 行、symmetrized 25 行、
outputScale 20 行——**其余全是接线税**。

## 2. 核心发现(按证据强度排序)

### F1. 同一事实要手写 N 处,且漏写大多静默降级

- **node id**:一个 observable 的 type 字符串要写 **8 处**(metadata / spec / registry /
  canvas / recorder handlers / `_allowed_obs` / guard 期望集 / viz map),外加 viz variant
  字符串 **5 处**(union / allowlist / spawn / 渲染 switch / 后端 map)。
- **参数**:给已有 node 加一个可 sweep 参数 = **5 份手工表**:defaults →
  `DEFAULT_MLP_PARAM_ORDER` → specCode `KNOWN_KEYS`/`pyTypeForKey` → 生成 param model →
  `trainSeriesPlan` sweep 轴。前 4 份在 review 中补齐后,第 5 份仍然漏掉,
  由用户实测发现:**多值输入静默只跑一组,无任何报错**。
- **失败模式是静默的**:漏 `_allowed_obs` 是 400(能发现);漏 sweep 轴、漏 viz map、
  漏 spawn 是静默功能缺失(发现不了)。

### F2. 静默降级已是存量事实:sweep 缺口清单(待修,用户决定后置)

- **A 类**(UI 多值、轴表漏登记,与 outputScale 同款):`bigram_low_rank_dataset`
  (corruptRatio/corruptScale)、`synthetic_playground_dataset`(depoWindow/manoModulus/
  lanoNestingDepth)、`biography_lm_dataset`(slotNoiseProb)、`multi_hop_fact_chain_dataset`
  (chainHops)、`in_context_associative_recall_dataset`(crossSampleRepeatProb/
  repeatedTokenCount)。
- **B 类**(整个类型不在 sweep 派发表):`modular_addition_dataset`(grokking 常用,
  格式化代码都以为它能 sweep——近乎确定的历史遗漏)、token 系模型×3、resnet/vit/hyena、
  vision 数据集×4、dataset_mixer×2、alphafold 别名(派发按 `type==="mlp_model"`
  精确匹配,组件复用的别名类型永远匹配不上)。
- **C 类**(轴表有、UI 只给单值框):lr_schedule / mup_lr_schedule 全部参数。
- 同类前科:PR #18 的 **3 个半接线功能**(effective_lr、batch_loss_stats 前端全注册
  后端零实现;Lanczos computeMethod 下拉选了也是跑 exact)。

### F3. 前端画布层没有事实源(计划外触点的唯一来源)

registry→manifest 外层三个 PR 零意外;**执行中冒出的意外全部在画布层**:
`ResearchCanvas` 连接规则硬编码(`sourceNode?.type === "mup_initialization"` ×2 处)、
add-node 逐 type 分支、共享 viz 组件的文案三元链 ×5、viz 样式集中在全局 index.css。
连接规则、节点创建、viz 渲染、样式四件事没有任何声明式来源。

### F4. guard 网络有效,且揭示了正确的批次边界

- 失败先行的期望集 guard 强制接线完整;Reads/Writes docstring guard 强制申报状态变更;
  同步 guard(PR-A 新增)首跑发现存量 22 个 observable 三面一致——**纪律靠人撑着,
  guard 把它变成机器的事**。
- 同步 guard 在 PR-C 拦住了"前端注册先行、后端 map 后补"的中间提交:
  **manifest viz metadata 与后端 map 必须同批落地**——这是 node 定义应当原子化的直接证据。
- **摊销效应可测量**:init 第 2 kind 14 文件(含状态迁移),第 3 kind 8 文件。
  "第一个实现者没给第二个 kind 留位置"是扩展贵的主因之一。

### F5. template/artifact 混合实锤

PR #18 的 template JSON 里嵌着 `memoryCheckpoint_b64`/`valueHistories`/
`lastTrainLoopSeconds` 等运行产物(单 template 最多 210k 行)。试金石系列以 strip 纪律
规避;**根治是 RunArtifact 分离**(plan_codex Phase 4)。新增证据:sweep 多组结果只能
存活最后一组(每组 run 原地覆写 viz 数据),用户明确需要 per-run 保留/对比——
`sweep_data_table` 只覆盖"每组一个标量",`metric_compare`(顺延)只覆盖两组对比。

### F6. 架构射程之外但制度内可解的两类问题

- **移植风险**:PR #18 的 2/η 输入框在 WebKit 不可编辑(label 复用 `.cr-tviz-check`
  继承 `user-select:none`)——单浏览器验证的代码 + 样式类继承是审查盲区;文本输入需要
  浏览器级验收。
- **实验设计正确性**:α-sweep 固定 lr 扫 α 混入训练时间尺度混淆("没训完"≠lazy);
  1/α² 理论补偿在离散 GD 下发散,需 warmup。node 系统不背这个锅,但 **phenomena
  scripts 制度**(--quick/--json + 探针标定阈值)证明了机器可查的现象证据是可维护的。

## 3. 对 plan_codex(旧重构草案)的证据对照

| plan_codex 主张 | 证据裁决 |
|---|---|
| 统一 NodeSpec 单一事实源 | **部分已实现且有效**:TS registry→manifest 外层零意外。缺口在 trainer 内层 + 画布层(F1/F3),不必推倒重来 |
| Runtime Provider 拆分(Model/Dataset/Observable/Init) | **方向被数据支持**:observable/init 是实测最痛两类;但 T3 已有 branch 级 provider,家族级拆分需 PR-D 实测数据(尚缺) |
| Phase 2 connection policy 从 spec 派生 | **强支持**(F3:画布层是计划外触点唯一来源) |
| Phase 4 template/artifact 分离 | **强支持 + 新证据**(F5:sweep 多组结果覆写) |
| Phase 5 外部 node 包 | 无新证据,与摩擦无直接关联,维持低优先级 |
| GenericSpecNode(普通节点免手写 TSX) | 中等支持:4 个新 node 组件全是"一两个字段 + 提示文案"的模板化 TSX,可由 fields 派生;但强自定义组件(trajectory viz 260 行)必须保留出口 |

## 4. 重构方向候选(按证据强度)

1. **Observable 定义原子化**:一个 observable 的 id/viz-variant/记录逻辑/viz 行为
   声明在一处,派生出 recorder 注册、allowlist、viz maps、spawn、渲染 switch
   (证据:F1 的 8+5 处、F4 的同批性)。
2. **参数定义原子化**:字段一次声明,派生 defaults/param-order/specCode/param-model/
   sweep 轴,**没有"第 5 份表"**;或至少加一致性 guard 让漏写变成红灯(证据:F1/F2)。
3. **画布策略声明化**:连接规则(socket 类型/init 源集合)、add-node 默认值从
   spec 派生(证据:F3;plan_codex Phase 2)。
4. **RunArtifact 分离**(证据:F5;plan_codex Phase 4)。
5. **init/transform 的 provider 化**:第 4 个 kind 应只是"一个函数 + 一行注册"
   (证据:F4 摊销曲线的下一步)。
6. (数据缺口)dataset/model family 链的方向判断**建议先补 PR-D 实测**再定,
   或显式接受静态审计的置信度。

**非目标(本轮不做,除非新证据)**:外部 node 包;PrepareState 全面拆解(T3d 已裁决:
80 字段触面大、无能力增益);recorder 文件拆分(已 registry 化,纯文件组织)。

## 5. 待办登记(审计范围,用户已知悉)

- sweep 缺口 A 类 5 处 + B 类 `modular_addition_dataset`(证据最硬)——机械修复,后置。
- B 类其余 + C 类——"哪些类型应该可 sweep"是设计决定,随重构一并考虑。
- hessian observable step-0 NaN(存量行为,已测试文档化)。
- PR #18 三个半接线功能:effective_lr / batch_loss_stats / Lanczos(PR-E 范围,顺延)。
- PR-D/E 顺延:teacher_classification_dataset、mlp_classification、normalized_teacher_init、
  α-sweep template、metric_compare、dataset_visualization。
