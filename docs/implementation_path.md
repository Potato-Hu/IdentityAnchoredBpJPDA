# 实现路径

## 1. 总体路线

本项目最现实、也最容易做出可信结果的技术路线是：

```text
关键目标弱先验
+ 局部匿名候选
+ 上一时刻 track state
+ 不确定性信息
        ->
时序编码器 / 小型 GNN
        ->
关联分数 + 漏检概率修正 + 存在概率修正
        ->
JPDA / LMB 等经典后端递推更新
        ->
局部关键航迹
        ->
集中式航迹融合与下游分配
```

换句话说，研究重点应定义为 **learned scoring for classical association**，而不是完全发明一种新的滤波器。

## 2. 与当前仓库的衔接

### 2.1 已有基础

当前仓库已经具备以下实现基础：

- 关键目标弱先验输入：`src/sensing/global_sensor.py`
- 局部匿名候选输入：`src/sensing/local_sensor.py`
- 经典关联后端：`src/sensing/association.py`
- 集中式航迹融合：`src/tracking/fusion_center.py`、`src/tracking/fusion.py`
- 基础评估脚本：`src/tracking/metrics.py`、`src/runner/export_tracking_rollout.py`

这意味着我们不需要从零搭跟踪系统，而是只需要把“学习化评分模块”插入到现有局部关联层之前或之中。

### 2.2 建议的代码落点

建议把新增能力集中在 `src/sensing/association.py` 所在层附近，形成如下职责划分：

- `global_sensor` / `local_sensor`：继续负责提供先验与候选观测
- `association`：保留 `JPDA`、`LMB`、`GM-PHD` 等经典递推后端
- 新增学习模块：负责输出候选边分数、漏检修正和存在修正
- `fusion_center`：继续使用局部关键航迹做集中式融合，不承担学习关联逻辑

如果后续需要把学习模块拆分得更清晰，可考虑新增如下文件：

- `src/sensing/learned_association.py`
- `src/sensing/association_features.py`
- `src/tracking/advanced_metrics.py`

## 3. 模型设计建议

### 3.1 输入设计

学习模块每一步可使用以下输入：

- `key prior`
- `local candidates`
- `previous track state`
- `previous existence probability`
- `track covariance / uncertainty`
- `candidate validity mask`
- 可选的局部几何上下文或邻近候选关系

其中最关键的是把“历史连续性”和“不确定性”一起送入模型，而不是只做单帧候选匹配。

### 3.2 编码器设计

建议优先从两类轻量结构开始：

- 时序编码器
  适合先做按 track 递推的状态表征，工程复杂度更低。
- 小型 GNN
  把 `track` 与 `candidate` 建成二部图，适合表达多候选竞争与 `key/non-key` 混淆。

第一阶段更建议先用轻量模型，确保能够稳定插入现有后端并跑通对比实验。

### 3.3 输出设计

模型输出不应直接是最终目标状态，而应是对经典递推最有帮助的中间量：

- 每个 `track-candidate` 的关联分数
- 每条关键航迹的漏检概率修正
- 每条关键航迹的存在概率修正
- 可选的歧义惩罚或 identity-preserving margin

这些量可直接作用于 `JPDA` 或 `LMB` 的边权、漏检分支和存在更新。

## 4. 分阶段实现建议

### 第一阶段：夯实现有基线

目标是把当前仓库作为可信的对照平台：

- 固化 `bp_jpda`、`gm_phd`、`lmb` 的实验配置
- 导出固定场景下的关键目标跟踪结果
- 明确当前在哪些场景下容易出现身份切换和跟踪中断

### 第二阶段：补齐关键指标

当前仓库已支持：

- `key-target RMSE`
- `key-target OSPA`

下一步应补充：

- `ID switch count`
- `track fragmentation count`

这一阶段很重要，因为没有这些指标，后续学习模块的优势很难被准确表达。两项新增指标都属于多目标跟踪领域的常用诊断量：

- `ID switch count`
  表示同一真实关键目标在相邻跟踪时刻被匹配到不同估计轨迹身份的次数，反映身份保持能力。
- `track fragmentation count`
  表示一条真实关键目标轨迹在已经被成功跟踪后，又经历中断并重新恢复的次数，反映持续稳定性。

### 第三阶段：接入学习化评分模块

建议先做“最小侵入式”集成：

- 保留原有 `JPDA` 或 `LMB` 递推框架
- 用学习模型替换或修正原始似然评分
- 保留基于高斯 / 伯努利 / 混合成分的后验更新

这样能最大程度减少调试风险，也最容易和经典基线做公平对比。

### 第四阶段：系统实验

实验设计建议围绕以下变量展开：

- 先验噪声强度
- 目标机动与模型失配程度
- 非关键目标贴近强度
- 漏检率与 clutter 强度

基线建议至少包含：

- 最近邻或无关联先验基线
- `BP-JPDA`
- `GM-PHD`
- `LMB`
- `delta-GLMB`
- `learned-JPDA`
- `learned-LMB`

## 5. 实验设计框架

建议把实验设计写成“任务导向的渐进验证”，而不是泛泛地报几个均值。

### 5.1 研究问题

核心实验问题应当是：

- 在弱关键先验条件下，学习化关联层是否能降低关键目标身份切换？
- 在非关键目标贴近和 clutter 增强时，学习化关联层是否能减少关键目标轨迹中断？
- 在模型失配存在时，学习化关联层是否还能保持对经典后端的稳定增益？

### 5.2 对比组

建议至少包含三层对比：

- 基础对比
  - 最近邻或无概率关联基线
  - `BP-JPDA`
  - `LMB`
  - `delta-GLMB`
- 目标方法对比
  - `learned-JPDA`
  - `learned-LMB`

### 5.3 场景变量

每组实验建议只改变一类因素，其余保持固定：

- 全局关键先验噪声：弱、中、强
- 局部漏检率：低、中、高
- clutter 强度：低、中、高
- 非关键目标贴近程度：疏、中、密
- 运动模型失配：轻度、中度

### 5.4 输出结果

每组实验至少报告：

- `key-target OSPA`
- `key-target RMSE`
- `ID switch count`
- `track fragmentation count`

如果篇幅允许，可附一组定性轨迹图，展示经典方法在关键目标贴近和轨迹交叉时出现的身份跳变，以及学习化方法的改进效果。

## 6. 推荐的论文式实现表述

实现路径可归纳为：

> 先构建关键目标优先的经典关联与融合基线，再在局部关联层引入学习化评分模块，对候选边似然、漏检分支和存在分支进行修正，最终通过关键目标连续性、身份一致性和漏跟踪时长等指标验证方法有效性。
