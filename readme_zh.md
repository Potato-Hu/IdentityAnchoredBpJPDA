# SwarmJam

一个面向多无人机协同压制关键目标的研究型仿真与多智能体强化学习项目。

当前仓库已经从最初的概念设计演化为一个**可运行的 MAPPO 基线系统**：场景中同时存在关键敌方节点和非关键节点，默认训练模式下策略主要依赖局部观测，外部全局观测只作为关键目标关联先验使用。

---

# 1. 项目概述

SwarmJam 关注的问题是：

> 在局部观测不完备、全局先验存在时延与噪声、且场景中存在 clutter 的条件下，多架友方 UAV 如何协同机动，以提高对关键敌方节点的压制效能？

当前实现包含以下闭环模块：

- 敌方与友方场景生成、运动仿真
- 仅针对关键目标的延迟带噪全局感知
- 面向每架 UAV 的局部匿名候选观测
- 全局关键先验与局部候选之间的关联模块
- 基于 MAPPO 的多智能体策略训练
- 干扰功率计算、奖励反馈与评估导出

需要特别注意的是，当前仓库的主线实现已经不同于早期构想：

- 默认训练模式下，**actor 不直接使用全局观测**
- 全局感知主要用于**关联局部候选与关键目标**
- 默认训练输入是**局部观测 + 候选 mask + 关联权重**

---

# 2. 当前系统流程

默认 `local_only` 模式下，整体流程可以概括为：

```text
场景真值
  -> 关键目标全局先验（延迟 + 噪声）
  -> 各 UAV 局部候选观测（探测半径内、匿名、含 clutter）
  -> 关联模块（none / nn / bp_jpda）
  -> 每个候选槽位上的关联权重
  -> MAPPO 策略
  -> 友方 UAV 离散机动
  -> 关键目标受扰功率与友方受扰功率
  -> 奖励
```

其中：

- 场景中的非关键节点仍然存在，并进入局部观测
- 但当前奖励主要围绕关键目标压制效能与友方安全展开

---

# 3. 当前实现的关键设计

## 3.1 观测设计

当前环境支持两种 `policy_input.mode`：

- `local_only`
- `groundtruth`

默认推荐和当前主线训练使用的是 `local_only`。

在 `local_only` 下，每个智能体的观测由以下部分拼接而成：

- 自身归一化位置
- 固定数量的局部相对候选位置
- 候选有效位 `mask`
- 与候选槽位对齐的关联权重向量

这意味着当前策略看到的是：

- 附近若干“匿名目标候选”
- 这些候选哪些更像关键目标的软提示

而不是：

- 直接的关键目标全局坐标
- 带身份标签的干净目标轨迹

## 3.2 全局观测的角色

当前 `src/sensing/global_sensor.py` 提供的是：

- 仅针对关键敌方节点的全局位置先验
- 含时间延迟
- 含高斯噪声

这些全局信息在默认模式下**不直接用于训练输入**，而是用于关联模块，为局部候选提供“哪个更可能是关键目标”的软引导。

## 3.3 关联模块

当前支持三种关联方式：

- `none`
- `nn`
- `bp_jpda`

其中：

- `none`：不使用全局先验，只取最近的局部候选
- `nn`：把全局关键先验与局部候选做最近邻匹配
- `bp_jpda`：使用近似 BP-JPDA 输出局部候选的软关联权重

当前策略真正接收的不是“融合后的关键目标状态”，而是**局部候选槽位上的权重向量**。

## 3.4 动作空间

友方 UAV 的动作空间当前是**离散 3D 机动动作**，不是连续控制。

具体来说：

- 1 个 `stop`
- 26 个三维方向动作

动作会按友方 `vmax` 转换为固定速度方向更新。

## 3.5 奖励设计

当前 `config/env.yaml` 中支持两种奖励模式：

- `difference`
- `sustained_tracking`

目前配置文件默认启用的是 `difference`。

这两种奖励都重点考虑：

- 对关键目标的压制贡献
- 友方自身受到的干扰安全约束

当前实现**没有**把非关键节点误压制作为显式奖励项直接写入主奖励公式；非关键节点目前主要通过 clutter 和关联难度影响训练。

---

# 4. 项目结构

当前仓库中的主要目录如下：

```text
SwarmJam/
├── config/                 # 环境与训练配置
├── docs/                   # 当前设计文档
├── outputs/                # 训练日志、检查点、导出结果
├── src/
│   ├── entities/           # 友方/敌方实体定义
│   ├── env/                # SwarmEnv、world、spaces
│   ├── interference/       # 干扰与信道计算
│   ├── rl/                 # MAPPO、buffer、reward、policy
│   ├── runner/             # train / evaluate / export
│   ├── sensing/            # global/local sensing、association、observation builder
│   ├── simulation/         # 场景初始化、ground truth、轨迹生成
│   ├── utils/              # 配置、日志、checkpoint、seed
│   └── visualization/      # 绘图与可视化
├── main.py                 # 生成初始图与轨迹图的简单入口
├── README.md
├── readme_zh.md
└── requirements.txt
```

几个当前最关键的代码入口：

- `src/env/swarm_env.py`：环境主逻辑、观测构造、奖励计算
- `src/sensing/global_sensor.py`：关键目标全局先验
- `src/sensing/local_sensor.py`：局部匿名候选观测
- `src/sensing/association.py`：关联算法实现
- `src/rl/mappo.py`：MAPPO 主实现
- `src/runner/train.py`：训练入口
- `src/runner/evaluate.py`：评估与 rollout 逻辑

---

# 5. 配置说明

当前最常用的配置文件有：

- `config/env.yaml`
- `config/mappo.yaml`

其中：

`config/env.yaml` 负责环境与观测相关配置，例如：

- 地图范围、仿真步长、episode 步数
- 敌我速度上限
- `policy_input.mode`
- `policy_input.association_method`
- 全局感知时延与噪声
- 局部感知半径、误检率、clutter 强度
- 奖励模式与超参数

`config/mappo.yaml` 负责训练相关配置，例如：

- key / non-key / friendly 数量
- rollout 长度和 batch size
- PPO 训练参数
- 网络宽度与层数
- 日志、评估、checkpoint 间隔

---

# 6. 训练与评估

## 6.1 安装依赖

```bash
pip install -r requirements.txt
```

## 6.2 训练

当前训练入口是：

```bash
python -m src.runner.train
```

如果需要显式指定设备或配置文件，可使用：

```bash
python -m src.runner.train --device cpu
python -m src.runner.train --config path/to/your_config.yaml
```

训练过程会在 `outputs/` 下写出：

- checkpoint
- history 日志
- 训练曲线图

## 6.3 评估与轨迹导出

当前仓库中：

- `src/runner/evaluate.py` 负责评估 rollout 逻辑
- `src/runner/export_trajectories.py` 用于导出策略轨迹和可视化结果

如果只是快速检查场景生成与基础可视化，也可以运行：

```bash
python main.py
```

它会在 `outputs/figures/` 下生成初始场景图和轨迹图。

---

# 7. 当前版本最值得注意的事实

为了避免继续把旧设计当成现状，阅读和开发时建议直接按下面几点理解项目：

1. 默认训练模式是 `local_only`，不是全局真值训练。
2. 全局观测当前主要服务于关联，不直接作为 actor 输入。
3. critic 虽然是 centralized，但在默认模式下仍然是拼接各智能体局部观测，而不是全知全能真值状态。
4. 动作空间当前是离散机动，不含单独的功率控制动作。
5. 非关键节点当前更像 clutter 来源，而不是奖励里的主惩罚项。
6. `groundtruth` 模式仍然存在，但更偏向调试和对照，而不是主线问题设定。

---

# 8. 后续可扩展方向

从当前实现继续往前推进时，比较自然的扩展方向包括：

- 把关联结果从“候选权重”进一步扩展为更稳定的目标级表示
- 引入更真实的功率控制或联合机动-功率动作
- 在奖励中显式建模 non-key 抑制代价
- 增强 critic 输入设计
- 引入更复杂的通信、协同或信息共享机制

---

# 9. 相关文档

当前更准确的系统说明请优先参考：

- [docs/system_model.md](docs/system_model.md)
- [docs/problem_formulation.md](docs/problem_formulation.md)

如果 `readme_zh.md` 与代码出现分歧，应以代码和 `docs/` 中的现状描述为准。
