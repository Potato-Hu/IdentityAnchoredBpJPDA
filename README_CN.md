# SwarmJam

[English](README.md) | 简体中文

SwarmJam 是一个面向多无人机协同场景的**关键目标优先多目标跟踪与数据关联研究平台**。项目在三维动态场景中模拟友方无人机、关键目标与非关键干扰目标，结合弱全局先验、局部匿名观测、多种关联算法和集中式航迹融合，评估关键目标的跟踪精度、覆盖率与身份连续性。

> 当前版本以多目标跟踪、数据关联和协同跟踪评估为主线，不包含可直接运行的强化学习训练入口。

## 主要内容

- **三维场景仿真**：支持关键/非关键目标、友方无人机、Gauss-Markov 与随机游走运动模型。
- **多源感知建模**：生成带延迟、噪声和漏检的关键目标弱全局先验，以及带噪声、漏检和杂波的局部匿名观测。
- **多目标数据关联**：实现最近邻、Surveillance-only、BP-JPDA、GM-PHD、LMB、δ-GLMB，以及多种先验锚定 BP-JPDA 方法。
- **集中式航迹融合**：汇总各无人机的局部航迹，并通过一致性检查与协方差交叉融合得到全局关键目标航迹。
- **协同跟踪闭环**：支持距离型任务分配（`hungarian` 配置模式）和离散三维机动控制。
- **评估与可视化**：输出 RMSE、OSPA、GOSPA、跟踪覆盖率、ID Switch、航迹碎片数、运行时间、轨迹图和 CSV/NPZ 数据。

系统流程：

```text
三维场景与目标运动
        │
        ├── 弱全局关键目标先验
        └── 各无人机局部匿名观测
                    │
                    ▼
          局部数据关联与航迹维护
                    │
                    ▼
            集中式航迹关联与融合
                    │
                    ▼
          任务分配 → 无人机机动控制
                    │
                    ▼
              指标统计与结果导出
```

## 环境要求

- Python 3.10–3.12
- Windows、Linux 或 macOS
- 主要依赖：NumPy、Matplotlib、Seaborn、PyYAML

建议在项目根目录创建独立虚拟环境。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 快速开始

所有命令均在项目根目录执行。

### 1. 检查场景生成与基础可视化

```bash
python main.py
```

运行后会在 `outputs/figures/` 生成初始三维场景、敌方目标轨迹和友方无人机轨迹 PDF。

### 2. 运行一次完整跟踪实验

```bash
python -m src.runner.export_tracking_rollout --no-show
```

默认参数来自 `config/default.yaml` 和 `config/env.yaml`。常用覆盖参数如下：

```bash
python -m src.runner.export_tracking_rollout \
  --seed 10 \
  --key-num 3 \
  --nonkey-num 30 \
  --mydrone-num 5 \
  --no-show
```

Windows PowerShell 也可以写成单行：

```powershell
python -m src.runner.export_tracking_rollout --seed 10 --key-num 3 --nonkey-num 30 --mydrone-num 5 --no-show
```

结果保存在：

```text
outputs/tracking_rollouts/<场景名>/<关联方法>/
```

其中包括轨迹 `NPZ`、位置与动作 `CSV`、关键目标指标 `CSV`，以及场景和轨迹 `PDF`。

### 3. 运行 Monte Carlo 方法对比

```bash
python -m src.runner.run_tracking_monte_carlo --num-seeds 10 --start-seed 0
```

默认比较 `bp_jpda`、`gm_phd`、`lmb`、`delta_glmb` 和 `labeled_anchor_bp_jpda`。也可以只指定部分方法：

```bash
python -m src.runner.run_tracking_monte_carlo \
  --num-seeds 10 \
  --methods bp_jpda labeled_anchor_bp_jpda
```

`--num-seeds` 至少为 2。汇总 CSV 和带 95% 置信区间的对比图保存在：

```text
outputs/figure/<场景名>/method_comparison/
```

已有指标数据时，可只重新绘图：

```bash
python -m src.runner.run_tracking_monte_carlo --plot-only --csv-path <指标CSV路径>
```

### 4. 非关键目标密度实验

下面的命令比较不同非关键/关键目标数量比下的跟踪表现：

```bash
python -m src.runner.run_nonkey_density_sweep \
  --num-seeds 10 \
  --key-num 3 \
  --ratios 3 5 10 15 20
```

该实验固定使用 `labeled_anchor_bp_jpda`。总运行次数等于“种子数 × 比例数”，大规模参数会耗时较长。

## 配置说明

### 全局与环境配置

| 文件 | 作用 |
| --- | --- |
| `config/default.yaml` | 默认随机种子 |
| `config/env.yaml` | 仿真步长、回合长度、跟踪模式、关联方法、任务分配、递推模型和融合参数 |
| `config/scenarios/*.yaml` | 实体数量、初始化区域、地图范围、运动模型、感知噪声、漏检率和杂波参数 |

在 `config/env.yaml` 中切换场景：

```yaml
selected_scenario: scenerios_many_nonkey_partitioned
```

仓库内置场景包括：

- `scenerios_base`：基础场景；
- `scenerios_many_nonkey_partitioned`：大量非关键目标的分区运动场景；
- `scenario_high_noise`：更强全局先验噪声场景。

场景名 `scenerios_*` 沿用当前文件名中的拼写，修改配置时请保持一致。

### 跟踪与关联方法

单次跟踪实验使用 `config/env.yaml` 中的配置：

```yaml
tracking:
  mode: local_only
  association_method: labeled_anchor_bp_jpda
  assignment: hungarian
```

- `tracking.mode: local_only`：使用弱全局先验、局部匿名观测和关联算法，是默认研究模式。
- `tracking.mode: groundtruth`：直接暴露关键目标真值航迹，仅用于调试或对照。
- `tracking.assignment`：支持 `hungarian` 和 `placeholder`。

当前 `hungarian` 模式对小规模分配使用排列穷举，对较大问题回退到贪心求解，不依赖 SciPy。

可用的 `association_method`：

| 方法 | 简要说明 |
| --- | --- |
| `none` | 最近局部候选基线 |
| `prior_only` | 仅使用外部关键目标先验与运动预测 |
| `bp_jpda` | BP 优化的 JPDA 基线 |
| `hard_anchor_bp_jpda` | 以当前带标签先验作为硬锚点 |
| `covariance_anchor_bp_jpda` | 基于协方差融合的锚定 BP-JPDA |
| `labeled_anchor_bp_jpda` | 固定关键目标标签的先验锚定 BP-JPDA（IA-BP-JPDA） |
| `gm_phd` | GM-PHD 递推基线 |
| `lmb` | Labeled Multi-Bernoulli 风格基线 |
| `delta_glmb` | 紧凑 δ-GLMB 多假设基线 |

关联与递推相关参数集中在 `config/env.yaml` 的 `association` 段；局部航迹融合参数位于 `track_fusion` 段。递推运动先验支持单一 `cv` 或由 `cv`、`ca`、`ct_left`、`ct_right` 组成的 `imm` 模型组。

## 项目结构

```text
SwarmJam/
├── config/                 # 全局、环境和场景配置
├── docs/                   # 系统模型、问题定义与算法说明
├── outputs/                # 仿真、指标和可视化结果
├── src/
│   ├── entities/           # 友方无人机与敌方目标实体
│   ├── env/                # 环境封装与世界状态更新
│   ├── interference/       # 信道与干扰功率计算
│   ├── runner/             # 单次实验、Monte Carlo 与密度实验入口
│   ├── sensing/            # 全局/局部感知与数据关联算法
│   ├── simulation/         # 场景初始化、运动与轨迹生成
│   ├── tracking/           # 航迹融合、指标、任务分配与控制
│   ├── utils/              # 配置与随机种子工具
│   └── visualization/      # 绘图与动画
├── main.py                 # 场景生成与基础可视化入口
└── requirements.txt        # Python 依赖
```

## 进一步阅读

- [系统模型](docs/system_model.md)
- [问题定义](docs/problem_formulation.md)
- [项目摘要](docs/project_summary.md)
- [Labeled-anchor BP-JPDA 说明](docs/labeled_anchor_bp_jpda.md)

本项目目前是研究原型。复现实验时建议固定随机种子，并同时记录场景、关联方法和完整配置。
