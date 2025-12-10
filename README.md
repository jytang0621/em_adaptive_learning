# EM Adaptive Learning

基于期望最大化（EM）算法的自适应学习框架，用于根因分析和错误诊断。该框架支持两个基准测试：RealDevBench 和 WebDevJudge。

## 目录结构

```
em_adaptive_learning/
├── src/
│   ├── config.py                    # 路径配置文件
│   ├── evaluation.py                # 评估工具函数
│   ├── em_data_process.py           # 数据处理模块
│   ├── em_data_merge.py             # 数据合并模块
│   ├── llm_provider.py              # LLM 提供者
│   └── em_adaptive_learning/
│       ├── em_evidencedh_refine.py  # EM 算法核心实现
│       └── run_rootcause.py         # EM 算法运行入口
├── realdevbench/                    # RealDevBench 数据目录
├── webdevjudge/                     # WebDevJudge 数据目录
├── logs/                            # 日志目录
└── .env                             # 环境变量配置（需自行创建）
```

## 环境配置

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn loguru tqdm openpyxl python-dotenv
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# API 配置
API_KEY=your_api_key_here
BASE_URL=https://newapi.deepwisdom.ai
MODEL=claude-sonnet-4-5
```

### 3. 路径配置

所有路径配置在 `src/config.py` 中，可根据实际情况修改：

```python
BASE_DIR = Path("/data/hongsirui/em_adaptive_learning")
```

## Benchmark 基础数据位置

### 数据目录结构说明

两个 Benchmark（RealDevBench 和 WebDevJudge）遵循统一的目录结构：

```
{benchmark_name}/
└── data/                          # 数据根目录
    ├── {benchmark_name}_gt.jsonl                    # GT 文件（测试用例和真实标签）
    ├── {benchmark_name}_code_evidence.jsonl        # 代码证据文件
    └── traj/                      # 轨迹目录（按实验时间组织）
        └── {timestamp}/           # 实验时间戳（格式：YYYYMMDD_HHMMSS）
            ├── {gui_evidence}.jsonl                 # GUI 交互证据
            └── {agent_evaluation}.xlsx              # Agent 轨迹评估结果
```

**重要说明**：
- **静态数据**（测试用例、GT、代码证据）统一存放在 `{benchmark_name}/data/` 目录下
- **动态数据**（GUI 证据和执行轨迹）存放在 `{benchmark_name}/data/traj/` 目录下
- `traj` 目录按实验时间戳（格式：`YYYYMMDD_HHMMSS`）创建子文件夹，便于管理不同实验的数据
- 所有路径配置在 `src/config.py` 中统一管理

### RealDevBench

所有 RealDevBench 相关的数据路径定义在 `src/config.py` 中：

- **基础目录**: `realdevbench/`
- **数据目录**: `realdevbench/data/`
- **GT 文件**: `realdevbench/data/realdevbench_gt.jsonl` - 真实标签（Ground Truth）
- **代码证据**: `realdevbench/data/realdevbench_code_evidence.jsonl` - 代码审查证据
- **轨迹目录**: `realdevbench/data/traj/`
- **GUI 证据**: `realdevbench/data/traj/20251120_155804/xxx_gui_evidence.jsonl` - GUI 交互证据
- **Agent 评估文件**: `realdevbench/data/traj/20251120_155804/mgx跑测_MGX低完成度_三合一_拆分case_带label_20251117_210104.xlsx` - Agent 轨迹评估结果

### WebDevJudge

所有 WebDevJudge 相关的数据路径定义在 `src/config.py` 中：

- **基础目录**: `webdevjudge/`
- **数据目录**: `webdevjudge/data/`
- **GT 文件**: `webdevjudge/data/webdevjudge_unit.jsonl` - 真实标签（Ground Truth）
- **代码证据**: `webdevjudge/data/webdevjudge_code_evidence.jsonl` - 代码审查证据
- **轨迹目录**: `webdevjudge/data/traj/`
- **GUI 证据**: `webdevjudge/data/traj/20251111_143620/20251111_143620_gui_evidence.jsonl` - GUI 交互证据
- **Agent 评估文件**: `webdevjudge/data/traj/20251111_143620/mgx_webdevjudge_w_expected_singletest_20251111_143620.xlsx` - Agent 轨迹评估结果

## 数据处理流程

### 1. 数据加载与评估

使用 `src/evaluation.py` 中的函数加载和评估数据：

```python
from src.evaluation import load_gt, load_code_evidence, load_gui_evidence, run_evaluation_realdevbench, run_evaluation_webdevjudge
from src.config import REALDEVBENCH_GT_JSONL, REALDEVBENCH_PRED_FILE

# 加载 GT 数据
gt_df = load_gt(REALDEVBENCH_GT_JSONL, tag="realdevbench")

# 加载代码证据
code_evidence = load_code_evidence(REALDEVBENCH_CODE_EVIDENCE_JSONL, tag="realdevbench")

# 加载 GUI 证据
gui_evidence = load_gui_evidence(REALDEVBENCH_GUI_EVIDENCE_JSONL)

# 运行评估
result = run_evaluation_realdevbench(REALDEVBENCH_GT_JSONL, REALDEVBENCH_PRED_FILE)
```

### 2. 数据合并 (`src/em_data_merge.py`)

合并代码证据、GUI 证据和 Agent 判断：

```python
from src.em_data_merge import *
from src.config import *

# 加载 Agent 判断
agent_judge = pd.read_excel(WEBDEVJUDGE_PRED_FILE)
agent_judge = agent_judge[["case_name", "os_agent_score"]]
agent_judge.columns = ["test_case_id", "agent_judge"]

# 加载代码证据
code_evidence = load_code_evidence(WEBDEVJUDGE_CODE_EVIDENCE_JSONL, tag="webdevjudge")
code_evidence = code_evidence.merge(agent_judge, on="test_case_id", how="left")

# 加载 GUI 证据
gui_evidence = load_gui_evidence(WEBDEVJUDGE_GUI_EVIDENCE_JSONL)

# 合并数据
merged_df = pd.merge(code_evidence, gui_evidence, on="test_case_id", how="left")
```

### 3. 数据处理 (`src/em_data_process.py`)

#### 3.1 过滤判断证据

使用 LLM 对 Agent 的判断证据进行过滤，识别无响应/异常情况：

```python
from src.em_data_process import filter_judge_evidence
import asyncio

# 异步处理判断证据
filter_df = asyncio.run(filter_judge_evidence(merged_df, max_concurrent=10))
```

该函数会：
- 从 `action_content` 中提取 `Tell()` 格式的证据
- 使用 LLM 判断是否存在"无响应"或"明显异常"
- 返回包含 `agent_noresp` 字段的数据框

#### 3.2 转换为训练数据

将合并后的数据转换为 EM 算法所需的训练格式：

```python
from src.em_data_process import convert_to_train_data

train_df = convert_to_train_data(merged_df, label_col="gt_x")
train_df.to_excel("train_em_df.xlsx", index=False)
```

**证据粒度级别说明**：

不同证据具有不同的粒度级别，在构建步骤级别（step-level/action-level）的训练数据时需要特别注意：

| 证据类型 | 原始粒度 | 数据构建方式 | 适用动作类型 | 说明 |
|---------|---------|------------|------------|------|
| **E2_code** (代码证据) | 测试用例级别 | 将用例级别的 `code_evidence` 值**复制**到该用例的所有步骤 | 所有动作 | 代码审查是对整个测试用例的判断，因此同一用例的所有步骤共享相同的代码证据值 |
| **E1_gui** (GUI 证据) | 步骤级别 | 直接使用步骤级别的 `gui_evidence` 值 | **仅 Click 动作** | GUI 点击准确性是针对每个点击动作的独立判断，非点击动作（如 Scroll、Type 等）的 `gui_evidence` 为 -1（不适用） |
| **E3_reflect** (反思证据) | 测试用例级别 | 将用例级别的反思结果标注在 **Tell 动作**上 | **仅 Tell 动作** | Agent 的反思判断是针对整个测试用例的，通常通过 `Tell()` 动作输出，因此只在该动作上标注 |
| **E4_noresp** (无响应证据) | 步骤级别 | 通过 LLM 分析 Tell 动作中的证据生成 | **仅 Tell 动作** | 从 Tell 动作的 evidence 中提取并判断是否存在无响应或异常情况 |

**数据构建流程**：

1. **代码证据（E2_code）**：
   ```python
   # code_evidence 是 test_case_id 级别的
   # 合并时通过 test_case_id 关联，每个步骤自动继承该用例的代码证据值
   merged_df = pd.merge(code_evidence, gui_evidence, on="test_case_id", how="left")
   # 结果：同一 test_case_id 的所有步骤都有相同的 code_evidence_x 值
   ```

2. **GUI 证据（E1_gui）**：
   ```python
   # gui_evidence 已经是 action_id 级别的
   # 每个 Click 动作都有独立的 gui_evidence 值
   # 非 Click 动作的 gui_evidence = -1（表示不适用）
   ```

3. **反思证据（E3_reflect）**：
   ```python
   # 通过 filter_judge_evidence 处理 Tell 动作
   # 只对包含 'Tell (' 的 action_content 进行处理
   filter_df = asyncio.run(filter_judge_evidence(merged_df))
   # 结果：只有 Tell 动作会有 agent_noresp 和 agent_judge 值
   ```

**掩码（Mask）机制**：

- `M_gui=1`：当 `gui_evidence=-1` 时（非 Click 动作），掩码 GUI 证据
- `M_code=0`：代码证据始终有效（因为已复制到步骤级别）
- `M_reflect=1`：反思证据始终有效（但只在 Tell 动作上有值）
- `M_noresp=1`：当没有 Tell 动作或无法提取证据时，掩码无响应证据

输出字段说明：
- `test_case_id`: 测试用例 ID
- `step`: 步骤 ID（action_id）
- `phi`: 真实标签（Ground Truth）
- `operation_desc`: 操作描述
- `action_content`: 动作内容
- `E1_gui`: GUI 点击是否符合预期（1=好，0=坏）
- `E2_code`: 代码审查是否符合预期（1=好，0=坏）
- `E3_reflect`: Agent 反思分数（仅在 is_reflection=1 时有值）
- `E4_noresp`: 无响应/异常标识（1=有问题，0=正常）
- `M_gui`, `M_code`, `M_reflect`, `M_noresp`: 掩码标识（1=mask，0=不mask）
- `weight`: 样本权重
- `is_reflection`: 是否为反思步骤（1=是，0=否）
- `agent_testcase_score`: Agent 测试用例分数

### 数据标注说明

生成标准训练数据格式后，建议进行**根因标注**，主要标注 `delta_label` 字段。这是 EM 算法实现半监督学习的关键。

**`delta_label` 字段说明**：
- **字段位置**: 在训练数据 DataFrame 中添加 `delta_label` 列
- **标注值**:
  - `0` = **EnvFail**（环境失败）：失败由环境因素导致，如 GUI 点击位置错误、代码实现问题、系统异常等
  - `1` = **AgentFail**（Agent 失败）：失败由 Agent 自身问题导致，如 Agent 判断错误、无响应、逻辑错误等
  - `NaN` = 未标注（算法会自动推断）

**标注建议**：
1. **标注粒度**: 按 `test_case_id`（用例级别）进行标注，同一用例的所有步骤共享相同的 `delta_label`
2. **标注策略**: 
   - 优先标注**典型失败案例**，特别是容易混淆的边界情况
   - 建议标注至少 **10-20%** 的数据，即使少量标注也能显著提升训练效果
   - 标注时重点关注：
     - GUI 证据（E1_gui）和代码证据（E2_code）都正常但最终失败的案例 → 可能是 AgentFail
     - GUI 或代码证据明显异常的案例 → 可能是 EnvFail
     - Agent 无响应（E4_noresp=1）的案例 → 通常是 AgentFail
3. **标注示例**：
```python
# 在训练数据中添加 delta_label 列
train_df["delta_label"] = np.nan

# 标注示例：某个用例是环境失败
train_df.loc[train_df["test_case_id"] == "case_001", "delta_label"] = 0  # EnvFail

# 标注示例：某个用例是 Agent 失败
train_df.loc[train_df["test_case_id"] == "case_002", "delta_label"] = 1  # AgentFail
```

**半监督学习机制**：
- 有 `delta_label` 标注的用例作为**硬锚点**（hard anchor），在 EM 算法的 E-Step 中固定其根因类型
- 未标注的用例由算法根据证据自动推断根因类型
- 少量高质量标注即可引导算法学习到正确的根因模式，显著提升整体预测准确率

## EM 流程说明

### 算法概述

EM（Expectation-Maximization）算法用于根因分析，将失败原因分为两类：
- **EnvFail (δ=0)**: 环境失败（如 GUI 点击位置错误、代码实现问题）
- **AgentFail (δ=1)**: Agent 失败（如 Agent 判断错误、无响应等）

### 证据类型

#### Step-level 证据（步骤级证据）
- **E1_gui**: GUI 点击准确性（从 `coordinate_analysis` 提取）
- **E2_code**: 代码审查结果（从 `code_review` 提取）
- **E4_noresp**: 无响应/异常标识（通过 LLM 判断生成）

#### Case-level 证据（用例级证据）
- **agent_testcase_score**: Agent 对测试用例的整体评分
- **delta_label**: 人工标注的根因标签（半监督学习）

### 运行 EM 算法

#### 1. 训练模式

```bash
python src/em_adaptive_learning/run_rootcause.py \
    --data_path train_em_df.xlsx \
    --val_ratio 0.2 \
    --seed 42 \
    --tau_agentfail 0.75 \
    --tau_envfail 0.75
```

参数说明：
- `--data_path`: 训练数据路径
- `--val_ratio`: 验证集比例
- `--seed`: 随机种子
- `--tau_agentfail`: AgentFail 的阈值
- `--tau_envfail`: EnvFail 的阈值

#### 2. 预测模式

```bash
python src/em_adaptive_learning/run_rootcause.py \
    --test_path test_em_df.xlsx \
    --params_path em_outputs_refine_webdevjudge/em_params.json \
    --out_dir em_outputs_refine_webdevjudge
```

输出文件：
- `pred_step.csv`: 步骤级预测结果（包含 P_EnvFail 和 P_AgentFail）
- `pred_corrected_cases.csv`: 修正后的用例级预测结果

### EM 算法参数

在 `em_evidencedh_refine.py` 中可以调整以下参数：

```python
em = SimpleEM4EvidenceH_Refine(
    max_iter=100,              # 最大迭代次数
    tol=1e-4,                  # 收敛阈值
    seed=0,                    # 随机种子
    bin_thresh=0.5,            # 二值化阈值
    w_gui=1.0,                 # GUI 证据权重
    w_code=1.0,                # 代码证据权重
    w_noresp=0.5,              # 无响应证据权重
    agent_weight=0.9,          # Agent 通道权重
    a_pi=5.0, b_pi=5.0,       # P_delta 先验参数
    a_c0=3.0, b_c0=3.0,       # psi EnvFail 先验参数
    a_c1=3.0, b_c1=3.0,        # psi AgentFail 先验参数
    theta_floor=0.05,          # theta 下限
    theta_ceil=0.95,           # theta 上限
    pi_floor=0.02,             # pi 下限
    temp=0.8,                  # 温度参数
)
```

### 算法流程

1. **初始化**: 随机初始化参数（theta, psi, p_delta）
2. **E-Step**: 计算每个 case 的后验概率 P(δ|证据)
3. **M-Step**: 根据后验概率更新参数
4. **迭代**: 重复 E-Step 和 M-Step 直到收敛
5. **预测**: 使用训练好的参数对新数据进行预测

### 输出结果

- **步骤级预测**: 每个步骤的 P_EnvFail 和 P_AgentFail 概率
- **用例级预测**: 基于步骤级预测聚合得到的用例级根因分析
- **参数文件**: `em_params.json` 包含训练好的所有参数

## 使用示例

### 完整流程示例

```python
# 1. 加载和评估数据
from src.evaluation import *
from src.config import *

gt_df = load_gt(WEBDEVJUDGE_GT_JSONL, tag="webdevjudge")
code_evidence = load_code_evidence(WEBDEVJUDGE_CODE_EVIDENCE_JSONL, tag="webdevjudge")
gui_evidence = load_gui_evidence(WEBDEVJUDGE_GUI_EVIDENCE_JSONL)

# 2. 合并数据
from src.em_data_merge import *
agent_judge = pd.read_excel(WEBDEVJUDGE_PRED_FILE)
agent_judge = agent_judge[["case_name", "os_agent_score"]]
agent_judge.columns = ["test_case_id", "agent_judge"]

code_evidence = code_evidence.merge(agent_judge, on="test_case_id", how="left")
merged_df = pd.merge(code_evidence, gui_evidence, on="test_case_id", how="left")

# 3. 处理数据
from src.em_data_process import filter_judge_evidence, convert_to_train_data
import asyncio

filter_df = asyncio.run(filter_judge_evidence(merged_df))
test_df = pd.merge(merged_df, filter_df, on='action_id', how='left')
train_df = convert_to_train_data(test_df)
train_df.to_excel("train_em_df.xlsx", index=False)

# 4. 运行 EM 算法
# 使用命令行或直接调用 run_rootcause.py
```

## 注意事项

1. **数据格式**: 确保输入数据符合预期格式，特别是 JSONL 文件的结构
2. **API 配置**: 使用 LLM 功能前需要正确配置 `.env` 文件
3. **路径配置**: 根据实际环境修改 `src/config.py` 中的路径
4. **内存管理**: 处理大规模数据时注意内存使用，可以分批处理
5. **并发控制**: `filter_judge_evidence` 中的 `max_concurrent` 参数控制并发数，根据 API 限制调整

## 日志

日志文件保存在 `logs/` 目录下，按日期命名（如 `logs/20251210.txt`）。

## 许可证

[根据实际情况填写]

## 联系方式

[根据实际情况填写]

