# Gate 模型使用说明

## 概述

Gate 模型用于判断 Agent 预测结果的可靠性，通过评估 Agent 预测错误的可能性来指导后续的归因和矫正流程。

## 1. gate.py 模块

### 功能说明

`gate.py` 模块提供了完整的 Gate 模型训练和预测功能，用于进行 Agent 预测结果的可靠性判断。

### 主要组件

#### 1.1 特征构建 (`build_case_features`)

将 step-level 的证据数据聚合为 case-level 特征，包括：
- 证据统计特征：GUI、代码、无响应证据的均值、总和、最大值、最小值
- 轨迹长度特征
- 最后一步的证据特征
- Terminal 步骤的统计特征
- Agent 预测结果

#### 1.2 特征二值化 (`binarize_features`)

将连续特征转换为二值特征，适配 BernoulliNB 模型：
- 均值/最后一步特征：使用阈值（默认 0.5）二值化
- 计数特征（sum/traj_len）：转换为是否大于 0 的指示器
- 已有 0/1 特征保持不变

#### 1.3 模型训练 (`fit_gate_nb_from_step_df`)

使用朴素贝叶斯（BernoulliNB）模型训练 Gate：
- 输入：step-level 数据框
- 输出：训练好的 Gate 模型和训练报告
- 使用校准分类器（CalibratedClassifierCV）进行概率校准
- 支持交叉验证和概率校准

#### 1.4 模型预测 (`infer_gate_nb_from_step_df` / `gate_predict_proba_nb`)

对新数据进行预测：
- 输入：step-level 数据框
- 输出：每个 case 的 `p_err`（预测错误概率）
- 自动处理特征顺序和缺失特征

#### 1.5 模型保存与加载

- `save_gate_model`: 保存训练好的模型到 `.pkl` 文件
- `load_gate_model`: 从文件加载模型，自动处理模块路径问题

### 使用示例

```python
from gate import fit_gate_nb_from_step_df, load_gate_model, infer_gate_nb_from_step_df

# 训练 Gate 模型
gate, report_train = fit_gate_nb_from_step_df(
    df_train,
    col_case="test_case_id",
    evidence_cols=["E1_gui", "E2_code", "E4_noresp"],
    col_agent_pred="agent_testcase_score_x",
    col_gt="phi"
)

# 保存模型
from gate import save_gate_model
save_gate_model(gate, "gate_model.pkl")

# 加载模型并预测
gate = load_gate_model("gate_model.pkl")
report_test = infer_gate_nb_from_step_df(
    gate=gate,
    df_step=df_test,
    col_case="test_case_id",
    evidence_cols=["E1_gui", "E2_code", "E4_noresp"],
    col_agent_pred="agent_testcase_score_x"
)
```

## 2. run_rootcase.py 中的 Gate 控制流程

### 2.1 工作流程

在 `run_rootcase.py` 中，Gate 模型用于控制是否对 case 进行 EM 归因和矫正：

```
┌─────────────────┐
│  加载 Gate 模型  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 预测每个 case   │
│ 的 p_err        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 根据 p_err 分层  │
│ low/mid/high    │
└────────┬────────┘
         │
         ▼
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│ low   │ │ mid/high │
│ 风险  │ │ 风险     │
└───┬───┘ └────┬─────┘
    │          │
    │          ▼
    │    ┌──────────────┐
    │    │ 使用 EM 进行 │
    │    │ 归因和矫正   │
    │    └──────────────┘
    │
    ▼
┌──────────┐
│ 保持原判 │
│ (KEEP)   │
└──────────┘
```

### 2.2 风险分层策略

根据 `p_err` 将 case 分为三个风险等级：

- **low (p_err < 0.4)**: 低风险，Agent 预测可能可靠
  - 直接保持 Agent 原判，不进行 EM 归因和矫正
  - 决策：`KEEP`

- **mid (0.4 ≤ p_err < 0.6)**: 中等风险
  - 使用 EM 模型预测 AgentFail 概率
  - 如果 `p_agent > 0.7`：翻转预测 (`FLIP`)
  - 否则：保持原判 (`KEEP`)

- **high (p_err ≥ 0.6)**: 高风险，Agent 预测可能存在问题
  - 使用 EM 模型预测 AgentFail 概率
  - 如果 `p_agent > 0.7`：翻转预测 (`FLIP`)
  - 否则：如果有方向性支持，翻转预测 (`FLIP`)，否则保持原判 (`KEEP`)

### 2.3 核心函数

#### `build_case_df_with_risk(df, gate_model_path)`

构建 case-level 数据框并计算风险等级：
- 加载 Gate 模型
- 构建 case-level 特征
- 预测 `p_err`
- 根据 `p_err` 分层（low/mid/high）
- 返回包含 `test_case_id`, `p_err`, `gt`, `agent_pred`, `risk` 的数据框

#### `apply_risk_based_correction(case_df_test, em, df)`

根据风险等级进行矫正：
- 对 low 风险：直接保持原判
- 对 mid/high 风险：使用 EM 模型进行归因和矫正
- 返回包含矫正结果的 DataFrame

### 2.4 使用示例

```python
# 在 run_prediction 函数中
if gate_model_path.exists():
    # 构建 case-level 数据框并计算风险等级
    case_df_test = build_case_df_with_risk(df, gate_model_path)
    
    # 根据风险等级进行矫正
    pred_correct = apply_risk_based_correction(case_df_test, em, df)
    
    # 计算并打印准确率
    calculate_and_print_accuracy(pred_correct, df)
```

## 3. 优势

1. **效率提升**: 只对高风险 case 进行 EM 归因和矫正，减少计算开销
2. **准确性提升**: 通过 Gate 模型识别可能出错的预测，有针对性地进行矫正
3. **可解释性**: 风险分层提供了清晰的决策依据
4. **灵活性**: 可以根据实际需求调整风险阈值和处理策略

## 4. 文件结构

```
em_adaptive_learning/
├── gate.py                    # Gate 模型训练和预测
├── run_rootcase.py            # 主脚本，包含 Gate 控制流程
├── evaluation_utils.py        # 评估工具函数
└── batch_run_rootcase.py      # 批量运行脚本
```

## 5. 注意事项

1. **模型训练**: 需要先使用训练数据训练 Gate 模型
2. **特征一致性**: 预测时使用的特征必须与训练时一致
3. **风险阈值**: 可以根据实际效果调整风险分层的阈值（当前为 0.4 和 0.6）
4. **EM 模型**: 需要先训练好 EM 模型，用于 mid/high 风险 case 的归因

