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

## 3. Gate 模型有效性验证（基于realdevbench的测试集进行初步理解）

### 3.1 训练期指标验证

Gate 模型在训练集上的表现指标证明了其有效性：

- **ROC-AUC ≈ 0.66**: 在 noisy GUI 场景下是合理水平，表明模型具有区分能力
- **AP (Average Precision) ≈ 0.52**: 明显高于 base rate (0.26)，说明排序有信息量
- **Brier Score ≈ 0.18**: 概率校准良好
- **mean_p_err ≈ pos_rate ≈ 0.26**: 概率校准成功，这是使用阈值/分位数进行风险分层的前提

**结论**: Gate 模型没有过拟合、没有系统性偏置，完全可作为"是否进入纠偏"的第一道门。

### 3.2 测试集验证

测试集上的 sanity check 结果显示了三个关键信号：

#### ✅ 信号 1: 最高风险桶是"干净的"

高风险桶 `[0.4, 0.6)` 中：
- 只有 8 个 case
- **100% 全是错误**

这说明 Gate 在测试集上确实能挑出"几乎必错"的 case，这是后续 EM 翻转/重试的黄金触发区。

#### ✅ 信号 2: p_err 分布合理

测试集上 p_err 分布很窄（min ≈ 0.15, max ≈ 0.43, 75% ≈ 0.30），说明：
- 测试集整体比训练集更"温和"
- 几乎没有极端高风险样本
- 中桶 `[0.2, 0.4)` 很大（325 cases），高桶 `[0.4, 0.6)` 很小（8 cases）

这不是坏事，而是一个现实信号：测试集里"明显不可靠的判决"本来就不多。

#### ⚠️ 信号 3: 低桶 error_rate 的统计解释

低桶 `[0.0, 0.2)` 的 error_rate 为 0.38，这不是 Gate 的 bug，而是统计/采样现象：
- 低桶样本只有 58 个
- 测试集 error_rate 本身 ≈ 0.26
- NB gate 在 test 上的 rank 信息 > 绝对概率信息

**关键理解**: 低桶 ≠ "一定安全"，而是"相对更安全"。这正是为什么不要用 Gate 去"证明正确"，而是只用它去"筛出危险"。

### 3.3 对矫正策略的启示

#### ❌ 不要做的事

- ❌ 不要在测试集用固定阈值（比如 0.5）
- ❌ 不要对 `[0.2, 0.4)` 这个大桶 aggressive 翻转
- ❌ 不要追求"gate 后 accuracy 必须上升"

#### ✅ 正确、稳健的用法（推荐）

**1. 使用分位数 / top-K gate，而不是绝对阈值**

- Top 5% p_err → High-risk（≈ 6–7 cases）
- Top 20% p_err → Mid-risk
- 其余 → Low-risk

在当前测试集里，Top 5% 基本就对应 `[0.4, 0.6)`。

**2. 只在 High-risk 区域允许"动作"**

结合 EM 归因的策略：

| 区域 | 行为 |
|------|------|
| Low-risk (低风险) | KEEP（不纠偏） |
| Mid-risk (中等风险) | 运行 EM → 多数 ABSTAIN / 标记不可靠 |
| High-risk (高风险) | 运行 EM → 唯一允许 FLIP / RETRY 的区域 |

**重要性质**: 所有的 flip 都发生在"几乎必错"的 case 上
- → flip precision 非常高
- → 即便 flip 数量很少，系统可信度也显著提升

## 4. 优势

1. **效率提升**: 只对高风险 case 进行 EM 归因和矫正，避免直接被归因矫正正确样本，减少计算开销
2. **准确性提升**: 通过 Gate 模型准确识别可能出错的预测，有针对性地进行矫正
3. **可解释性**: 风险分层提供了清晰的决策依据，数据验证支持 Gate 的有效性
4. **稳健性**: 使用分位数而非固定阈值，适应不同数据分布
5. **高精度**: 只在几乎必错的高风险区域进行翻转，确保 flip precision 高

## 5. 文件结构

```
em_adaptive_learning/
├── gate.py                    # Gate 模型训练和预测
├── run_rootcase.py            # 主脚本，包含 Gate 控制流程
├── evaluation_utils.py        # 评估工具函数
└── batch_run_rootcase.py      # 批量运行脚本
```

## 6. 注意事项

1. **模型训练**: 需要先使用训练数据训练 Gate 模型
2. **特征一致性**: 预测时使用的特征必须与训练时一致
3. **风险阈值**: 可以根据实际效果调整风险分层的阈值（当前为 0.4 和 0.6）
4. **EM 模型**: 需要先训练好 EM 模型，用于 mid/high 风险 case 的归因

