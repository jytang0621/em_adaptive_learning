from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, make_scorer
from sklearn.feature_selection import SelectKBest, f_classif

# -----------------------------
def build_case_features(
    df: pd.DataFrame,
    col_case: str = "test_case_id",
    col_step: Optional[str] = None,  # 如果有 step index/时间顺序列可填；没有也可以
    col_terminal: Optional[str] = None,  # 如果有 terminal 标记列更好
    evidence_cols: List[str] = ("E_gui", "E_code", "E_noresp"),
    col_agent_pred: str = "agent_testcase_score_x",
    col_gt: str = "phi",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    返回:
      X_case: case-level feature dataframe
      y_err : A = 1[agent_pred != gt]
      y_gt  : gt (方便后续分析)
    """
    use_cols = [col_case, col_agent_pred, col_gt, *evidence_cols]
    if col_step and col_step in df.columns:
        use_cols.append(col_step)
    if col_terminal and col_terminal in df.columns:
        use_cols.append(col_terminal)

    d = df[use_cols].copy()

    # 若有 step 顺序列，先排序以便抽取“last step”特征
    if col_step and col_step in d.columns:
        d = d.sort_values([col_case, col_step])

    # 基础：case 的 agent_pred / gt（假设 case 内一致；不一致则取众数）
    def mode01(s: pd.Series) -> int:
        # 处理可能有 NaN
        s = s.dropna()
        if len(s) == 0:
            return 0
        v = int((s.mean() >= 0.5))
        return v

    g = d.groupby(col_case, sort=False)

    agent_pred_case = g[col_agent_pred].agg(mode01).astype(int)
    gt_case = g[col_gt].agg(mode01).astype(int)
    y_err = (agent_pred_case != gt_case).astype(int)

    feats: Dict[str, pd.Series] = {}

    # 证据的统计特征：mean/sum/max/min
    for c in evidence_cols:
        feats[f"{c}_mean"] = g[c].mean()
        feats[f"{c}_sum"] = g[c].sum()
        feats[f"{c}_max"] = g[c].max()
        feats[f"{c}_min"] = g[c].min()

        # “出现过”比 max 更稳（尤其当列可能不是严格 0/1）
        feats[f"{c}_any"] = (g[c].max() > 0).astype(int)

    # 轨迹长度特征
    feats["traj_len"] = g.size()

    # last step 的 evidence（若无顺序列，则用 groupby.tail(1) 的结果）
    if col_step and col_step in d.columns:
        last_rows = g.tail(1).set_index(col_case)
    else:
        last_rows = g.tail(1).set_index(col_case)

    for c in evidence_cols:
        feats[f"{c}_last"] = last_rows[c].astype(float)

    # terminal-only 的统计（如果你有 terminal 标记列）
    if col_terminal and col_terminal in d.columns:
        term = d[d[col_terminal].astype(bool)]
        if len(term) > 0:
            gt = term.groupby(col_case, sort=False)
            for c in evidence_cols:
                feats[f"{c}_term_mean"] = gt[c].mean()
                feats[f"{c}_term_sum"] = gt[c].sum()
        else:
            # 没有 terminal 行：补 0
            for c in evidence_cols:
                feats[f"{c}_term_mean"] = 0.0
                feats[f"{c}_term_sum"] = 0.0

    # 把 agent_pred 本身作为 gate 输入（很重要：同样的证据在 PASS/FAIL 判决下含义不同）
    feats["agent_pred"] = agent_pred_case.astype(int)
    feats["phi"] = gt_case.astype(int)
    X_case = pd.DataFrame(feats)
    X_case.index.name = col_case

    # 填缺失（比如某些 case 没 terminal 行）
    X_case = X_case.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_case, y_err, gt_case
# --------- 复用你已有的 build_case_features ----------
# 建议你让 build_case_features 输出“二值化后的特征”，因为 BernoulliNB 假设 feature 是 0/1
def binarize_features(X: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """
    将聚合得到的连续特征二值化：
      - *_any / agent_pred 已经是 0/1，保持
      - mean/last 等在 [0,1] 的特征用阈值化
      - sum/traj_len 这类计数特征建议转成“是否大于0/是否大于1/是否大于k”这种 indicator
    """
    Xb = X.copy()

    for c in Xb.columns:
        col = Xb[c]
        # 如果已经是 0/1
        if set(pd.unique(col.dropna())).issubset({0, 1}):
            Xb[c] = col.astype(int)
            continue

        # 对均值/最后一步这类一般在 [0,1]，用 thr 二值化
        if c.endswith(("_mean", "_last", "_min", "_max", "_term_mean")):
            Xb[c] = (col >= thr).astype(int)
            continue

        # 对 sum / term_sum / traj_len 等计数，转成 indicator：>0
        if c.endswith(("_sum", "_term_sum")) or c in ("traj_len",):
            Xb[c] = (col > 0).astype(int)
            continue

        # 兜底：阈值化
        Xb[c] = (col >= thr).astype(int)

    return Xb


@dataclass
class GateModelNB:
    model: CalibratedClassifierCV
    feature_names: List[str]


def train_gate_model_nb(
    X: pd.DataFrame,
    y_err: pd.Series,
    alpha: float = 1.0,                  # Laplace smoothing
    calibrate_method: str = "sigmoid",   # sigmoid 推荐；isotonic 数据大才用
    n_splits: int = 5,
    seed: int = 42,
    optimize_alpha: bool = True,         # 是否优化 alpha 参数
    feature_selection: bool = False,     # 是否进行特征选择
    n_features: Optional[int] = None,   # 特征选择时保留的特征数（None 表示保留所有）
) -> GateModelNB:
    """
    朴素贝叶斯 gate：
      - BernoulliNB 适配 0/1 evidence
      - CalibratedClassifierCV 做概率校准
      - 可选的超参数优化和特征选择
    """
    X_work = X.copy()
    
    # 特征选择（可选）
    selected_features = list(X.columns)
    if feature_selection and len(X.columns) > 1:
        if n_features is None:
            n_features = min(len(X.columns), max(10, len(X.columns) // 2))
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(X.columns)))
        X_selected = selector.fit_transform(X_work, y_err)
        selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
        X_work = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
        print(f"[Feature Selection] Selected {len(selected_features)}/{len(X.columns)} features")
    
    # 超参数优化（可选）
    best_alpha = alpha
    if optimize_alpha and len(np.unique(y_err)) > 1:
        # 尝试不同的 alpha 值
        alpha_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
        cv = StratifiedKFold(n_splits=min(n_splits, 5), shuffle=True, random_state=seed)
        best_score = -np.inf
        
        for a in alpha_candidates:
            base = BernoulliNB(alpha=a)
            # 使用 AP (Average Precision) 作为评估指标
            scorer = make_scorer(average_precision_score, needs_proba=True)
            scores = cross_val_score(base, X_work, y_err, cv=cv, scoring=scorer, n_jobs=-1)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = a
        
        print(f"[Alpha Optimization] Best alpha: {best_alpha:.2f} (AP: {best_score:.4f})")
    
    # NB 不需要标准化
    base = Pipeline(steps=[
        ("nb", BernoulliNB(alpha=best_alpha))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    calib = CalibratedClassifierCV(
        estimator=base,
        method=calibrate_method,
        cv=cv,
        n_jobs=-1,  # 使用并行加速
    )
    calib.fit(X_work, y_err)

    return GateModelNB(model=calib, feature_names=selected_features)


def gate_predict_proba_nb(gate: GateModelNB, X: pd.DataFrame) -> pd.Series:
    p = gate.model.predict_proba(X)[:, 1]
    return pd.Series(p, index=X.index, name="p_err")


def eval_gate(y_true: pd.Series, p_err: pd.Series) -> Dict[str, float]:
    y = y_true.values.astype(int)
    p = p_err.values.astype(float)

    out = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["ap"] = float("nan")

    out["brier"] = float(brier_score_loss(y, p))
    out["mean_p_err"] = float(np.mean(p))
    out["pos_rate"] = float(np.mean(y))
    return out


def fit_gate_nb_from_step_df(
    df_step: pd.DataFrame,
    col_case: str = "test_case_id",
    col_step: Optional[str] = None,
    col_terminal: Optional[str] = None,
    evidence_cols=("E1_gui", "E2_code", "E4_noresp"),
    col_agent_pred: str = "agent_testcase_score",
    col_gt: str = "phi",
    alpha: float = 1.0,
    thr: float = 0.5,
    calibrate_method: str = "sigmoid",
    optimize_alpha: bool = True,      # 是否优化 alpha 参数
    feature_selection: bool = False,  # 是否进行特征选择
    n_features: Optional[int] = None, # 特征选择时保留的特征数
):
    # 1) 聚合到 case features
    X_case, y_err, _ = build_case_features(
        df_step,
        col_case=col_case,
        col_step=col_step,
        col_terminal=col_terminal,
        evidence_cols=list(evidence_cols),
        col_agent_pred=col_agent_pred,
        col_gt=col_gt,
    )

    # 2) 二值化（BernoulliNB 最稳）
    X_bin = binarize_features(X_case, thr=thr)

    # 3) 训练 NB gate + 校准（带优化选项）
    gate = train_gate_model_nb(
        X_bin,
        y_err,
        alpha=alpha,
        calibrate_method=calibrate_method,
        optimize_alpha=optimize_alpha,
        feature_selection=feature_selection,
        n_features=n_features,
    )

    # 4) 预测与评估
    p_err = gate_predict_proba_nb(gate, X_bin)
    metrics = eval_gate(y_err, p_err)

    report = X_case.copy()
    report["y_err"] = y_err
    report["p_err"] = p_err 
    
    print("[NB Gate metrics]", metrics)
    return gate, report


def save_gate_model(gate: GateModelNB, filepath: str | Path) -> None:
    """
    保存训练好的 gate 模型到本地文件
    
    参数:
        gate: 训练好的 GateModelNB 模型
        filepath: 保存路径（.pkl 文件）
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(gate, f)
    
    print(f"Gate model saved to: {filepath}")


def load_gate_model(filepath: str | Path) -> GateModelNB:
    """
    从本地文件加载训练好的 gate 模型
    
    参数:
        filepath: 模型文件路径（.pkl 文件）
    
    返回:
        GateModelNB: 加载的模型对象
    """
    import sys
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # 修复pickle加载时的模块路径问题
    # 如果模型是在__main__中保存的，需要确保GateModelNB类可用
    # 创建一个自定义的Unpickler来修复模块路径
    class GateUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 如果模块是__main__且类名是GateModelNB，则从gate模块导入
            if module == '__main__' and name == 'GateModelNB':
                return GateModelNB
            return super().find_class(module, name)
    
    # 同时确保__main__模块中有GateModelNB（以防万一）
    if '__main__' in sys.modules:
        if not hasattr(sys.modules['__main__'], 'GateModelNB'):
            sys.modules['__main__'].GateModelNB = GateModelNB
    
    with open(filepath, 'rb') as f:
        unpickler = GateUnpickler(f)
        gate = unpickler.load()
    
    print(f"Gate model loaded from: {filepath}")
    return gate


def infer_gate_nb_from_step_df(
    gate: GateModelNB,
    df_step: pd.DataFrame,
    col_case: str = "test_case_id",
    col_step: Optional[str] = None,
    col_terminal: Optional[str] = None,
    evidence_cols=("E1_gui", "E2_code", "E4_noresp"),
    col_agent_pred: str = "agent_testcase_score",
    col_gt: Optional[str] = "phi",  # 推理时可以没有 gt
    thr: float = 0.5,
) -> pd.DataFrame:
    """
    使用训练好的 gate 模型对新数据进行推理
    
    参数:
        gate: 训练好的 GateModelNB 模型
        df_step: step-level dataframe（新数据）
        col_case: case ID 列名
        col_step: step 顺序列名（可选）
        col_terminal: terminal 标记列名（可选）
        evidence_cols: 证据列名列表
        col_agent_pred: agent 预测列名
        col_gt: ground truth 列名（可选，推理时可能没有）
        thr: 特征二值化阈值
    
    返回:
        report: case-level dataframe，包含特征和预测的 p_err
    """
    # 1) 聚合到 case features（与训练时相同的逻辑）
    # 如果推理时没有 gt，build_case_features 仍然可以工作（gt 列会被设为默认值）
    if col_gt and col_gt not in df_step.columns:
        # 如果没有 gt 列，创建一个全 0 的列（仅用于特征构建，不影响预测）
        df_step = df_step.copy()
        df_step[col_gt] = 0
    
    X_case, y_err, gt_case = build_case_features(
        df_step,
        col_case=col_case,
        col_step=col_step,
        col_terminal=col_terminal,
        evidence_cols=list(evidence_cols),
        col_agent_pred=col_agent_pred,
        col_gt=col_gt,
    )
    
    # 2) 二值化（与训练时相同的阈值）
    X_bin = binarize_features(X_case, thr=thr)
    
    # 3) 确保特征顺序与训练时一致
    # gate.feature_names 保存了训练时的特征顺序
    missing_features = set(gate.feature_names) - set(X_bin.columns)
    if missing_features:
        raise ValueError(
            f"Missing features in inference data: {missing_features}. "
            f"Expected features: {gate.feature_names}"
        )
    
    # 按训练时的特征顺序排列
    X_bin = X_bin[gate.feature_names]
    
    # 4) 预测
    p_err = gate_predict_proba_nb(gate, X_bin)
    
    # 5) 构建报告
    report = X_case.copy()
    report["p_err"] = p_err
    
    # 如果有 gt，也加入 y_err 用于评估
    if col_gt and col_gt in df_step.columns:
        report["y_err"] = y_err
        report["gt"] = gt_case
    
    return report

import pandas as pd
import numpy as np

# df_step: step-level dataframe
# 必须包含：case_id, agent_pred, gt(phi), p_err, p_agentfail(可选), decision(可选)

def build_case_table(df_step, case_col="test_case_id"):
    g = df_step.groupby(case_col, sort=False)

    case_df = pd.DataFrame({
        "p_err": g["p_err"].mean(),  # p_err 是 case-level 的话，这里 mean/max 都一样
        "agent_pred": g["agent_pred"].agg(lambda x: int(x.mean() >= 0.5)),
        "gt": g["phi"].agg(lambda x: int(x.mean() >= 0.5)),
    })

    # 判错标记
    case_df["is_error"] = (case_df["agent_pred"] != case_df["gt"]).astype(int)

    # 如果你已经算了 EM 归因 posterior
    if "p_agentfail" in df_step.columns:
        case_df["p_agentfail"] = g["p_agentfail"].mean()

    # 如果你已经有决策结果
    if "decision" in df_step.columns:
        case_df["decision"] = g["decision"].agg(lambda x: x.value_counts().index[0])

    return case_df.reset_index()

def bucketize_p_err(case_df):
    bins = [0.0, 0.2, 0.4, 0.6, 1.0]
    labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,1.0]"]

    case_df = case_df.copy()
    case_df["p_err_bucket"] = pd.cut(
        case_df["p_err"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return case_df

def sanity_check_report(case_df):
    rows = []

    for bucket, sub in case_df.groupby("p_err_bucket", dropna=False):
        if len(sub) == 0:
            continue

        row = {
            "bucket": bucket,
            "n_cases": len(sub),
            "mean_p_err": sub["p_err"].mean(),
            "error_rate": sub["is_error"].mean(),  # 真实错误率
        }

        # EM 归因分布（如果有）
        if "p_agentfail" in sub.columns:
            row["mean_p_agentfail"] = sub["p_agentfail"].mean()
            row["agentfail_rate(p>0.5)"] = (sub["p_agentfail"] > 0.5).mean()

        # 决策分布（如果有）
        if "decision" in sub.columns:
            row["flip_rate"] = (sub["decision"] == "FLIP").mean()
            row["abstain_rate"] = (sub["decision"] == "ABSTAIN").mean()
            row["keep_rate"] = (sub["decision"] == "KEEP").mean()

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ========== 训练阶段 ==========
    df_train = pd.read_excel("/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_train_filter.xlsx")
    print("Training data columns:", df_train.columns.tolist())
    
    # 训练 gate 模型
    gate, report_train = fit_gate_nb_from_step_df(df_train)
    print("\nTraining report shape:", report_train.shape)
    print("Training report columns:", report_train.columns.tolist())

    # ========== 保存模型 ==========
    model_path = Path("/data/hongsirui/em_adaptive_learning/src/em_adaptive_learning/gate_model.pkl")
    save_gate_model(gate, model_path)
    
    # ========== 推理阶段 ==========
    # 方式1：使用训练好的模型直接推理
    df_test = pd.read_excel("/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_test.xlsx")
    report_test = infer_gate_nb_from_step_df(
        gate=gate,
        df_step=df_test,
        col_case="test_case_id",
        evidence_cols=("E1_gui", "E2_code", "E4_noresp"),
        col_agent_pred="agent_testcase_score",
        col_gt="phi",
        thr=0.5,
    )
    print("\nInference report shape:", report_test.shape)
    print("Predicted p_err:\n", report_test["p_err"].describe())
    
    # 方式2：从文件加载模型进行推理（示例）
    # gate_loaded = load_gate_model(model_path)
    # report_test_loaded = infer_gate_nb_from_step_df(
    #     gate=gate_loaded,
    #     df_step=df_test,
    #     col_case="test_case_id",
    #     evidence_cols=("E1_gui", "E2_code", "E4_noresp"),
    #     col_agent_pred="agent_testcase_score",
    #     col_gt="phi",
    #     thr=0.5,
    # )

    # ========== 分析阶段 ==========
    # 1) 从 step-level 构造 case-level
    case_df = build_case_table(report_train)

    # 2) 分桶
    case_df = bucketize_p_err(case_df)

    # 3) 统计
    report_stats = sanity_check_report(case_df)
    print("\nSanity check report:")
    print(report_stats)