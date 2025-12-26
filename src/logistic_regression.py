"""
逻辑回归模型用于根因分析
使用与EM算法相同的数据处理方式，但采用监督学习方法
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path: str) -> pd.DataFrame:
    """加载数据文件，支持xlsx和csv格式"""
    path = Path(path)
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix in [".csv"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file: {path}")
    return df


def make_train_val_split(df: pd.DataFrame,
                         case_col: str = "test_case_id",
                         val_ratio: float = 0.2,
                         seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 case 粒度划分 train/val"""
    rng = np.random.default_rng(seed)
    case_ids = df[case_col].astype(str).unique()
    rng.shuffle(case_ids)
    n_val = int(len(case_ids) * val_ratio)
    val_ids = set(case_ids[:n_val])
    train_ids = set(case_ids[n_val:])
    train_df = df[df[case_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[case_col].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


class LogisticRegressionRootCause:
    """
    逻辑回归模型用于根因分析

    两类根因 δ ∈ {0=EnvFail, 1=AgentFail}
    step 级证据: E1_gui, E2_code, E4_noresp
    case 级证据: agent_testcase_score_x

    与EM算法的差异:
    - 使用监督学习方式，需要标签(phi)
    - 在case级别聚合特征后进行训练
    - 输出P(AgentFail)概率，P(EnvFail) = 1 - P(AgentFail)
    """

    def __init__(self,
                 seed: int = 42,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 class_weight: str = 'balanced'):
        """
        初始化逻辑回归模型

        Args:
            seed: 随机种子
            C: 正则化强度的倒数
            max_iter: 最大迭代次数
            class_weight: 类别权重，'balanced'表示自动平衡
        """
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        # 列名配置
        self.col_case = "test_case_id"
        self.col_gui = "E1_gui"
        self.col_code = "E2_code"
        self.col_noresp = "E4_noresp"
        self.col_agent = "agent_testcase_score_x"
        self.col_label = "phi"

    def _binarize(self, x: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """将连续值二值化"""
        x = np.asarray(x, float)
        x = np.clip(x, 0.0, 1.0)
        uniq = np.unique(x[~np.isnan(x)])
        if set(uniq).issubset({0.0, 1.0}):
            return x
        return (x >= thresh).astype(float)

    def _aggregate_case_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将step级别特征聚合到case级别

        聚合策略:
        - mean: 平均值
        - max: 最大值
        - sum: 总和
        - count: 步骤数量
        """
        # 确保证据列存在并二值化
        df = df.copy()
        df[self.col_gui] = self._binarize(df[self.col_gui].values)
        df[self.col_code] = self._binarize(df[self.col_code].values)
        if self.col_noresp in df.columns:
            df[self.col_noresp] = self._binarize(df[self.col_noresp].values)
        else:
            df[self.col_noresp] = 0.0

        # 按case聚合
        agg_dict = {
            self.col_gui: ['mean', 'max', 'sum'],
            self.col_code: ['mean', 'max', 'sum'],
            self.col_noresp: ['mean', 'max', 'sum'],
        }

        # 聚合特征
        case_features = df.groupby(self.col_case).agg(agg_dict)
        case_features.columns = ['_'.join(col).strip()
                                 for col in case_features.columns.values]
        case_features = case_features.reset_index()

        # 添加步骤数量
        step_counts = df.groupby(
            self.col_case).size().reset_index(name='step_count')
        case_features = case_features.merge(step_counts, on=self.col_case)

        # 添加agent判断 (取每个case的最后一个非nan值)
        agent_scores = df.groupby(self.col_case)[self.col_agent].apply(
            lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
        ).reset_index()
        agent_scores.columns = [self.col_case, 'agent_score']
        case_features = case_features.merge(agent_scores, on=self.col_case)

        # 添加标签 (取每个case的最后一个非nan值)
        if self.col_label in df.columns:
            labels = df.groupby(self.col_case)[self.col_label].apply(
                lambda x: x.dropna(
                ).iloc[-1] if len(x.dropna()) > 0 else np.nan
            ).reset_index()
            labels.columns = [self.col_case, 'label']
            case_features = case_features.merge(labels, on=self.col_case)

        return case_features

    def _get_feature_matrix(self, case_features: pd.DataFrame) -> np.ndarray:
        """从case特征DataFrame中提取特征矩阵"""
        feature_cols = [
            f'{self.col_gui}_mean', f'{self.col_gui}_max', f'{self.col_gui}_sum',
            f'{self.col_code}_mean', f'{self.col_code}_max', f'{self.col_code}_sum',
            f'{self.col_noresp}_mean', f'{self.col_noresp}_max', f'{self.col_noresp}_sum',
            'step_count', 'agent_score'
        ]
        self.feature_names = feature_cols

        X = case_features[feature_cols].values
        # 填充nan为0
        X = np.nan_to_num(X, nan=0.0)
        return X

    def fit(self,
            df: pd.DataFrame,
            col_case: str = "test_case_id",
            col_agent: str = "agent_testcase_score_x",
            col_label: str = "phi",
            col_gui: str = "E1_gui",
            col_code: str = "E2_code",
            col_noresp: str = "E4_noresp"):
        """
        训练逻辑回归模型

        Args:
            df: 训练数据DataFrame
            col_case: case ID列名
            col_agent: agent判断列名
            col_label: 标签列名
            col_gui: GUI证据列名
            col_code: 代码证据列名
            col_noresp: 无响应证据列名
        """
        # 绑定列名
        self.col_case = col_case
        self.col_agent = col_agent
        self.col_label = col_label
        self.col_gui = col_gui
        self.col_code = col_code
        self.col_noresp = col_noresp

        # 聚合case级别特征
        case_features = self._aggregate_case_features(df)

        # 过滤掉没有标签的样本
        case_features = case_features[case_features['label'].notna()]

        # 提取特征和标签
        X = self._get_feature_matrix(case_features)
        y = case_features['label'].values.astype(int)

        print(f"Training on {len(y)} cases")
        print(
            f"Class distribution: EnvFail(0)={sum(y==0)}, AgentFail(1)={sum(y==1)}")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 训练逻辑回归
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
            solver='lbfgs'
        )
        self.model.fit(X_scaled, y)

        # 打印训练结果
        train_proba = self.model.predict_proba(X_scaled)[:, 1]
        train_pred = (train_proba >= 0.5).astype(int)
        train_acc = (train_pred == y).mean()
        print(f"Training accuracy: {train_acc:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预测概率

        Args:
            df: 输入数据DataFrame

        Returns:
            case_features: case级别特征DataFrame
            proba: 概率数组 [:,0]=P(EnvFail), [:,1]=P(AgentFail)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # 聚合case级别特征
        case_features = self._aggregate_case_features(df)

        # 提取特征
        X = self._get_feature_matrix(case_features)
        X_scaled = self.scaler.transform(X)

        # 预测概率
        proba = self.model.predict_proba(X_scaled)

        return case_features, proba

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if self.model is None:
            return {}

        params = {
            "coefficients": {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.model.coef_[0])
            },
            "intercept": float(self.model.intercept_[0]),
            "C": self.C,
            "class_weight": self.class_weight,
            "classes": self.model.classes_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        return params

    def load_params(self, params: Dict[str, Any]):
        """加载模型参数"""
        # 恢复特征名
        self.feature_names = list(params["coefficients"].keys())

        # 创建模型
        self.C = params.get("C", 1.0)
        self.class_weight = params.get("class_weight", "balanced")

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
            solver='lbfgs'
        )

        # 设置模型参数
        n_features = len(self.feature_names)
        self.model.coef_ = np.array(
            [[params["coefficients"][name] for name in self.feature_names]])
        self.model.intercept_ = np.array([params["intercept"]])
        self.model.classes_ = np.array(params["classes"])

        # 恢复scaler
        self.scaler.mean_ = np.array(params["scaler_mean"])
        self.scaler.scale_ = np.array(params["scaler_scale"])
        self.scaler.n_features_in_ = n_features

    def save_model(self, path: str):
        """保存模型到文件"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'C': self.C,
            'class_weight': self.class_weight,
        }, path)

    def load_model(self, path: str):
        """从文件加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.C = data['C']
        self.class_weight = data['class_weight']


def analyze_flips(val_df: pd.DataFrame, out_dir: Optional[str] = None):
    """区分两类子集: 原判断正确 vs 错误，查看flip比例"""
    df = val_df.copy()
    df["is_correct_before"] = (df["human_gt"] == df["agent_original"])
    df["is_correct_after"] = (df["human_gt"] == df["corrected_label"])
    df["is_flipped"] = df["agent_original"] != df["corrected_label"]

    # subset A: 原判断正确
    subset_A = df[df["is_correct_before"]]
    misflipped = subset_A[subset_A["is_flipped"]]
    if len(subset_A) > 0:
        print(f"[Subset A] 原判断正确: {len(subset_A)} cases, "
              f"被误翻 {len(misflipped)} ({len(misflipped)/len(subset_A):.2%})")
    else:
        print(
            f"[Subset A] 原判断正确: {len(subset_A)} cases, 被误翻 {len(misflipped)}")

    # subset B: 原判断错误
    subset_B = df[~df["is_correct_before"]]
    corrected = subset_B[subset_B["is_correct_after"]]
    if len(subset_B) > 0:
        print(f"[Subset B] 原判断错误: {len(subset_B)} cases, "
              f"被成功纠正 {len(corrected)} ({len(corrected)/len(subset_B):.2%})")
    else:
        print(
            f"[Subset B] 原判断错误: {len(subset_B)} cases, 被成功纠正 {len(corrected)}")

    # 打印confusion matrix
    print("\n=== Confusion Matrix (GT vs Corrected) ===")
    print(pd.crosstab(df["human_gt"], df["corrected_label"],
                      rownames=["GT"], colnames=["Corrected"]))


def confusion_matrix(val_df: pd.DataFrame):
    """统计人类标注 vs 矫正后结果"""
    conf_matrix = pd.crosstab(val_df["human_gt"], val_df["corrected_label"],
                              rownames=["GT"], colnames=["Corrected"])
    print("=== Confusion Matrix (GT vs Corrected) ===")
    print(conf_matrix)
    acc = (val_df["human_gt"] == val_df["corrected_label"]).mean()
    print(f"Accuracy after correction: {acc:.3f}")
    return conf_matrix


def correct_agent_judgment_lr(df: pd.DataFrame,
                              lr_model: LogisticRegressionRootCause,
                              tau_agentfail: float = 0.75,
                              tau_envfail: float = 0.75,
                              col_case: str = "test_case_id",
                              col_agent: str = "agent_testcase_score_x",
                              col_label: str = "phi") -> pd.DataFrame:
    """
    使用逻辑回归模型对agent判断进行矫正

    逻辑与EM版本一致:
    - 若 agent 判 FAIL (0) 且 P_AgentFail >= tau_agentfail → flip_to_AgentFail
    - 若 agent 判 PASS (1) 且 P_EnvFail >= tau_envfail → flip_to_EnvFail

    Args:
        df: 输入数据DataFrame
        lr_model: 训练好的逻辑回归模型
        tau_agentfail: AgentFail翻转阈值
        tau_envfail: EnvFail翻转阈值
        col_case: case ID列名
        col_agent: agent判断列名
        col_label: 标签列名

    Returns:
        矫正结果DataFrame
    """
    # 获取case级别预测
    case_features, proba = lr_model.predict_proba(df)

    # 获取每个case的ground truth
    gt_map = df.groupby(col_case)[col_label].apply(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
    ).to_dict()

    rows = []
    for idx, row in case_features.iterrows():
        cid = row[col_case]
        C_case = row['agent_score']
        P_EnvFail = proba[idx, 0]
        P_AgentFail = proba[idx, 1]

        gt = gt_map.get(cid, np.nan)

        # 处理agent原判
        if pd.isna(C_case):
            C_case_int = None
        else:
            C_case_int = int(C_case)

        action = "keep_AgentJudge"
        corrected = C_case_int

        if C_case_int is not None:
            # 场景1: agent 判 FAIL (0)，我们要区分 Env vs Agent
            if C_case_int == 0:
                if P_AgentFail >= tau_agentfail:
                    corrected = 1  # 认定 AgentFail
                    action = "flip_to_AgentFail"
                elif P_EnvFail >= tau_envfail:
                    corrected = 0  # 环境问题，维持
                    action = "keep_EnvFail"

            # 场景2: agent 判 PASS (1)，当强 EnvFail 证据时 flip
            elif C_case_int == 1:
                if P_EnvFail >= tau_envfail:
                    corrected = 0
                    action = "flip_to_EnvFail"

        rows.append({
            "case_id": cid,
            "human_gt": gt,
            "agent_original": C_case_int,
            "P_case_EnvFail": P_EnvFail,
            "P_case_AgentFail": P_AgentFail,
            "corrected_label": corrected,
            "action": action
        })

    val_df = pd.DataFrame(rows).sort_values("case_id")

    # 计算准确率
    valid_df = val_df[val_df["agent_original"].notna() &
                      val_df["human_gt"].notna()]
    if len(valid_df) > 0:
        acc_original = (valid_df["agent_original"] ==
                        valid_df["human_gt"]).mean()
        print(f"Original accuracy: {acc_original:.4f}")

        acc_correct = (valid_df["corrected_label"] ==
                       valid_df["human_gt"]).mean()
        print(f"Corrected accuracy: {acc_correct:.4f}")

        analyze_flips(valid_df)
        confusion_matrix(valid_df)

    return val_df


def main(args):
    """主函数：加载数据、训练模型、评估结果"""
    # 加载数据
    df = load_data(args.data_path)

    # 检查必要列
    for col in ["E1_gui", "E2_code"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # 补充缺失列
    if "E4_noresp" not in df.columns:
        df["E4_noresp"] = 0.0
    if "agent_testcase_score_x" not in df.columns:
        df["agent_testcase_score_x"] = df.get("agent_testcase_score", np.nan)
    if "phi" not in df.columns:
        raise ValueError("Missing label column: phi")

    # 划分训练集和验证集
    train_df, val_df = make_train_val_split(
        df,
        case_col="test_case_id",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"#cases train={train_df['test_case_id'].nunique()}, "
          f"val={val_df['test_case_id'].nunique()}")
    print(f"#rows  train={len(train_df)}, val={len(val_df)}")

    # 创建并训练逻辑回归模型
    lr_model = LogisticRegressionRootCause(
        seed=args.seed,
        C=args.C,
        max_iter=args.max_iter,
        class_weight='balanced' if args.balanced else None,
    )

    lr_model.fit(
        train_df,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        col_label="phi",
        col_gui="E1_gui",
        col_code="E2_code",
        col_noresp="E4_noresp",
    )

    # 获取并保存参数
    params = lr_model.get_params()
    print("\nModel parameters:")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存参数
    with open(out_dir / "lr_params.json", 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"\nParams saved to {out_dir / 'lr_params.json'}")

    # 保存模型
    lr_model.save_model(str(out_dir / "lr_model.joblib"))
    print(f"Model saved to {out_dir / 'lr_model.joblib'}")

    # 验证集预测
    print("\n" + "="*50)
    print("Validation Results:")
    print("="*50)

    val_case_features, val_proba = lr_model.predict_proba(val_df)

    # 保存step级别预测（实际上是case级别，但保持与EM输出格式一致）
    val_pred = val_case_features[[lr_model.col_case]].copy()
    val_pred["P_EnvFail"] = val_proba[:, 0]
    val_pred["P_AgentFail"] = val_proba[:, 1]
    val_pred.to_csv(out_dir / "val_pred_step.csv", index=False)

    # 矫正判断
    val_correct = correct_agent_judgment_lr(
        val_df,
        lr_model,
        tau_agentfail=args.tau_agentfail,
        tau_envfail=args.tau_envfail,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        col_label="phi",
    )

    # 保存矫正结果
    val_correct.to_csv(out_dir / "val_corrected_cases.csv", index=False)
    print(f"\nResults saved to {out_dir}")


def run_prediction(df_path: str, out_dir: str, params_path: str = None, args=None):
    """
    使用已有模型进行预测

    Args:
        df_path: 数据文件路径
        out_dir: 输出目录
        params_path: 模型文件路径（.joblib格式），如果为None则从out_dir中查找
        args: 命令行参数
    """
    # 加载数据
    df = load_data(df_path)
    print(f"Loaded {len(df)} rows")

    # 确保必要的列存在
    for col in ["E1_gui", "E2_code"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if "E4_noresp" not in df.columns:
        df["E4_noresp"] = 0.0
    if "agent_testcase_score_x" not in df.columns:
        df["agent_testcase_score_x"] = df.get("agent_testcase_score", np.nan)
    if "test_case_id" not in df.columns:
        raise ValueError("Missing column: test_case_id")

    # 加载模型
    if params_path is None:
        params_path = Path(out_dir) / "lr_model.joblib"
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Model file not found: {params_path}")

    lr_model = LogisticRegressionRootCause()
    lr_model.load_model(str(params_path))
    print(f"Loaded model from {params_path}")

    # 预测
    case_features, proba = lr_model.predict_proba(df)

    # 构建预测结果
    pred_df = case_features[[lr_model.col_case]].copy()
    pred_df["P_EnvFail"] = proba[:, 0]
    pred_df["P_AgentFail"] = proba[:, 1]

    # 从 args 获取 tau 参数
    tau_agentfail = args.tau_agentfail if args is not None else 0.75
    tau_envfail = args.tau_envfail if args is not None else 0.75

    # 矫正判断
    pred_correct = correct_agent_judgment_lr(
        df,
        lr_model,
        tau_agentfail=tau_agentfail,
        tau_envfail=tau_envfail,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        col_label="phi",
    )

    # 保存结果
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path / "pred_step.csv", index=False)
    if pred_correct is not None:
        pred_correct.to_csv(out_path / "pred_corrected_cases.csv", index=False)

    print(f"\nPredictions saved to {out_path}")
    print(f"Case-level predictions: {len(pred_df)} cases")

    return pred_df, pred_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logistic Regression for Root Cause Analysis")
    parser.add_argument("--data_path", type=str,
                        default="em_df_webdevjudge_claude_4_train.xlsx",
                        help="Path to training data file")
    parser.add_argument("--test_path", type=str,
                        default="em_df_webdevjudge_claude_4_test.xlsx",
                        help="Path to test data file")
    parser.add_argument("--params_path", type=str,
                        default=None,
                        help="Path to model file for prediction")
    parser.add_argument("--out_dir", type=str,
                        default="lr_outputs_webdevjudge_claude_4",
                        help="Output directory")

    parser.add_argument("--val_ratio", type=float, default=0.25,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=127,
                        help="Random seed")
    parser.add_argument("--tau_agentfail", type=float, default=0.75,
                        help="Threshold for flipping to AgentFail")
    parser.add_argument("--tau_envfail", type=float, default=0.75,
                        help="Threshold for flipping to EnvFail")

    # 逻辑回归特有参数
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum number of iterations")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Use balanced class weights")

    args = parser.parse_args()

    # 训练和验证（验证集即测试集，结果直接在main中输出）
    main(args)
